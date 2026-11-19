# on distributed training & inference paradigms

## overview

deep learning models have gotten progressively larger over the last decade. with the proliferation of large language models, making efficient use of hardware is more important than ever for optimising cost, and in many cases, even getting things to run. for large models, you can't simply download the weights and start inference / training on your gpu without worrying about their memory footprint. a couple of days ago, thinking machines released their latest multi-modal model `Inkling-2-Small` - the _small_ in the title had me excited for a moment. all the excitement came crashing down when i saw the parameter count - 594B (!!!). at half precision (bf16), this model would be 1.2 TB. a single h100 gpu has "only" 80 GB HBM - you'd need at least 15 of these just to hold the weights. the practical way to start using the modern frontier-scale models is through distributed multi-gpu setups, which will be the focus of our discussion.

this article takes a theoretical and technical approach - we'll define the formal framework for gpu communication, look at various paradigms for distributing a model across gpu nodes and clusters, and examine what kind of distributed setup works best under different conditions and requirements. 

## gpu communication

a distributed setup, where the model weights and intermediate results are scattered across devices, will require them to communicate and coordinate. how the data moves and its throughput depends on the connections between those devices: are they connected via high bandwidth nv-links, or do they communicate over a network? similarly, the connection topology also impacts the throughput by dictating which paths exist and how contended they are: are the gpus linked through the host cpu, directly point-to-point, or through an nv-switch that gives full-bandwidth any-to-any connectivity?

understanding the hardware connections in depth, however, is a topic for another day, and frankly not my area of expertise (yet). in this section, we're going to discuss the api-level functions exposed by libraries like nccl for inter-device communication. these primitives are agnostic to the underlying hardware topology and are universally available. however, the knowledge of the hardware topology remains crucial to make decisions on choosing the parallelism strategy, and to understand the bottlenecks that can hurt our system's performance.

### communication primitives

a communication primitive is a basic operation for moving data between devices, with precisely defined semantics: who holds what before, and who holds what after. they come in two kinds: point-to-point operations between a pair of gpus, and collectives, where a whole group of gpus participates together. every parallelism strategy we’ll look at is built out of these, so it’s worth getting comfortable with them before moving on.

#### send / recv

send / recv is the simplest primitive: one gpu copies a tensor into the memory of another. the sender keeps its data, the receiver ends up with an identical copy, and nobody else is involved. both sides must agree on the shape and dtype, and every send needs a matching recv. as we'll see later, this is the building block for pipeline parallelism, where activations are handed from one stage to the next.

#### all-reduce

all-reduce is a collective primitive with every device in a group participating in the transaction. at the start, each participating device holds its own value of the tensor, of the same shape and data type as the others. all-reduce allows all the devices to sync the tensor by applying a reducing function, like sum, product, min, and max to the values. after the operation, every device holds the same tensor, the elementwise reduction across all devices’ inputs. the efficient implementations of this primitive (ring / tree) deserve their own discussion, but is out of scope for the current one. all-reduce is used to sync gradients across gpus in data parallelism (we'll cover this later).

#### all-gather

all-gather is another collective, but where all-reduce combines values, all-gather assembles them. at the start, each participating device holds a part of the full tensor. a concatenation of the tensors from each device (in rank order), along an axis, gives us the entire tensor. typically, the parts are disjoint shards that together make up the full tensor. after the operation, every device holds the same concatenated tensor. for e.g., say we have 4 devices in the group containing `d, a, b` shaped slice each. then after the all-gather operation along axis-0, each device holds the full `4d, a, b` tensor.  

#### broadcast

broadcast is a one-to-many collective: one designated device, the root, sends its tensor to every other device in the group. before the operation only the root holds the data, and afterwards every device holds an identical copy. semantically, it is equivalent to individual send/recv operations between the source device and every target device, but nccl exposes this as a separate primitive to allow for more optimized data transfer (in a ring). it’s typically used to get all devices to the same starting state, for example when the weights loaded on one device need to be distributed to every other device in the group.

#### reduce-scatter

reduce-scatter starts like all-reduce: each device holds its own tensor of the same shape, and the tensors are aggregated elementwise with a reduction function. the difference is where the result ends up. instead of every device receiving the full reduced tensor, each device receives only one shard of it (the i-th chunk goes to device i), which is the same state all-gather starts from. the result is equivalent to an all-reduce where each device then discards everything except its shard, but it’s cheaper, since the full result is never materialized on every device. it’s used in sharded data-parallel schemes like zero and fsdp: each device computes gradients on its own mini-batch, and a reduce-scatter sums them while leaving each device with only the gradients for the parameters it owns (more on that later).

an all-reduce is equivalent to a reduce-scatter followed by an all-gather: the reduce-scatter leaves each device with one fully reduced shard, and the all-gather then shares those shards so every device has the complete result. this is also how efficient implementations like ring all-reduce work in practice.

#### all-to-all

all-to-all is a collective in which every device sends a different shard of its tensor to every other device. each device splits its tensor into one chunk per device, sends chunk j to device j, and ends up holding the chunks addressed to it from everyone else (in order of source device rank). if you imagine the chunks as a grid of source device by destination device, all-to-all transposes it. unlike all-gather, the devices end with different data, and unlike reduce-scatter, nothing is reduced: it’s pure routing. it’s the primitive behind expert parallelism, where tokens are dispatched to the devices holding their assigned experts and then returned (to be covered later).

## data parallelism

data parallelism is used when the model being trained is small enough to fit inside a single gpu. in that case, having an identical copy of the model on multiple gpus (n), and splitting the mini-batch into n parts allows us to both train the model faster (because the overall task is now divided among n devices), and also unlock higher batch sizes (if the maximum possible batch size for a single gpu's memory is b, splitting the mini-batch allows an overall batch size of n\*b), thus improving the throughput.

only once at the beginning of a training run, the model is copied to all devices using the broadcast operation. while training on a mini-batch, the data is split into n equal shards and each device completes its own forward and backward pass. after the backward pass on each device is complete, the gradients are synced using the all-reduce primitive and the local copies of the model and the optimizer states held by each device are updated in parallel. this ensures consistency across devices. after this synchronization, the distributed system is ready for the next mini-batch.

a common optimization of data parallelism is that during the backward pass, the gradient synchronization starts as soon as the gradients are available for a layer or a group of layers. say, the backward pass is finished for the final layer on each device and the gradients are available. there's no need for the system to wait for the rest of the backward pass to finish; the synchronization can start immediately. therefore, in data parallelism, apart from the initial model broadcast, the communication hides behind computation for the most part.

observe carefully that merely being able to fit the model weights on each device is not sufficient: each device must also be able to hold its copies of the gradient, the optimizer states, and a master copy (full-precision) of the model if using mixed-precision training. there are ways to sidestep some of these requirements by trading-in lower memory usage at the cost of more communications across devices by using techniques like fsdp or deepspeed-zero - both of which we will cover later.

pytorch allows two ways to use data parallelism: dp (`nn.DataParallel`) and ddp (`nn.DistributedDateParallel`). pytorch's own documentation rightfully suggests using the latter but it's still worth looking at the differences. in data parallel, one gpu (rank 0) acts as the coordinator and does the extra computation (splitting data, computing loss, averaging gradients, and updating the model). the updated model is broadcast at the start of each forward pass, and the other devices stay idle while device 0 does the extra work. distributed data parallel is a truly distributed system where all devices share the workload and data transfers are not funneled through a single device. there's almost no reason to use data parallel over distributed data parallel.

since data parallelism needs only one gradient all-reduce per step, and most of it overlaps with the backward pass, it tolerates relatively low-bandwidth links well, such as the network between nodes (infiniband or ethernet). this is why data parallelism is usually the strategy that spans nodes, while communication-heavy strategies like tensor parallelism are kept inside a node on nvlink.

for inference, data parallelism is simple: there's no need for gradient synchronization, we can just host multiple instances of the model and run them independently or behind a load-balancer. it's often just called replication, though some serving frameworks do use the term data parallelism.

## model parallelism

when the memory required to train a model (which includes weights, gradients, and optimizer states) surpasses that of a single device, we can leverage multiple devices and combine them to make the training run feasible. in model parallelism, the model is split horizontally, and the model's layers are arranged sequentially along a series of devices. for instance, for a 12-layer model in a system with 4 gpus, the layers 0-1-2 could sit on gpu-0, the layers 3-4-5 could sit on gpu-1, and all the way up to the layers 9-10-11 which could sit on the final gpu-3. this is of course a simple example: the layers don't have to be equally split - although we would want each gpu device to get a similar amount of workload for maximum usage. in addition to the model weights, each device will also only hold the gradients and optimizer states for the corresponding layers. this cuts the memory requirement per device by a factor of number of devices (n).

during the forward pass, device-0 gets the complete mini-batch. it passes the data through its layers, and forwards the output of the final layer held by it onto the next device for the rest of the forward computation. each device will hold on to at least some of the intermediate activations (depending upon the amount of activation checkpointing used) for the gradient computation during the backward pass. the final device containing the final layers of the model will compute the loss and initiate the backpropagation. the backward pass traverses the devices in reverse order, computing the gradients as it goes and then updating the corresponding optimizer states and model weights. the intermediate activations stored on a device can be flushed as soon as the backward pass crosses it. once the pass reaches the first device, and its update completes, the system is ready for the next mini-batch.

in terms of communication, there are only 2\*(n-1) send/recv operations per step. communication in model parallelism is light: only the activations at each stage boundary move forward, and the matching gradients move backward, so the volume per step is small compared to the full gradient all-reduce of data parallelism. this makes it tolerant of slower links, and it is commonly used across nodes over infiniband or ethernet. the catch is latency: the stages run strictly one after another, so each device waits on the previous one, and any transfer delay lands directly on the critical path with nothing to overlap it. data parallelism can hide its all-reduce behind the backward pass, but naive model parallelism has no such cushion. efficiency is one of the biggest drawbacks of model parallelism - at any time, only a single gpu is actively involved in computation while the others sit idle, thus capping its utilization at 1/n. this idle time is known as a pipeline bubble. a related parallelism technique called pipeline parallelism addresses this concern which we will cover soon.

## pipeline parallelism

pipeline parallelism builds upon model parallelism and aims to reduce the pipeline bubbles caused by gpus sitting idle. it does so by splitting a mini-batch into m micro-batches, thereby increasing throughput and device efficiency. we start with a mini-batch made up of m micro-batches (each of which could be as large as the largest batch a single gpu allows - which increases the effective batch size by a factor of m); we'll call the micro-batches b_0, b_1, ..., b_(m-1). similar to model parallelism, b_0 starts on device 0; however, as b_0 moves forward to device 1, instead of keeping device 0 idle, we start processing the next micro-batch on device 0. b_1 trails b_0 and leads other micro-batches through the stages. this ensures that between the initial ramp-up and the final cool-down, all devices are kept busy.

after the first micro-batch finishes through all the stages in the forward direction, it is ready for the backward pass. however, the previous devices are still busy with the forward passes of the trailing micro-batches. we have the option of either waiting for all the forward passes to finish and then commencing the backward passes, or commencing the backward pass as soon as the forward pass finishes for a micro-batch, and handling the scheduling of the competing forward and backward passes with a pre-defined policy. the 2 strategies come from 2 commonly used scheduling protocols: `gpipe` and `1f1b` (introduced in pipe-dream).

in the `gpipe` scheduling policy (introduced by google), the system waits for all the forward passes to complete before triggering the backward pass in the reverse order of micro-batches. the intermediate activations for each micro-batch are held by each device for its layers, and are freed when the backward pass for the corresponding micro-batch passes over. after all the micro-batches have been through a device, the accumulated gradients can now be used for the step update. after the update, the held gradients are zeroed as the device readies itself for the next batch. at its peak, each device holds activations for all m micro-batches in the step, which can make the gpipe scheduling memory constrained at larger number of micro-batches.

by contrast, the `1f1b` scheduling policy doesn't wait for all m micro-batches to finish their forward passes, and begins the backward pass as soon as the forward pass for a micro-batch reaches the last stage. this means that at each stage, a device will have 2 jobs: one forward and one backward pass; the 1f1b policy schedules them alternately. this interweaving of forward and backward passes means that each device only has to hold a maximum of n sets of activations (instead of m, which depends on the number of micro-batches and can blow up). in doing so, this strategy relieves a critical memory constraint of the gpipe strategy, at the same efficiency.

### efficiency computations

#### g-pipe scheduling

```python
# number of devices = n
# number of mini-batches = m

# the data flow and usage for forward and backward passes is identical
# so it suffices to compute efficiency for forward pass only

# i-th batch starts at t = i
# i-th batch ends at t = i + n (because there are n stages)
# final batch, m-1, ends at t = m + n - 1 (this is the total time taken)

# busy time for each device = m
# ==> free time for each device = m + n - 1 - m = n - 1

# bubble ratio = free time for a device / total time for process = (n - 1) / (m + n - 1)
```

#### 1f1b scheduling

```python
# number of devices = n
# number of mini-batches = m

# the first batch reaches the final stage at t = n - 1
# after that, there are n sequential forward and backward passes at final device
# ==> the final device is done with its work at t = (n - 1) + (3 * m) (1 for forward and 2 for backward pass)
# after this, the batch goes through n-1 steps of backward passes (and is the final batch)
# ==> job completes at t = (n - 1) + (3 * m) + (2 * (n - 1)) = 3 * (m + n - 1)

# busy time for each device = 3 * m
# ==> free time for each device = 3 * (n - 1)

# bubble ratio = free time for a device / total time for process = (n - 1) / (m + n - 1)
```