# transformer arithmetic & resource estimation

## overview

understanding the computational and memory requirements of transformer models is essential for training llms efficiently, sizing gpu clusters, estimating training time, estimating inference latency and throughput, and provisioning inference infrastructure.

model computations are typically expressed in **FLOPs** (floating point operations). hardware accelerators like gpus specify their compute capacity in **FLOPs/s** (floating point operations per second). peak compute throughput varies significantly across data types, where lower-bit precision formats (such as FP8, INT8, or FP4) deliver higher tensor-core throughput than standard FP16 or BF16, which in-turn deliver higher throughput than FP32.

---

## numerical data types in deep learning

floating-point representations determine how numbers are stored in memory and processed by hardware accelerators. understanding the bit-level anatomy of data types is critical for balancing memory footprint against numerical stability.

### bit-level breakdown

* **`float32`** (32-bit = 4 bytes): `1 sign + 8 exponent + 23 fraction`
* **`float16`** (16-bit = 2 bytes): `1 sign + 5 exponent + 10 fraction`
* **`bfloat16`** (16-bit = 2 bytes): `1 sign + 8 exponent + 7 fraction`
* **`fp8_e4m3`** (8-bit = 1 byte): `1 sign + 4 exponent + 3 fraction`
* **`fp8_e5m2`** (8-bit = 1 byte): `1 sign + 5 exponent + 2 fraction`

newer hardware architectures, such as NVIDIA H100 GPUs, introduce 8-bit floating-point formats.

### anatomy of a floating-point number

the general mathematical value of a floating-point number is expressed as:

$$\text{value} = (-1)^S \times \left(1 + \frac{M}{2^m}\right) \times 2^{E - \text{bias}}$$

where the fractional term $\left(1 + \frac{M}{2^m}\right)$ ranges from $1$ (inclusive) to $2$ (exclusive).

### range vs. precision trade-off

* **exponent ($E$)**: determines the **dynamic range** (how high or low $2^x$ can go). having a high range for the exponent allows us to represent smaller numbers better and prevents underflow.
* **fraction / mantissa ($M$)**: determines the **precision or resolution** (the granularity between representable numbers).

`bfloat16` shares the same 8-bit exponent width as `float32`, giving it an identical dynamic range (making training numerically stable without frequent underflows), albeit at a lower precision due to the 7-bit mantissa.

inspecting numerical properties in python can be done via `torch.finfo(torch.float32)`.

### mixed precision training & data type roles

* 16-bit data types are generally sufficient for feed-forward computations.
* however, training operations such as storing optimizer states (e.g., AdamW momentum/variance) and accumulating gradient values require 32-bit precision to prevent catastrophic cancellation and rounding errors.
* **Mixed Precision Training** bridges this gap by executing forward and backward passes in 16-bit while maintaining master weights and optimizer updates in `float32`.

### intuition on FP32 vs. BF16 in mixed precision training

think of `bfloat16` as something which is **transitory**. forward-pass values which are computed and used once are fine in `bf16`. but values that are being accumulated over time—such as gradient updates—must be maintained in `float32`.
