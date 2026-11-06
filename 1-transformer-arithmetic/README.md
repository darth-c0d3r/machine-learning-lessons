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

---

## counting parameters and flops: linear projection

with the numerical data types established, we now turn to resource estimation. as our first building block, we examine a standard linear projection (matrix multiplication).

### setup

consider an input matrix $X$ of shape $(\text{bs} \times d_1)$, where $\text{bs}$ represents the batch size (or number of tokens) and $d_1$ is the input feature dimension. we project this input through a weight matrix $W$ of size $(d_1 \times d_2)$ to produce an output matrix $Y = X @ W$ of shape $(\text{bs} \times d_2)$ (ignoring bias for simplicity).

### number of parameters

the learnable parameters of the linear layer reside entirely within the weight matrix $W$. since $W$ has dimensions $d_1 \times d_2$, the total parameter count is:

$$\text{parameters} = d_1 \times d_2$$

### number of FLOPs (forward pass)

each element in the output matrix $Y$ is computed as a dot product of a length-$d_1$ row of $X$ and a length-$d_1$ column of $W$. a single dot product of two vectors of length $d_1$ requires $d_1$ multiplications and $d_1 - 1$ additions, which counts as $2d_1$ floating-point operations (FLOPs).

```text
Input X (bs × d₁)       Weight W (d₁ × d₂)         Output Y (bs × d₂)
┌───────────────┐       ┌───────────────┐          ┌───────────────┐
│               │       │   Col j       │          │       ·       │
│  Row i        │  ×    │   ↓           │    =     │     (i, j)    │  ← Y[i,j] = Row_i(X) • Col_j(W)
│  ───►         │       │   [  ]        │          │       ·       │
│               │       │   [  ]        │          │               │
└───────────────┘       └───────────────┘          └───────────────┘
```

since output matrix $Y$ has dimensions $\text{bs} \times d_2$, there are $\text{bs} \times d_2$ individual dot products to compute. multiplying the operations per dot product by the total number of output elements gives:

$$\text{FLOPs} = \text{bs} \times d_2 \times 2d_1 = 2 \times \text{bs} \times d_1 \times d_2$$

*note:* this is a simple introductory exercise, but we will be using similar calculations throughout to derive parameter counts and FLOPs for the entire transformer model (including attention projections, feed-forward networks, and layer norms).

---

## full transformer model architecture

building upon our linear projection building block, the complete decoder-only transformer architecture stacks multiple identical layers together, transforming token inputs into vocabulary logits.

### architectural flow

```text
       [ Language Modeling Head ]  (linear projection to vocab size)
                  ▲
                  │
        [ Final Layer Norm ]
                  ▲
                  │
          ┌───────┴───────┐
          │ × num_layers  │  (stacked transformer decoder blocks)
          └───────┬───────┘
                  │
         ┌────────┴────────────────────────────────────────┐
         │ Block per layer (top to bottom):                │
         │   • Feed Forward Network (FFN) ◄─(Residual 2)   │
         │   • Layer Norm 2                                │
         │   • Self-Attention Block       ◄─(Residual 1)   │
         │   • Layer Norm 1                                │
         └────────┬────────────────────────────────────────┘
                  ▲
                  │
       [ Token Embeddings + Positional Embeddings ]
                  ▲
                  │
           [ Input Tokens ]
```

### core components

1. **token & positional embeddings (bottom)**: converts input token IDs into continuous vectors of dimension $d_{\text{model}}$ and adds positional information.
2. **stacked decoder layers (`num_layers`)**: multiple identical blocks repeating the sequence:
   * **layer norm 1** $\rightarrow$ **self-attention** with **residual connection 1**.
   * **layer norm 2** $\rightarrow$ **feed-forward network (FFN)** with **residual connection 2**.
3. **language modeling head (top)**: projects the final hidden states back to vocabulary logits to predict next tokens.


## transformer inference: parameter and flop breakdown

### introduction

this document counts the parameters and inference FLOPs of a decoder-only transformer, module by module. inference has two stages:

- **prefill:** the whole prompt ($s$ tokens per sequence) is processed in parallel and the kv cache is filled. every weight matrix is read from memory once and reused across all $b \times s$ tokens, so each byte of weights supports a large amount of arithmetic (high arithmetic intensity). the large matmuls keep the compute units busy, so prefill is **compute-bound**. its latency is the time to first token (ttft).
- **decode:** one new token per sequence is generated per step, attending to the cached keys and values. every step must still read all the weights (and the whole kv cache), but each weight is used for only $b$ tokens, about $2 \times b$ flops per parameter. with small $b$ the arithmetic per byte moved is very low, so decode is **memory-bandwidth-bound**: step time is set by how fast weights and the kv cache can be streamed from memory, not by flops. its latency is the time per output token (tpot).

#### notation and assumptions

| symbol | meaning |
|---|---|
| $b$ | batch size |
| $s$ | number of prompt tokens (prefill length) |
| $L$ | number of layers |
| $d_{\text{model}}$ | model width |
| $d_{\text{hidden}}$ | ffn hidden width |
| $V$ | vocabulary size |

- **architecture:** pre-norm blocks (rmsnorm, attention, residual, rmsnorm, ffn, residual), mha, rope, relu ffn with two projections, no biases, untied embeddings.
- **flop convention:** a multiply-add counts as 2 flops. exp, division, and max count as 1 flop each. per-row scalar operations are ignored.
- **prefill:** assumes the full $s \times s$ score matrix is computed and logits are computed for all $s$ positions.
- **decode:** assumes the kv cache stores keys after rope, so cached keys are never re-rotated.

---

### part 1: components outside the layer loop

#### token embedding

- **parameters:** $V \times d_{\text{model}}$
- **flops (prefill):** 0
- **flops (decode):** 0
- **notes:**
  - it is a table lookup, not a matmul.
  - prefill gathers $b \times s$ rows; decode gathers only $b$ rows.
  - if the output projection is tied to this table, count these parameters once.

#### additive positional embedding (only if not using rope)

- **parameters:** max_seq_len $\times d_{\text{model}}$ (learned), 0 (sinusoidal)
- **flops (prefill):** $b \times s \times d_{\text{model}}$
- **flops (decode):** $b \times d_{\text{model}}$
- **notes:**
  - the position lookup is free; only the elementwise add to the token embeddings costs flops.
  - in decode, the new token's position index is $s$.
  - skip this row entirely when the model uses rope.

#### final rmsnorm

- **parameters:** $d_{\text{model}}$
- **flops (prefill):** $4 \times b \times s \times d_{\text{model}}$
- **flops (decode):** $4 \times b \times d_{\text{model}}$
- **notes:**
  - gain vector only.
  - separate from the two per-layer norms.

#### output projection (unembedding)

- **parameters:** $d_{\text{model}} \times V$ (0 if tied)
- **flops (prefill):** $2 \times b \times s \times d_{\text{model}} \times V$
- **flops (decode):** $2 \times b \times d_{\text{model}} \times V$
- **notes:**
  - real matmul; weight tying changes parameter count but not flops.
  - decode reads this weight matrix in full every step.

---

### part 2: components inside each layer (repeated $L$ times)

#### rmsnorm 1 (pre-attention)

- **parameters:** $d_{\text{model}}$
- **flops (prefill):** $4 \times b \times s \times d_{\text{model}}$
- **flops (decode):** $4 \times b \times d_{\text{model}}$
- **notes:**
  - decode normalizes only the new token.

#### self-attention: q, k, v projections

- **parameters:** $3 \times d_{\text{model}} \times d_{\text{model}}$
- **flops (prefill):** $6 \times b \times s \times d_{\text{model}} \times d_{\text{model}}$
- **flops (decode):** $6 \times b \times d_{\text{model}} \times d_{\text{model}}$
- **notes:**
  - decode only projects the new token; cached keys/values are never re-projected.
  - decode reads these weights in full every step, driving memory-bound behavior.

#### Self-attention: RoPE on Q and K

- **Parameters:** 0
- **FLOPs (prefill):** 6 · b · s · nh · d_head
- **FLOPs (decode):** 6 · b · nh · d_head
- **Notes:**
  - Each pair (x₁, x₂) is rotated by a position-dependent angle: x₁' = x₁ · cos − x₂ · sin and x₂' = x₁ · sin + x₂ · cos. That is 4 multiplies and 2 adds per pair, or 3 FLOPs per element.
  - It is applied to Q and K only, not V, so the total is 3 · 2 = 6 per (token, head, d_head) position for MHA. With GQA it is 3 · b · s · d_head · (nh + nkv).
  - The cos/sin tables are precomputed and cached, so no per-pass generation cost is counted.
  - In decode, only the new Q and the new K are rotated, because cached keys were stored already rotated.
  - If an implementation re-rotates the whole cache, decode cost becomes 3 · b · nh · d_head (Q) + 3 · b · s_kv · nh · d_head (K).

#### Self-attention: S = QKᵀ

- **Parameters:** 0
- **FLOPs (prefill):** 2 · b · nh · s² · d_head
- **FLOPs (decode):** 2 · b · nh · s_kv · d_head
- **Notes:**
  - In prefill, each head multiplies (s × d_head) by (d_head × s). The full s × s matrix is counted; causal masking could in principle halve it, but the usual convention ignores that.
  - In decode, a single query row is multiplied against s_kv cached keys, so cost is linear in context length instead of quadratic.
  - Decode must read the whole key cache: b · s_kv · nkv · d_head elements per layer. This KV-cache traffic grows with context and is a second source of memory-boundedness, alongside weight reads.

#### Self-attention: softmax (scale, mask, exp, sum, divide)

- **Parameters:** 0
- **FLOPs (prefill):** 5 · b · nh · s²
- **FLOPs (decode):** 4 · b · nh · s_kv
- **Notes:**
  - Softmax is elementwise over the score matrix S, so there is no d_head factor.
  - The 5 ops per element in prefill are: scale by 1/√d_head, apply the causal mask, exp, sum (reduction), divide by the sum.
  - Decode needs no mask, since the single query may attend to every cached position, leaving 4 ops per element over the s_kv scores per head.
  - Max-subtraction for numerical stability (a compare and a subtract per element) is not counted. Including it adds 2 per element.
  - Many implementations fold 1/√d_head into Q or the weights, and fused kernels such as FlashAttention fuse the mask. Counting them keeps the tally conservative.
  - It is about 1/d_head of the matmul cost, so it is a small correction.

#### Self-attention: O = S·V

- **Parameters:** 0
- **FLOPs (prefill):** 2 · b · nh · s² · d_head
- **FLOPs (decode):** 2 · b · nh · s_kv · d_head
- **Notes:**
  - It has the same shape and cost as QKᵀ: (s × s) @ (s × d_head) per head in prefill, and one row of weights against s_kv cached values in decode.
  - Together with S, the core matmuls total 4 · b · nh · s² · d_head (prefill) and 4 · b · nh · s_kv · d_head (decode).
  - Decode must also read the whole value cache, b · s_kv · nkv · d_head elements per layer. Total KV-cache reads per layer are therefore 2 · b · s_kv · nkv · d_head elements.
  - With GQA the matmul FLOPs stay the same, because queries still use all nh heads, but cache size and traffic shrink by nh / nkv.

#### Self-attention: output projection

- **Parameters:** d_model · nh · d_head
- **FLOPs (prefill):** 2 · b · s · d_model · nh · d_head
- **FLOPs (decode):** 2 · b · d_model · nh · d_head
- **Notes:**
  - It maps the concatenated heads (nh · d_head) back to d_model.
  - Decode applies it to the new token only.
  - With the Q, K, V projections the attention projections total 4 · d_model · nh · d_head parameters and 8 · b · s · d_model · nh · d_head prefill FLOPs.

#### Residual add 1 (around attention)

- **Parameters:** 0
- **FLOPs (prefill):** b · s · d_model
- **FLOPs (decode):** b · d_model
- **Notes:**
  - One elementwise add per element.
  - It is counted here at the layer level, so do not count it again inside the attention block.

#### RMSNorm 2 (pre-FFN)

- **Parameters:** d_model
- **FLOPs (prefill):** 4 · b · s · d_model
- **FLOPs (decode):** 4 · b · d_model
- **Notes:**
  - It is identical in cost to RMSNorm 1.
  - Per layer there are 2 · d_model norm parameters and 8 · b · s · d_model prefill FLOPs across the two norms.

#### FFN: up projection

- **Parameters:** d_model · d_hidden
- **FLOPs (prefill):** 2 · b · s · d_model · d_hidden
- **FLOPs (decode):** 2 · b · d_model · d_hidden
- **Notes:**
  - It maps d_model to d_hidden. The common choice is d_hidden = 4 · d_model.
  - With a gated FFN (SwiGLU) there are two such matrices (gate and up), usually with d_hidden ≈ (8/3) · d_model.
  - In decode, the whole matrix is read to process b tokens, which is memory-bound at small batch.

#### FFN: ReLU

- **Parameters:** 0
- **FLOPs (prefill):** b · s · d_hidden
- **FLOPs (decode):** b · d_hidden
- **Notes:**
  - There is a single ReLU per layer, applied to the hidden activations, so the width is d_hidden, not d_model. Each element costs one max-with-zero.
  - With SwiGLU, replace this with a SiLU (several FLOPs per element) plus an elementwise gate × up multiply of b · s · d_hidden.

#### FFN: down projection

- **Parameters:** d_hidden · d_model
- **FLOPs (prefill):** 2 · b · s · d_model · d_hidden
- **FLOPs (decode):** 2 · b · d_model · d_hidden
- **Notes:**
  - It maps d_hidden back to d_model.
  - The FFN matmuls total 2 · d_model · d_hidden parameters and 4 · b · s · d_model · d_hidden prefill FLOPs. With d_hidden = 4 · d_model that is 8 · d_model² parameters and 16 · b · s · d_model² FLOPs.

#### Residual add 2 (around FFN)

- **Parameters:** 0
- **FLOPs (prefill):** b · s · d_model
- **FLOPs (decode):** b · d_model
- **Notes:**
  - It is the second per-layer residual, one elementwise add per element.
  - It is counted here at the layer level, not inside the FFN.

---

### Part 3: Totals

#### Per layer

- **Parameters:** 4 · d_model · nh · d_head + 2 · d_model · d_hidden + 2 · d_model
- **FLOPs (prefill):**
  8·b·s·d_model·nh·d_head + 6·b·s·nh·d_head + 4·b·nh·s²·d_head + 5·b·nh·s² + 4·b·s·d_model·d_hidden + b·s·d_hidden + 10·b·s·d_model
- **FLOPs (decode):**
  8·b·d_model·nh·d_head + 6·b·nh·d_head + 4·b·nh·s_kv·d_head + 4·b·nh·s_kv + 4·b·d_model·d_hidden + b·d_hidden + 10·b·d_model

The 10 · b · s · d_model term is 2 RMSNorms (8) + 2 residuals (2).

#### Full model

- **Parameters:** V·d_model + L·(4·d_model·nh·d_head + 2·d_model·d_hidden + 2·d_model) + d_model + d_model·V
  - Add max_seq_len · d_model for learned positions. Subtract d_model · V if embeddings are tied.
- **FLOPs (prefill):** L · (per-layer prefill) + 4·b·s·d_model + 2·b·s·d_model·V
  - Add b · s · d_model for an additive positional embedding. Replace s with 1 in the last two terms if only last-position logits are computed.
- **FLOPs (decode):** L · (per-layer decode) + 4·b·d_model + 2·b·d_model·V
  - Add b · d_model for an additive positional embedding.

#### Simplified case: nh · d_head = d_model, d_hidden = 4 · d_model

- **Parameters per layer:** 12 · d_model² + 2 · d_model
- **Prefill FLOPs per layer:** 24·b·s·d_model² + 4·b·s²·d_model + 5·b·nh·s² + 20·b·s·d_model
- **Decode FLOPs per layer:** 24·b·d_model² + 4·b·s_kv·d_model + 4·b·nh·s_kv + 20·b·d_model

The 20 · b · s · d_model linear term is RoPE (6) + ReLU (4) + norms and residuals (10).

#### Reading the totals

- **Prefill:** the 24·b·s·d_model² projection and FFN term dominates at short context. The 4·b·s²·d_model attention-core term overtakes the projections once s exceeds about 6 · d_model.
- **Decode:** per-token FLOPs are small (about 24·b·d_model² per layer plus 2·b·d_model·V for the output projection), but each step reads all the weights plus the full KV cache. That is about 2 FLOPs per parameter per sequence in the batch, which is why decode is memory-bound at small batch sizes and why batching more sequences improves hardware utilization.
- The attention-over-cache term in decode grows linearly with context and overtakes the projection term once s_kv exceeds about 6 · d_model.

#### Part 4: How cost scales with length

| Component | Prefill FLOPs | Decode FLOPs |
|---|---|---|
| Time to first token (TTFT) | $O(s \, d_{\text{model}}^2 + s^2 \, d_{\text{model}})$ | n/a |
| Cost per subsequent token | n/a | $O(d_{\text{model}}^2 + (s + l_2) d_{\text{model}})$ |

- **TTFT (prefill, $s$ tokens):** $O(s \, d_{\text{model}}^2 + s^2 \, d_{\text{model}})$. roughly linear in prompt length while $s$ is small relative to $d_{\text{model}}$ (ffn and projections dominate), turning quadratic once $s$ exceeds about $6 \, d_{\text{model}}$ (self-attention dominates).
- **TPOT (decode, $l_2$-th token):** $O(d_{\text{model}}^2 + (s + l_2) d_{\text{model}})$. flat at short context, then linear in context length once $s + l_2$ exceeds about $6 \, d_{\text{model}}$.

---

### conclusion: How cost scales with length

- **TTFT (prefill, $s$ tokens):** $O(s \, d_{\text{model}}^2 + s^2 \, d_{\text{model}})$. Roughly linear in prompt length while $s$ is small relative to $d_{\text{model}}$, turning quadratic once $s$ exceeds about $6 \, d_{\text{model}}$.
- **TPOT (decode, $l_2$-th token):** $O(d_{\text{model}}^2 + (s + l_2) d_{\text{model}})$. Flat at short context, then linear in context length once $s + l_2$ exceeds about $6 \, d_{\text{model}}$.
- **When $l_2 \ll s$:** The context is $s + l_2 \approx s$ for every generated token, so TPOT is set almost entirely by the prompt, $O(d_{\text{model}}^2 + s \, d_{\text{model}})$, and stays essentially constant across the generation.
- **Where the FLOPs go:** Total request cost is prefill plus $l_2 \times$ TPOT. Since $l_2 \ll s$, prefill carries about $s / l_2$ times the FLOPs of decode, so TTFT dominates.
- **Wall-clock caveat:** Decode is memory-bound. With a long prompt, TPOT rises with $s$ even when few tokens are generated, because KV cache reads eventually exceed weight reads.
- **Contrast, $l_2 \gg s$:** TPOT grows with $l_2$ as the cache fills, and cumulative decode cost becomes quadratic in generated tokens.

---

## training-time flops

training is shaped like prefill. the whole sequence is known in advance (teacher forcing), so all $s$ positions are processed in parallel, with no decode loop and no kv cache. like prefill, it is compute-bound, so time is roughly flops divided by achieved throughput. the cost of one training step has three parts: the forward pass, the backward pass, and the optimizer update.

### forward pass

this costs the same as the prefill stage over the same tokens, with logits computed at every position because the loss needs them. per token, it is about $2n + 4lds$ flops, where $n$ is the parameter count. the $2n$ comes from the weight matmuls, and the $4lds$ comes from the attention core (qkᵀ and s·v). this uses the full-matrix convention and $nh \cdot d_{\text{head}} = d$. this is the same number we saw in inference prefill, substituting $n = 12ld^2$ and multiplying by $s$, we have total flops for a sequence = $24ld^2s + 4lds^2$ (refer to part 3 in inference table).

### backward pass

this costs about 2x the forward pass. for a linear layer $y = xw$, backpropagation needs two matmuls, each the same size as the forward one:

- $dx = dy \, w^\top$, the gradient with respect to the layer's input, which is passed to the previous layer
- $dw = x^\top dy$, the gradient with respect to the weights

the parameter-free attention matmuls (qkᵀ and s·v) follow the same 2x rule, because both of their operands need gradients. elementwise operations (norms, softmax, relu, residuals) have backward passes that cost about their forward cost. they are negligible next to the matmuls, so the 2x holds overall. the embedding gradient is a scatter-add into the table and costs essentially no flops.

### optimizer update

the update is elementwise over the parameters, so it costs $o(n)$ per step, independent of the number of tokens in the batch. plain sgd is about 2 flops per parameter, and adam is roughly 10 (momentum, variance, bias correction, and the update). forward plus backward costs about $6n$ per token, so over $b \cdot s$ tokens the optimizer is a fraction of about $1/(b \cdot s)$ of the total. for any realistic number of tokens per step, it is negligible.

### loss

softmax cross-entropy over the vocabulary costs about $b \cdot s \cdot v$, also small next to the matmuls.

### total

forward plus backward is 3x the forward pass, which gives the rule of thumb:

$$\text{flops per step} \approx 6n \cdot (b s) + 12 \, l \, d \cdot b s^2$$

the first term is the weight matmuls ($2nd$ forward, $4nd$ backward, with $d = b \cdot s$ tokens). the second is the attention core, which is quadratic in sequence length. the two terms are equal at $s \approx 6d$, using $n \approx 12 l d^2$ (see below). below that, $6nd$ is a good estimate, and cost is linear in tokens. above it, which can happen in long-context training, the attention term dominates, and cost per token rises with $s$. with causal masking, an implementation that skips the masked half does about half the attention work, but the full-matrix count is the usual convention.

### why $n \approx 12 l d^2$

per layer, the attention projections (q, k, v, o) contribute $4d^2$ parameters and the two ffn projections with $d_{\text{hidden}} = 4d$ contribute $2 \cdot d \cdot 4d = 8d^2$, so each layer has about $12d^2$ and the model has $n \approx 12 l d^2$ (ignoring embeddings, norms, and the output projection, which are small by comparison).

### activation recomputation

with gradient checkpointing, part of the forward pass is re-run during backward to save memory. full recomputation raises the total to about 4x forward ($8nd$), and selective recomputation falls between 3x and 4x. so the multiplier is 3x without checkpointing and up to 4x with it.

### relation to inference

the 3x figure is relative to a prefill-style forward pass with all-position logits. if inference computes only the last position's logits, the ratio is larger. however the takeaway is that training takes about 3x-4x flops as compared to inference prefill.

### what flops don't capture

- **memory:** backward needs the stored forward activations, which scale with $l \cdot b \cdot s \cdot d$. mixed-precision adam also holds about 16 bytes per parameter (weights, gradients, an fp32 master copy, and two moments).
- **communication:** data-parallel gradient all-reduce and tensor-parallel collectives add time that doesn't show up in flop counts.
- **utilization:** achieved throughput (mfu) is typically well below peak, so wall-clock time is flops divided by peak throughput times mfu.

### aside: gradient checkpointing

the problem: backward-pass needs the full forward pass intermediates. for a linear layer, the gradient computation needs the layer's input, and the non-linearities and softmax need their inputs or outputs too. normally you store all of them during forward, so the activation memory scales with $l . b . s . d$ + $l . b . nh . s^2$ for attention scores if they're materialized too. for large models and long sequences, this is often the largest memory cost.

the trick: as with most optimizations in machine learning, we trade memory with compute. rather than storing all the intermediate values, we only store values at fixed checkpoints, like inputs of each layer, etc. then, during the backward-pass, when we arrive at a layer, we do the entire forward-pass of the layer again to regenerate the intermediate values, and then backpropagate through it.

---

transformer arithmetic & resource estimation — darth-c0d3r
