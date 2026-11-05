# why initialization matters

when training deep neural networks, choosing how weights are initialized dictates whether activations flow stably or collapse into numerical chaos. without proper scaling, the variance of forward activations grows linearly with layer width, causing exponential explosion or vanishing gradients across deep architectures. this lesson investigates the mathematical foundations of matrix multiplication ($Y = X @ W$), deriving exact variance propagation rules and demonstrating why variance-preserving scaling (such as xavier and he initialization) is essential for stable gradient descent and preventing catastrophic overflows.

## variance definition and useful form

variance measures the dispersion of a random variable around its expected value. by definition:

$$\text{Var}(A) = E[(A - E[A])^2]$$

expanding this expression using the linearity of expectation yields the computational form:

$$\text{Var}(A) = E[A^2] - (E[A])^2$$

similarly, covariance between two random variables $a$ and $b$ is defined as $\text{Cov}(a, b) = E[ab] - E[a]E[b]$.

## variance of sums and products

for any two random variables $a$ and $b$:

$$\text{Var}(a + b) = \text{Var}(a) + \text{Var}(b) + 2\text{Cov}(a, b)$$

when $a$ and $b$ are independent, $\text{Cov}(a, b) = 0$, so variances simply add: $\text{Var}(a + b) = \text{Var}(a) + \text{Var}(b)$. for the product of independent zero-mean variables $a$ and $b$:

$$\text{Var}(ab) = \text{Var}(a)\text{Var}(b)$$

for a scaled variable $ka$, constants are pulled out squared: $\text{Var}(ka) = k^2 \text{Var}(a)$, while adding a constant $c$ leaves variance unchanged: $\text{Var}(a + c) = \text{Var}(a)$.

## the linear layer setup: $Y = X @ W$

consider a standard linear layer activation where input vector $x$ has shape $(1 \times d_1)$ with i.i.d. entries $x_i \sim \mathcal{N}(0, 1)$, and weight matrix $w$ has shape $(d_1 \times d_2)$ with i.i.d. entries $w_{ij} \sim \mathcal{N}(0, 1)$ independent of $x$.

each output element $y_j$ is computed as the dot product:

$$y_j = \sum_{i=1}^{d_1} x_i w_{ij}$$

### mean and variance of $y_j$

since $x_i$ and $w_{ij}$ are independent zero-mean variables, $E[x_i w_{ij}] = 0$, leading to $E[y_j] = 0$. for the variance, each product term has $\text{Var}(x_i w_{ij}) = \text{Var}(x_i)\text{Var}(w_{ij}) = 1 \times 1 = 1$. because the $d_1$ terms are uncorrelated, their variances sum up:

$$\text{Var}(y_j) = d_1$$

it is important to note that the distribution of $y_j$ is **not a normal gaussian**. each product term $x_i w_{ij}$ is the product of two normals (a scale mixture of normals with heavy tails), meaning $y_j$ before taking the limit is technically non-gaussian. nevertheless, because finite variance terms combine under the central limit theorem as $d_1$ grows, we can still robustly compute and rely on its exact mean ($0$) and variance ($d_1$).

this reveals the core problem: **the output variance grows linearly with input dimension $d_1$**. with wide layers (e.g. $d_1 = 1024$), activation standard deviations reach $32$, causing exponential overflow or vanishing across deep networks.

## conclusion: the lecun solution

to counteract the linear growth of activation variance with input dimension $d_1$, we scale the weight distribution. by applying lecun initialization with scaling factor $k = \frac{1}{\sqrt{d_1}}$, we ensure that $\text{Var}(y_j) = d_1 \times \left(\frac{1}{\sqrt{d_1}}\right)^2 = 1$. this variance-preserving transformation stabilizes forward propagation, keeping activation variances at exactly $1$ across arbitrary layer widths.

## summary table

| quantity | result | condition |
| :--- | :--- | :--- |
| `Var(a)` | $E[a^2] - (E[A])^2$ | always |
| `Var(a + b)` | $\text{Var}(a) + \text{Var}(b) + 2\text{Cov}(a,b)$ | always |
| `Var(a + b)` | $\text{Var}(a) + \text{Var}(b)$ | independent |
| `Var(ab)` | $\text{Var}(a)\text{Var}(b)$ | independent, zero-mean |
| `Var(ka)` | $k^2 \text{Var}(a)$ | always |
| `E[y_j]` ($x, w \sim \mathcal{N}(0,1)$) | $0$ | zero-mean inputs |
| `Var(y_j)` ($w \sim \mathcal{N}(0,1)$) | $d_1$ | unscaled weights |
| `Var(y_j)` ($w \sim \mathcal{N}(0,1/d_1)$) | $1$ | variance-preserving |

---
initialization basics — darth-c0d3r
