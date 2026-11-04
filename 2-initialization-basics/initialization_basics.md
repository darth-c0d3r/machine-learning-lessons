# Why Initialization Matters: Mean and Variance of X @ W

## 1. Variance: definition and useful form

**Definition:**

Var(A) = E[(A - E[A])^2]

**Derivation of the useful form.** Let mu = E[A]. Expand the square and use linearity of expectation:

Var(A) = E[A^2 - 2*mu*A + mu^2]
       = E[A^2] - 2*mu*E[A] + mu^2
       = E[A^2] - 2*mu^2 + mu^2
       = E[A^2] - mu^2

**Var(A) = E[A^2] - (E[A])^2**

Covariance works the same way:

Cov(A, B) = E[(A - E[A])(B - E[B])] = E[AB] - E[A]E[B]

---

## 2. Variance of a sum

Var(A + B) = E[(A+B)^2] - (E[A+B])^2
           = E[A^2] + 2E[AB] + E[B^2] - (E[A] + E[B])^2
           = (E[A^2] - E[A]^2) + (E[B^2] - E[B]^2) + 2(E[AB] - E[A]E[B])

**Var(A + B) = Var(A) + Var(B) + 2 Cov(A, B)**

Note the factor of **2** on the covariance term. For independent A, B we have Cov(A, B) = 0, so:

**Var(A + B) = Var(A) + Var(B)** (independent)

For n terms: Var(sum A_i) = sum Var(A_i) + 2 * sum_{i<j} Cov(A_i, A_j). If all terms are pairwise independent, only the first sum survives.

---

## 3. Variance of a product (independent A, B)

Var(AB) = E[A^2 B^2] - (E[AB])^2
        = E[A^2]E[B^2] - (E[A]E[B])^2          (independence)
        = (Var(A) + E[A]^2)(Var(B) + E[B]^2) - E[A]^2 E[B]^2

**Var(AB) = Var(A)Var(B) + Var(A)E[B]^2 + Var(B)E[A]^2**

For **zero-mean** independent A, B the last two terms vanish:

**Var(AB) = Var(A) * Var(B)**

---

## 4. Variance of a scaled variable

Var(kA) = E[k^2 A^2] - (k E[A])^2 = k^2 (E[A^2] - E[A]^2)

**Var(kA) = k^2 Var(A)**

Constants come out **squared**. Adding a constant does not change variance: Var(A + c) = Var(A).

---

## 5. The setup: Y = X @ W

- X has shape (1 x d1), every entry X_i ~ N(0, 1), i.i.d.
- W has shape (d1 x d2), every entry W_ij ~ N(0, 1), i.i.d., independent of X.
- Y = X @ W has shape (1 x d2).

Each output element is a sum of d1 products:

**Y_j = sum_{i=1}^{d1} X_i * W_ij**

### Mean of Y_j

Each term X_i W_ij is a product of independent variables, so E[X_i W_ij] = E[X_i] E[W_ij] = 0. By linearity:

**E[Y_j] = 0**

### Variance of Y_j

Each term X_i W_ij is a product of independent zero-mean variables, so

Var(X_i W_ij) = Var(X_i) Var(W_ij) = 1 * 1 = 1

The d1 terms are mutually uncorrelated (they involve different X_i and different W_ij; for terms i != k, Cov = E[X_i X_k W_ij W_kj] = 0). So the variances add:

**Var(Y_j) = d1**

### Is Y_j normal?

**Not exactly.** Reasons:

- Each term $X_i W_{ij}$ is a *product* of two normals. That distribution is not normal; it is peaked at 0 with heavy tails (kurtosis 9 vs. 3 for a normal).
- Conditioned on $X$, $Y_j$ is exactly normal: $Y_j \mid X \sim \mathcal{N}(0, \|X\|^2)$. But $\|X\|^2$ is itself random (chi-squared with $d_1$ degrees of freedom), so $Y_j$ is a **scale mixture of normals**, which has heavier tails than a normal.
- Its kurtosis is $3 + \frac{6}{d_1}$, i.e. excess kurtosis $\frac{6}{d_1}$.

However, although the distribution of $Y_j$ is **not a normal Gaussian**, by the **Central Limit Theorem**, a sum of $d_1$ independent-ish terms with finite variance approaches a normal as $d_1$ grows, and the excess kurtosis $\frac{6}{d_1}$ shrinks to 0. Thus, we can still compute its mean and variance reliably:

**$Y_j \sim \text{approximately } \mathcal{N}(0, d_1)$** (variance $d_1$, not standard deviation), with the approximation improving as $d_1$ grows.

Yes, the answer is N_approx(0, d1). The mean is fine, but the variance **grows linearly with the width d1**, so the standard deviation grows like sqrt(d1). With d1 = 1024, outputs already have std of 32, and stacking layers compounds this exponentially. Activations explode (or, with the opposite scaling, vanish).

---

## 6. The fix: scale W

We want Var(Y_j) = 1. Scale W by a constant k, so the weights are k * W:

Var(Y_j) = k^2 * d1 = 1  =>  k = 1 / sqrt(d1)

**W_scaled = W / sqrt(d1)**, equivalently W ~ N(0, 1/d1).

(Note: this is *dividing by sqrt(d1)*, or equivalently *multiplying by 1/sqrt(d1)*. Dividing by 1/sqrt(d1) would make the variance blow up to d1^2.)

Check: Var(Y_j) = d1 * Var(W/sqrt(d1)) * Var(X) = d1 * (1/d1) * 1 = 1, and the mean stays 0.

### Naming

This variance-preserving rule, Var(W) = 1/fan_in, is the idea behind **Xavier (Glorot) initialization** and is exactly what **LeCun initialization** uses. The full Xavier/Glorot formula is Var(W) = 2 / (fan_in + fan_out), which also keeps the *backward* pass stable; it equals 1/d1 only when d1 = d2.
Note that for ReLU networks, half of the activations are zeroed, which halves the variance, so He initialization uses Var(W) = 2/fan_in instead.

---

## Summary table

| Quantity | Result | Condition |
|---|---|---|
| Var(A) | E[A^2] - (E[A])^2 | always |
| Var(A + B) | Var(A) + Var(B) + 2Cov(A,B) | always |
| Var(A + B) | Var(A) + Var(B) | independent |
| Var(AB) | Var(A)Var(B) | independent, zero-mean |
| Var(kA) | k^2 Var(A) | always |
| E[Y_j], X,W ~ N(0,1) | 0 | |
| Var(Y_j), X,W ~ N(0,1) | d1 | |
| Var(Y_j), W ~ N(0, 1/d1) | 1 | |
