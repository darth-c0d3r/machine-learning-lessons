# project 2 — why initialization matters: mean and variance of $X @ W$

this lesson explores the mathematical foundations of neural network weight initialization, deriving the mean and variance of matrix multiplications ($Y = X @ W$) and explaining why improper scaling leads to exploding or vanishing activations.

## structure

- `index.html` — interactive lesson with mathematical derivations, variance rules, and interactive simulation.
- `initialization_basics.md` — raw notes and derivations.

## key topics

1. **variance definitions & useful form**: $\text{Var}(A) = E[A^2] - (E[A])^2$.
2. **variance of sums & products**: rules for independent variables and zero-mean conditions.
3. **the matrix multiplication setup**: $Y = X @ W$ where $X, W \sim \mathcal{N}(0, 1)$.
4. **variance growth**: proving that $\text{Var}(Y_j) = d_1$ (growing linearly with input dimension).
5. **the fix**: scaling weights by $\frac{1}{\sqrt{d_1}}$ (Xavier / He initialization principles).

---
darth-c0d3r
