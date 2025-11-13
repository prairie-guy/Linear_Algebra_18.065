# Problem Sets Related to Lectures and Readings

## MIT 18.065 - Matrix Methods in Data Analysis, Signal Processing, and Machine Learning
### Spring 2018

---

## Course Overview Table

| LEC # | TITLE | Reading | Assignment |
|-------|-------|---------|------------|
| 1 | The Column Space of A Contains All Vectors Ax | Section I.1 | Problem Set I.1 |
| 2 | Multiplying and Factoring Matrices | Section I.2 | Problem Set I.2 |
| 3 | Orthonormal Columns In Q Give Q'Q= I | Section I.5 | Problem Set I.5 |
| 4 | Eigenvalues and Eigenvectors | Section I.6 | Problem Set I.6 |
| 5 | Positive Definite and Semidefinite Matrices | Section I.7 | Problem Set I.7 |
| 6 | Singular Value Decomposition (SVD) | Section I.8 | Problem Set I.8 |
| 7 | Eckart-Young: The Closest Rank k Matrix to A | Section I.9 | Problem Set I.9 |
| 8 | Norms of Vectors and Matrices | Section I.11 | Problem Set I.11 |
| 9 | Four Ways to Solve Least Squares Problems | Section II.2 | Problem Set II.2 Problems 2, 8, 9 |
| 10 | Survey of Difficulties with Ax = b | Intro Ch. 2 | Problem Set II.2 Problems 12 and 17 |
| 11 | Minimizing ‖x‖ Subject to Ax = b | Section I.11 | Problem Set I.11 Problem 6<br>Problem Set II.2 Problem 10 |
| 12 | Computing Eigenvalues and Singular Values | Section II.1 | Problem Set II.1 |
| 13 | Randomized Matrix Manipulation | Section II.4 | Problem Set II.4 |
| 14 | Low Rank Changes in A and Its Inverse | Section III.1 | Problem Set III.1 |
| 15 | Matrices A(t) depending on t / Derivative = dA/dt | Sections III.1–2 | Problem Set III.2 Problems 1, 2, 5 |
| 16 | Derivatives of Inverse and Singular Values | Sections III.1–2 | Problem Set III.2 Problems 3, 12 |
| 17 | Rapidly Decreasing Singular Values | Section III.3 | Problem Set III.3 |
| 18 | Counting Parameters in SVD, LU, QR, Saddle Points | Append., Sec. III.2 | Problem Set III.2 |
| 19 | Saddle Points Continued / Maxmin Principle | Sections III.2, V.1 | Problem Set V.1 Problems 3, 8 |
| 20 | Definitions and Inequalities | Sections V.1, V.3 | Problem Set V.1 Problems 10, 12<br>Problem Set V.3 Problem 3 |
| 21 | Minimizing a Function Step by Step | Sections VI.1, VI.4 | Problem Set VI.1 |
| 22 | Gradient Descent: Downhill to a Minimum | Section VI.4 | Problem Set VI.4 Problems 1, 6 |
| 23 | Accelerating Gradient Descent (Use Momentum) | Section VI.4) | Problem Set VI.4 Problem 5 |
| 24 | Linear Programming and Two-Person Games | Sections VI.2–VI.3 | Problem Set VI.2 Problem 1<br>Problem Set VI.3 Problems 2, 5 |
| 25 | Stochastic Gradient Descent | Section VI.5 | Problem Set VI.5 |
| 26 | Structure of Neural Nets for Deep Learning | Section VII.1 | Problem Set VII.1 |
| 27 | Backpropagation to Find Derivative of the Learning Function | Section VII.2 | Problem Set VII.2 |
| 28 | Computing in Class | Section VII.2 and Appendix 3 | [No Problems Assigned] |
| 29 | [No Video Recorded] | No Readings | [No Problems Assigned] |
| 30 | Completing a Rank-One Matrix / Circulants! | Sections IV.8, IV.2 | Problem Set IV.8<br>Problem Set IV.2 |
| 31 | Eigenvectors of Circulant Matrices: Fourier Matrix | Section IV.2 | Problem Set IV.2 |
| 32 | ImageNet is a CNN / The Convolution Rule | Section IV.2 | Problem Set IV.2 |
| 33 | Neural Nets and the Learning Function | Sections VII.1, IV.10 | Problem Set VII.1<br>Problem Set IV.10 |
| 34 | Distance Matrices / Procrustes Problem / First Project | Sections IV.9, IV.10 | Problem Set IV.9 |
| 35 | Finding Clusters in Graphs / Second Project: Handwriting | Sections IV.6–IV.7 | Problem Set IV.6 |
| 36 | Third Project / Alan Edelman and Julia Language | Sections III.3, VII.2 | [No Problems Assigned] |

---

## Problems for Lecture 1 (from textbook Section I.1)

**1.** Give an example where a combination of three nonzero vectors in **R⁴** is the zero vector. Then write your example in the form Ax = 0. What are the shapes of A and x and 0?

**4.** Suppose A is the 3 by 3 matrix **ones(3, 3)** of all ones. Find two independent vectors x and y that solve Ax = 0 and Ay = 0. Write that first equation Ax = 0 (with numbers) as a combination of the columns of A. Why don't I ask for a third independent vector with Az = 0?

**9.** Suppose the column space of an m by n matrix is all of **R³**. What can you say about m? What can you say about n? What can you say about the rank r?

**18.** If A = CR, what are the CR factors of the matrix [0 A; 0 A]?

---

## Problems for Lecture 2 (from textbook Section I.2)

**2.** Suppose **a** and **b** are column vectors with components a₁,...,aₘ and b₁,...,bₚ. Can you multiply **a** times **b**ᵀ (yes or no)? What is the shape of the answer **ab**ᵀ? What number is in row i, column j of **ab**ᵀ? What can you say about **aa**ᵀ?

**6.** If A has columns **a₁**, **a₂**, **a₃** and B = I is the identity matrix, what are the rank one matrices **a₁b₁ᵀ** and **a₂b₂ᵀ** and **a₃b₃ᵀ**? They should add to AI = A.

---

## Problems for Lecture 3 (from textbook Section I.5)

**2.** Draw unit vectors **u** and **v** that are *not* orthogonal. Show that **w** = **v** − **u**(**u**ᵀ**v**) is orthogonal to **u** (and add **w** to your picture).

**4.** Key property of every orthogonal matrix: ‖Q**x**‖² = ‖**x**‖² for every vector **x**. More than this, show that (Q**x**)ᵀ(Q**y**) = **x**ᵀ**y** for every vector **x** and **y**. So *lengths and angles are not changed by Q*. **Computations with Q never overflow!**

**6.** A **permutation matrix** has the same columns as the identity matrix (in some order). Explain why this permutation matrix and every permutation matrix is orthogonal:

```
P = [0 1 0 0]
    [0 0 1 0]
    [0 0 0 1]
    [1 0 0 0]
```

has orthonormal columns so PᵀP = _____ and P⁻¹ = _____.

When a matrix is symmetric or orthogonal, **it will have orthogonal eigenvectors**. This is the most important source of orthogonal vectors in applied mathematics.

---

## Problems for Lecture 4 (from textbook Section I.6)

**2.** Compute the eigenvalues and eigenvectors of A and A⁻¹. Check the trace!

```
A = [0 2]    and    A⁻¹ = [-1/2  1]
    [1 1]              [ 1/2  0]
```

A⁻¹ has the _____ eigenvectors as A. When A has eigenvalues λ₁ and λ₂, its inverse has eigenvalues _____.

**11.** The eigenvalues of A equal the eigenvalues of Aᵀ. This is because det(A − λI) equals det(Aᵀ − λI). That is true because _____. Show by an example that the eigenvectors of A and Aᵀ are not the same.

**15.** (a) Factor these two matrices into A = XΛX⁻¹:

```
A = [1 2]    and    A = [1 1]
    [0 3]              [3 3]
```

(b) If A = XΛX⁻¹ then A³ = ( )( )( ) and A⁻¹ = ( )( )( ).

---

## Problems for Lecture 5 (from textbook Section I.7)

**3.** For which numbers b and c are these matrices positive definite?

```
S = [1 b]    S = [2 4]    S = [c b]
    [b 9]        [4 c]        [b c]
```

With the pivots in D and multiplier in L, factor each A into LDLᵀ.

**14.** Find the 3 by 3 matrix S and its pivots, rank, eigenvalues, and determinant:

```
[x₁ x₂ x₃] S [x₁]
            [x₂] = 4(x₁ − x₂ + 2x₃)²
            [x₃]
```

**15.** Compute the three upper left determinants of S to establish positive definiteness. Verify that their ratios give the second and third pivots.

**Pivots = ratios of determinants**

```
S = [2 2 0]
    [2 5 3]
    [0 3 8]
```

---

## Problems for Lecture 6 (from textbook Section I.8)

**1.** A symmetric matrix S = Sᵀ has orthonormal eigenvectors **v₁** to **vₙ**. Then any vector **x** can be written as a combination **x** = c₁**v₁** + ⋯ + cₙ**vₙ**. Explain these two formulas:

**x**ᵀ**x** = c₁² + ⋯ + cₙ²        **x**ᵀS**x** = λ₁c₁² + ⋯ + λₙcₙ²

**6.** Find the σ's and **v**'s and **u**'s in the SVD for A = [3 4; 0 5]. Use equation (12).

---

## Problems for Lecture 7 (from textbook Section I.9)

**2.** Find a closest rank-1 approximation to these matrices (L² or Frobenius norm):

```
A = [3 0 0]    A = [0 3]    A = [2 1]
    [0 2 0]        [2 0]        [1 2]
    [0 0 1]
```

**10.** If A is a 2 by 2 matrix with σ₁ ≥ σ₂ > 0, find ‖A⁻¹‖₂ and ‖A⁻¹‖²ₖ.

---

## Problems for Lecture 8 (from textbook Section I.11)

**1.** Show directly this fact about ℓ¹ and ℓ² and ℓ∞ vector norms: ‖**v**‖₂ ≤ ‖**v**‖₁ ‖**v**‖∞.

**7.** A short proof of ‖AB‖ₖ ≤ ‖A‖ₖ ‖B‖ₖ starts from multiplying rows times columns:

|(AB)ᵢⱼ|² ≤ ‖row i of A‖² ‖column j of B‖² is the Cauchy-Schwarz inequality

Add up both sides over all i and j to show that ‖AB‖²ₖ ≤ ‖A‖²ₖ ‖B‖²ₖ.

---

## Problems for Lecture 9 (from textbook Section II.2)

**2.** Why do A and A⁺ have the same rank? If A is square, do A and A⁺ have the same eigenvectors? What are the eigenvalues of A⁺?

**8.** What multiple of **a** = [1; 1] should be subtracted from **b** = [4; 0] to make the result A₂ orthogonal to **a**? Sketch a figure to show **a**, **b**, and A₂.

**9.** Complete the Gram-Schmidt process in Problem 8 by computing **q₁** = **a**/‖**a**‖ and **q₂** = A₂/‖A₂‖ and factoring into QR:

```
[1 4] = [q₁ q₂] [‖a‖    ?   ]
[1 0]           [0   ‖A₂‖]
```

The backslash command A\b is engineered to make A block diagonal when possible.

---

## Problems for Lecture 10 (from textbook Introduction Chapter 2)

**Problems 12 and 17 use four data points b = (0, 8, 8, 20) to bring out the key ideas.**

**12.** With b = 0, 8, 8, 20 at t = 0, 1, 3, 4, set up and solve the normal equations AᵀA**x̂** = Aᵀ**b**. For the best straight line in Figure II.3a, find its four heights pᵢ and four errors eᵢ. What is the minimum squared error E = e₁² + e₂² + e₃² + e₄²?

**17.** Project **b** = (0, 8, 8, 20) onto the line through **a** = (1, 1, 1, 1). Find **x̂** = **a**ᵀ**b**/**a**ᵀ**a** and the projection **p** = **x̂a**. Check that **e** = **b** − **p** is perpendicular to **a**, and find the shortest distance ‖**e**‖ from **b** to the line through **a**.

---

## Problems for Lecture 11 (from textbook Section I.11)

### Problem Set I.11

**6.** The first page of I.11 shows *unit balls* for the ℓ¹ and ℓ² and ℓ∞ norms. Those are the three sets of vectors **v** = (v₁, v₂) with ‖**v**‖₁ ≤ 1, ‖**v**‖₂ ≤ 1, ‖**v**‖∞ ≤ 1.

Unit balls are always convex because of the triangle inequality for vector norms:

If ‖**v**‖ ≤ 1 and ‖**w**‖ ≤ 1 show that ‖(**v** + **w**)/2‖ ≤ 1.

### Problem Set II.2

**10.** What multiple of **a** = [1; 1] should be subtracted from **b** = [4; 0] to make the result A₂ orthogonal to **a**? Sketch a figure to show **a**, **b**, and A₂.

---

## Problem for Lecture 12 (from textbook Section II.1)

These problems start with a bidiagonal n by n backward difference matrix D = I − S. Two tridiagonal second difference matrices are DDᵀ and A = −S + 2I − Sᵀ. The shift S has one nonzero subdiagonal Sᵢ,ᵢ₋₁ = 1 for i = 2, ..., n. A has diagonals −1, 2, −1.

**1.** Show that DDᵀ equals A except that 1 ≠ 2 in their (1, 1) entries. Similarly DᵀD = A except that 1 ≠ 2 in their (n, n) entries.

---

## Problems for Lecture 13 (from textbook Section II.4)

**1.** Given positive numbers a₁, ..., aₙ find positive numbers p₁ ... pₙ so that

p₁ + ⋯ + pₙ = 1  and  V = a₁²/p₁ + ⋯ + aₙ²/pₙ reaches its minimum (a₁ + ⋯ + aₙ)².

The derivatives of L(p, λ) = V − λ(p₁ + ⋯ + pₙ − 1) are zero as in equation (8).

**4.** If M = **11**ᵀ is the n by n matrix of 1's, prove that nI − M is positive semidefinite. Problem 3 was the energy test. For Problem 4, find the eigenvalues of nI − M.

---

## Problems for Lecture 14 (from textbook Section III.1)

**1.** Another approach to (I − **uv**ᵀ)⁻¹ starts with the formula for a geometric series: (1 − x)⁻¹ = 1 + x + x² + x³ + ⋯ Apply that formula when x = **uv**ᵀ = matrix:

(I − **uv**ᵀ)⁻¹ = I + **uv**ᵀ + **uv**ᵀ**uv**ᵀ + **uv**ᵀ**uv**ᵀ**uv**ᵀ + ⋯
                = I + **u**[1 + **v**ᵀ**u** + **v**ᵀ**uv**ᵀ**u** + ⋯]**v**ᵀ.

Take x = **v**ᵀ**u** to see I + **uv**ᵀ/(1 − **v**ᵀ**u**). This is exactly equation (1) for (I − **uv**ᵀ)⁻¹.

**4.** Problem 3 found the inverse matrix M⁻¹ = (A − **uv**ᵀ)⁻¹. In solving the equation M**y** = **b**, we compute only the solution **y** and not the whole inverse matrix M⁻¹. You can find **y** in two easy steps:

**Step 1** Solve A**x** = **b** and A**z** = **u**. Compute D = 1 − **v**ᵀ**z**.

**Step 2** Then **y** = **x** + (**v**ᵀ**x**/D)**z** is the solution to M**y** = (A − **uv**ᵀ)**y** = **b**.

Verify (A − **uv**ᵀ)**y** = **b**. We solved two equations using A, no equations using M.

---

## Problems for Lecture 15 (from textbook Sections III.1-III.2)

**1.** A unit vector **u**(t) describes a point moving around on the unit sphere **u**ᵀ**u** = 1. Show that the velocity vector d**u**/dt is orthogonal to the position: **u**ᵀ(d**u**/dt) = 0.

**2.** Suppose you add a positive semidefinite **rank two** matrix to S. What interlacing inequalities will connect the eigenvalues λ of S and α of S + **uu**ᵀ + **vv**ᵀ?

**5.** Find the eigenvalues of A₃ and A₂ and A₁. Show that they are interlacing:

```
A₃ = [ 1 -1  0]    A₂ = [ 1 -1]    A₁ = [1]
     [-1  2 -1]         [-1  2]
     [ 0 -1  1]
```

---

## Problems for Lecture 16 (from textbook Sections III.1-III.2)

**3.** (a) Find the eigenvalues λ₁(t) and λ₂(t) of A = [2 1; 1 0] + t[1 1; 1 1].

(b) At t = 0, find the eigenvectors of A(0) and verify dλ/dt = **y**ᵀ(dA/dt)**x**.

(c) Check that the change A(t) − A(0) is positive semidefinite for t > 0. Then verify the interlacing law λ₁(t) ≥ λ₁(0) ≥ λ₂(t) ≥ λ₂(0).

**12.** If **x**ᵀS**x** > 0 for all **x** ≠ 0 and C is invertible, why is (C**y**)ᵀS(C**y**) also positive? This shows again that if S has all positive eigenvalues, so does CᵀSC.

---

## Problems for Lecture 17 (from textbook Section III.3)

**2.** Show that the evil **Hilbert matrix** H passes the Sylvester test AH − HB = C

```
Hᵢⱼ = 1/(i + j − 1)    A = (1/2)diag(1, 3, ..., 2n−1)    B = −A    C = ones(n)
```

**6.** If an invertible matrix X satisfies the Sylvester equation AX − XB = C, find a Sylvester equation for X⁻¹.

---

## Problems for Lecture 18 (from textbook Section III.2)

**4.** S is a symmetric matrix with eigenvalues λ₁ > λ₂ > ... > λₙ and eigenvectors **q₁**, **q₂**, ..., **qₙ**. Which i of those eigenvectors are a basis for an i-dimensional subspace Y with this property: The minimum of **x**ᵀS**x**/**x**ᵀ**x** for **x** in Y is λᵢ.

**10.** Show that this 2n × 2n KKT matrix H has n positive and n negative eigenvalues:

```
H = [S    C  ]    S positive definite
    [Cᵀ   0  ]    C invertible
```

The first n pivots from S are positive. The last n pivots come from −CᵀS⁻¹C.

---

## Problems for Lecture 19 (from textbook Sections III.2 and V.1)

**3.** We know: 1/3 of all integers are divisible by 3 and 1/7 of integers are divisible by 7. What fraction of integers will be divisible by 3 or 7 or both?

**8.** Equation (4) gave a second equivalent form for S² (the variance using samples):

S² = 1/(N−1) sum of (xᵢ − m)² = 1/(N−1)[(sum of xᵢ²) − Nm²].

Verify the matching identity for the expected variance σ² (using m = Σ pᵢ xᵢ):

σ² = sum of pᵢ(xᵢ − m)² = (sum of pᵢ xᵢ²) − m².

---

## Problems for Lecture 20 (from textbook Section V.1)

**10.** Computer experiment: Find the average A₁₀₀₀₀₀₀ of a million random 0-1 samples! What is your value of the standardized variable X = (Aₙ − 1/2)/(2√N)?

**12.** For any function f(x) the expected value is E[f] = Σ pᵢ f(xᵢ) or ∫ p(x) f(x) dx (discrete or continuous probability). The function can be x or (x − m)² or x².

If the mean is E[x] = m and the variance is E[(x − m)²] = σ² what is E[x²]?

### Problem for Lecture 20 (from textbook Section V.3)

**3.** A fair coin flip has outcomes X = 0 and X = 1 with probabilities 1/2 and 1/2. What is the probability that X ≥ 2X̄? Show that Markov's inequality gives the exact probability X̄/2 in this case.

---

## Problems for Lecture 21 (from textbook Sections VI.1 and VI.4)

**1.** When is the union of two circular discs a convex set? Or two squares?

**5.** Suppose K is convex and F(**x**) = 1 for **x** in K and F(**x**) = 0 for **x** not in K. Is F a convex function? What if the 0 and 1 are reversed?

---

## Problems for Lecture 22 (from textbook Section VI.4)

**1.** For a 1 by 1 matrix in Example 3, the determinant is just det X = x₁₁. Find the first and second derivatives of F(X) = − log(det X) = − log x₁₁ for x₁₁ > 0. Sketch the graph of F = − log x to see that this function F is convex.

**6.** What is the gradient descent equation **x**ₖ₊₁ = **x**ₖ − sₖ∇f(**x**ₖ) for the least squares problem of minimizing f(**x**) = (1/2)‖A**x** − **b**‖²?

---

## Problem for Lecture 23 (from textbook Section VI.4)

**5.** Explain why projection onto a convex set K is a *contraction* in equation (24). Why is the distance ‖**x** − **y**‖ never increased when **x** and **y** are projected onto K?

---

## Problem for Lecture 24 (from textbook Section VI.2)

**1.** Minimize F(**x**) = (1/2)**x**ᵀS**x** = (1/2)x₁² + 2x₂² subject to Aᵀ**x** = x₁ + 3x₂ = b.

(a) What is the Lagrangian L(**x**, λ) for this problem?

(b) What are the three equations "derivative of L = zero"?

(c) Solve those equations to find **x**\* = (x₁\*, x₂\*) and the multiplier λ\*.

(d) Draw Figure VI.4 for this problem with constraint line tangent to cost circle.

(e) Verify that the derivative of the minimum cost is ∂F/∂b = −λ\*.

### Problems for Lecture 24 (from textbook Section VI.3)

**2.** Suppose the constraints are x₁ + x₂ + 2x₃ = 4 and x₁ ≥ 0, x₂ ≥ 0, x₃ ≥ 0. Find the three corners of this triangle in **R³**. Which corner minimizes the cost **c**ᵀ**x** = 5x₁ + 3x₂ + 8x₃?

**5.** Find the optimal (minimizing) strategy for X to choose rows. Find the optimal (maximizing) strategy for Y to choose columns. What is the payoff from X to Y at this optimal minimax point **x**\*, **y**\*?

```
Payoff matrices:  [1 2]    [1 4]
                  [4 8]    [8 2]
```

---

## Problem for Lecture 25 (from textbook Section VI.5)

**1.** Suppose we want to minimize F(x, y) = y² + (y − x)². The actual minimum is F = 0 at (**x**\*, **y**\*) = (0, 0). Find the gradient vector ∇F at the starting point (x₀, y₀) = (1, 1). For full gradient descent (not stochastic) with step s = 1/2, where is (x₁, y₁)?

---

## Problems for Lecture 26 (from textbook Section VII.1)

**4.** Explain with words or show with graphs why each of these statements about Continuous Piecewise Linear functions (CPL functions) is true:

- **M** The maximum M(x, y) of two CPL functions F₁(x, y) and F₂(x, y) is CPL.
- **S** The sum S(x, y) of two CPL functions F₁(x, y) and F₂(x, y) is CPL.
- **C** If the one-variable functions y = F₁(x) and z = F₂(y) are CPL, so is the composition C(x) = z = (F₂(F₁(x)).

**Problem 7 uses the blue ball, orange ring example on playground.tensorflow.org with one hidden layer and activation by ReLU (not Tanh). When learning succeeds, a white polygon separates blue from orange in the figure that follows.**

**7.** Does learning succeed for N = 4? What is the count r(N, 2) of flat pieces in F(**x**)? The white polygon shows where flat pieces in the graph of F(**x**) change sign as they go through the base plane z = 0. How many sides in the polygon?

---

## Problems for Lecture 27 (from textbook Section VII.2)

**2.** If A is an m by n matrix with m > n, is it faster to multiply A(AᵀA) or (AAᵀ)A?

**5.** Draw a computational graph to compute the function f(x, y) = x³(x − y). Use the graph to compute f(2, 3).

---

## Problem for Lecture 30 (from textbook Section IV.8)

**3.** For a connected graph with M edges and N nodes, what requirement on M and N comes from each of the words *spanning tree*?

### Problem for Lecture 30 (from textbook Section IV.2)

**1.** Find **c** ∗ **d** and **c** ⊛ **d** for **c** = (2, 1, 3) and **d** = (3, 1, 2).

---

## Problems for Lecture 31 (from textbook Section IV.2)

**3.** If **c** ∗ **d** = **e**, why is (Σ cᵢ)(Σ dᵢ) = (Σ eᵢ)? Why was our check successful?
(1 + 2 + 3)(5 + 0 + 4) = (6)(9) = 54 = 5 + 10 + 19 + 8 + 12.

**5.** What are the eigenvalues of the 4 by 4 circulant C = I + P + P² + P³? Connect those eigenvalues to the discrete transform F**c** for **c** = (1, 1, 1, 1). For which three real or complex numbers z is 1 + z + z² + z³ = 0?

---

## Problem for Lecture 32 (from textbook Section IV.2)

**4.** Any two circulant matrices of the same size commute: CD = DC. They have the same eigenvectors **qₖ** (the columns of the Fourier matrix F). Show that the eigenvalues λₖ(CD) are equal to λₖ(C) times λₖ(D).

---

## Problem for Lecture 33 (from textbook Section VII.1)

**5.** How many weights and biases are in a network with m = N₀ = 4 inputs in each feature vector **v₀** and N = 6 neurons on each of the 3 hidden layers? How many activation functions (ReLU) are in this network, before the final output?

### Problem for Lecture 33 (from textbook Section IV.10)

**2.** ‖**x₁** − **x₂**‖² = 9 and ‖**x₂** − **x₃**‖² = 16 and ‖**x₁** − **x₃**‖² = 25 do satisfy the triangle inequality 3 + 4 > 5. Construct G and find points **x₁**, **x₂**, **x₃** that match these distances.

---

## Problem for Lecture 34 (from textbook Sections IV.9 and IV.10)

**1.** Which orthogonal matrix Q minimizes ‖X − YQ‖²ₖ? Use the solution Q = UVᵀ above and also minimize as a function of θ (set the θ-derivative to zero):

```
X = [1 2]    Y = [1 0]    Q = [cos θ  -sin θ]
    [2 1]        [0 1]        [sin θ   cos θ]
```

---

## Problem for Lecture 35 (from textbook Sections IV.6-IV.7)

**1.** What are the Laplacian matrices for a triangle graph and a square graph?

---

## MIT OpenCourseWare

https://ocw.mit.edu

18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning
Spring 2018

For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms
