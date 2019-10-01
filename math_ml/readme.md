# Math for Machine Learning

## From: Hal Daume III

## 1 Calculus

**Calculus** is classically the study of the relationship between variables and their rates of change. However, this is not what we use calculus for. We use **differential calculus** as a method for finding extrema of functions; we use **integral calculus** as a method for probabilistic modeling.

## 1.1 Differential Calculus

Example 1. To be more concrete, a classical statistics problem is  **linear regression**.  Suppose that I have a bunch of points $(x1, y1), (x2, y2), \dots , (xN, yN)$, and I want to t a line of the form $y = mx + b$. If I have
a lot of points, it's pretty unlikely that there is going to be a line that actually passes exactly through all of
them. So we can ask instead for a line $y = mx+b$ that lies as close to the points as possible. See Figure 1.

One easy option is to use squared error as a measure of closeness. For a point $(xn, yn)$ and a line defined
by m and b, we can measure the squared error as $[(mx_n + b) - y_n]2$. That is: our predicted value minus the true value, all squared. We can easily sum all of the point-wise errors to get a total error (which, for some strange reason, we'll call "$J$") of:

$$J(m,b)=\sum^N_{n=1}[(mx_n+b)-y_n]^2    \ \ \ (1) $$

Note that we have written the error $J$ as a function of m and b, since, for any setting of $m$ and $b$, we will
get a different error.
Now, our goal is to find values of m and b that minimize the error. How can we do this? Differential
calculus tells us that the minimum of the J function can be computed by finding the zeros of its derivatives.
(Assuming it is convex: see Section 1.3.)



The derivative of a function at a point is the slope of the function at that point (a derivative is like a velocity). See Figure 2. To be precise, suppose we have a function f that maps real numbers to real velocity
numbers. (That is: $ f : R -> R$; see Section 2). For instance, $f(x) = 3x^2 - e^x$. The derivative of $f$ with
respect to $x$, denoted $\partial{f}/\partial{x}$ $is^2$:

$\frac{\partial{f}}{\partial{x}}(x_0)=lim_{h\rightarrow0}\frac{f(x_0+h) - f(x_0)}{h} \ \ \ (2)$

This essentially says that the derivative of f with respect to $x$, evaluated at a point $x_0$, is the rate of change
of $f$ at $x_0$. It is fairly common to see $\partial{f}/\partial{x}$ denoted by $f{'}$. The disadvantage to this notation is that when
$f$ is a function of multiple variables (such as $J$ in linear regression; see Example 1), then $f{'}$ is ambiguous as to which variable the derivative is being taken with respect to. Nevertheless, when clear from context, we
will also use $f{'}$.
Also regarding notation, if we want to talk about the derivative of a function without naming the function,
we will write something like:

$$\frac{\partial}{\partial{x}}[3x^2-e^x] \ \ \ (3)$$

Or, if we're really trying to save space, will write @x for the derivative with respect to x, yielding: $\partial_x[3x^2-e^x]$.
In case you are a bit rusty taking derivatives by hand, the important rules are given below:

-   Scalar multiplication: $\partial_x[af(x)] = a[\partial_xf(x)]$
-    Polynomials: $\partial_x[x^k] = kx^{k-1}$
-   Function addition: $\partial_x[f(x) + g(x)] = [\partial_xf(x)] + [\partial_xg(x)]$
-   Function multiplication: $\partial_x[f(x)g(x)] = f(x)[\partial_xg(x)] + [\partial_xf(x)]g(x)$
-   Function division: $\partial_x\frac{f(x)}{g(x)}=\frac{[\partial_xf(x)]g(x)-[\partial_xg(x)]f(x)}{[g(x)]^2}$
-   Function composition: $\partial_x[f(g(x))]=[\partial_xg(x)][\partial_x(f)](g(x))$
-   Exponentiation: $\partial_x[e^x]=e^x\  and\ \partial_x[\alpha^x]=log(\alpha)*e^x$
-   Logarithms: $\partial_x[log x] = \frac{1}{x}$
    

Note that throughout this document, $log$ means $natural\ log$ - that is, logarithm base $e$. You may have seen this previously as $ln$, but we do not use this notation. If we intend a log base other than e, we will write,
eg., $log_{10}x$, which can be converted into natural log as $log x/log 10$.

### Exercise 1. Compute derivatives of the following functions:

1. $f(x) = e^x+1$
2. $f(x) = e^{-\frac{1}{2}x^2}$
3. $f(x) = x^ax^{1-a}$
4. $f(x) = (e^x + x^2 + 1/x)3$
5. $f(x) = log(x^2 + x - 1)$
6. $f(x) = \frac{e^x+1}{e^{-x}}$

### Answer1:

1.  $f{'}(x)=e^x$
2.  $f{'}(x)=e^{-1/2x^2}*\partial_x(-1/2x^2)=e^{-1/2x^2}(-x)$
3.  $f{'}(x)=\partial_x(x^a)*x^{(1-a)} + x^a * \partial_x(x^{(1-a)}) \\=ax^{(a-1)}x^{(1-a)} + x^a (1-a)x^{(-a)}\\=a + 1-a=1 \\ f(x)=x $
4.  $f{'}(x)=(e^x+2x-1/x^2)3$
5.  $f{'}(x)=\frac{2x+1}{x^2+x+1}$
6.  $f{'}(x)=\frac{\partial(e^x+1)e^{-x} - (e^x+1)\partial{e^{-x}}}{e^{(-2x)}} \\=\frac{e^xe^{-x}-(e^x+1)(-e^{-x})}{e^{-2x}} \\=\frac{1+1+e^{-x}}{e^{-2x}} \\=2e^{2x}+e^{x} \\ f(x)=e^{2x}+e^{x}$



Example 2. Returning to Example 1, we have a function $J(m, b)$ and we want to compute its derivative
with respect to m and its derivative with respect to $b$. Working through the case for m, we have:

$\begin{aligned} \partial_{m} J(m, b) &=\partial_{m}\left(\sum_{n=1}^{N}\left[\left(m x_{n}+b\right)-y_{n}\right]^{2}\right) \\ &=\sum_{n=1}^{N} \partial_{m}\left[\left(m x_{n}+b\right)-y_{n}\right]^{2} \\ &=\sum_{n=1}^{N}\left[2\left[\left(m x_{n}+b\right)-y_{n}\right]\right] \partial_{m}\left[\left(m x_{n}+b\right)-y_{n}\right] \\ &=\sum_{n=1}^{N}\left[2\left[\left(m x_{n}+b\right)-y_{n}\right]\right] x_{n} \end{aligned}$

Exercise 2. Compute $\partial_bJ(m,b)$
One nice thing about derivatives is that they allow us to find extreme points of functions in a straightforward way. (Usually you can think of an extreme point as a maximum or minimum of a function.) Consider maximum again Figure ??; here, we can easily see that the point at which the function is minimized has a derivative minimum (slope) of zero. Thus, we we can find zeros of the derivative of a function, we can also find minima (or maxima) of that function.

**Example 3**. The example plotted in Figure ?? is of the function $f(x) = 2x^2 - 3x + 1$. We can compute
the derivative of this function as $\partial_xf(x) = 4x - 3$. We equate this to zero $(4x - 3 = 0) $and apply algebra to
solve for x, yielding x = 3=4. As we can see from the plot, this is indeed a minimum of this function.

**Exercise 3**. Using $\partial_mJ$ and $\partial_bJ$ from previous examples and exercises, compute the values of m and b that
minimize the function J, thus solving the linear regression problem!



## 1.2 Integral Calculus

An integral is the "opposite" of a derivative. Its most common use, at least by us, is in computing areas
under a curve. We will never actually have to compute integrals by hand, though you should be familiar
with their properties.

The "area computing" integral typically has two bounds, $a$ (the lower bound) and $b$ (the upper bound). We
will write them as $\int_{a}^{b} \mathrm{d} x f(x)$ to mean the area under the curve given by the function $f$ between $a$ and $b$.
You should think of these integrals as being the continuous analogues of simple sums. That is, you can "kind of" read such an integral as $\sum^b_{x=a}f(x)$.

The interpretation of an integral as a sum comes from the following thought experiment. Suppose we were to discretize the range $[a, b]$ into R many units of width $(a - b)/R$. Then, we could approximate the area
under the curve by a sum over these units, evaluating $f(x)$ at each position (to get the height of a rectangle
there) and multiplying by $(a - b)/R$, which is the width. Summing these rectangles (see Figure ??) will
approximate the area. As we let $R\rightarrow \infty$, we'll get a better and better approximation. However, as $R \rightarrow \infty $,
the width of each rectangle will approach $0$. We name this width "$dx$," and thus the integral notation mimics
almost exactly the "rectangular sum" notation (we have width of $dx$ times height of $f(x)$, summed over the
range). 



An common integral is that over an unbounded range, for instance $\int_{-\infty}^{\infty}dxf(x)$. While it may seem crazy
to try to sum up things over an infinite range, there are actually many functions $f$ for which the result of
this integration is finite. For instance, a half-bounded integral of $1/x^2$ is finite  :

$\int_{1}^{\infty} \mathrm{d} x \frac{1}{x^{2}}=\lim _{b \rightarrow \infty} \int_{1}^{b} \mathrm{d} x \frac{1}{x^{2}}=\lim _{b \rightarrow \infty}\left[-\frac{1}{b}-\left(-\frac{1}{1}\right)\right]=0+1=1$

A similar calculation can show the following (called Gauss' integral):

$\int_{-\infty}^{\infty}dxe^{-x^2}=\sqrt{\pi}$



## 1.3 Convexity

The notion of a convex function and a convex set will turn out to be incredibly important in our studies. A convex function is, in many ways, "well behaved." Although not a precise definition, you can think of
a convex function as one that has a single point at which the derivative goes to zero, and this point is a
minimum. For instance, the function $f(x) = 2x^2 + 3x + 1$ from Figure ?? is convex. One usually thinks of
context functions as functions that "hold water" - i.e., if you were to pour water into them, it wouldn't spill
out.



The opposite of a convex function is a concave function. A function $f$ is concave if the function $-f$ is convex. So convex functions look like valleys, concave functions like hills.



The reason we care about convexity is because it means that finding minima is easy. For instance, the fact
that $f(x) = 2 x^2 + 3 x + 1$ is convex means that once we've found a point that has a zero derivative, we have found the unique, global minimum. For instance, consider the function $f(x) = x^4 + x^3 + 4x^2$, which is plotted in Figure ??. This function is non-convex. It has three points at which the derivative goes to zero. The left-most corresponds to a global minimum, the middle to a local maximum and the right-most to a local minimum. What this means is that even if we are able to find a point x for which $\partial_xf(x) = 0$, it is not necessarily true that x is a minimum (or maximum) of f.



More formally, a function f is convex on the range $[a{,}b]$ if its second derivative is positive everywhere in that range. The second derivative is simply the derivative of the derivative (and is physically associated with acceleration). The second derivative of $f$ with respect to $x$ is typically denoted by one of the following:

$\frac{\partial^{2} f}{\partial x \partial x}=\frac{\partial^{2} f}{\partial x^{2}}=\frac{\partial}{\partial x}\left[\frac{\partial f}{\partial x}\right] \quad=\quad \partial_{x} \partial_{x} f$

A function $f$ is convex everywhere if $f$ is convex on the range $(-\infty, \infty)$. 

**Example 4**. Consider the function $f(x) = 2x^2 +3x+1$. We've already computed the first derivative of this
function: $\partial_xf(x) = 4x + 3$. To compute the second derivative of f, we just re-differentiate the derivative,
yielding $\partial_x\partial_xf(x) = 4$. Clearly, the function that maps everything to $4$ is positive everywhere, so we know
that f is convex.



**Exercise 4**. Verify whether the functions from Exercise 1 are convex, concave or neither. An analogous notion to a convex function is a convex set. Consider some subset A of the real line. We'll convex set
denote the real line by R, so we have $A \subset R$. We say that A is convex whenever the following holds: for all convex $x, y \in A$ and $\lambda \in [0, 1]$, the point $\lambda x + (1-\lambda)y $is also in A. In more mathy terms, A is convex if it is closed under convex combination.

The way to think of this is as follows. Given two points $x$ and $y$ on the plane, the function $f(\lambda) = \lambda x+(1-\lambda)y$ on the range $lambda \in [0, 1] $denotes the line segment that joins $x and y$.  A set $A$ is convex if all points on all line segment such line segments are also contained in A.



In general, all open and closed intervals of the real line are convex.

**Exercise 5**. Show that $[-3,-1] \cup[1, 3]$ (the union of the closed interval $[-3,-1]$ and the closed interval
$[1, 3]$) is not convex.

Why do we care about convex sets? A lot of times we're going to be trying to minimize some function f(x),
but under a constraint that x lies in some set A. If A is convex, the life is much easier. This is because it
means that if we have two solutions, both in A, we can try to nd a better solution between them, and this
is guaranteed to still be in A (by convexity). (We'll come back to this later in Section 3.1.
An immediate question is: convex sets and convex functions share the word "convex." This implies that
they have something in common. They do, but we'll need to get to multidimensional analogues before we
can see this (see Section 3.1.

1.4 Wrap-up
The important concepts from this section are:

-   Differentiation as a tool to finding maxima/minima of a function.
-   Integration as a tool for computing area under a function.
-    Convex functions hold water.
    If you feel comfortable with these issues and can solve most of the exercises, you're in good shape!