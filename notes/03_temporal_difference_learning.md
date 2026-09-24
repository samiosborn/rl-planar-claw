# notes/03_temporal_difference_learning.md

# Temporal-Difference Learning

Temporal-difference learning concerns the estimation of value functions from sampled experience.

---

## Value Functions

**Definition — State-Value Function**

The state-value function under policy $\pi$ is the expected return when starting in state $s$ and subsequently following $\pi$:

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s
\right]
$$

Where the return-to-go is

$$
G_t
=
\sum_{k=0}^{\infty}
\gamma^k R_{t+k+1}
$$

Thus $V^\pi(s)$ measures how valuable it is, in expectation, to be in state $s$ while following policy $\pi$.

**Definition — Action-Value Function**

The action-value function under policy $\pi$ is the expected return when starting in state $s$, taking action $a$, and subsequently following $\pi$:

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s,\,
A_t=a
\right]
$$

Thus $Q^\pi(s,a)$ measures the expected value of choosing action $a$ in state $s$.

---

## Bellman Expectation Equations

The return can be decomposed recursively:

$$
\begin{aligned}
G_t
&=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2 R_{t+3}
+
\cdots
\\
&=
R_{t+1}
+
\gamma
\left(
R_{t+2}
+
\gamma R_{t+3}
+
\cdots
\right)
\\
&=
R_{t+1}
+
\gamma G_{t+1}
\end{aligned}
$$

This recursive structure gives the Bellman expectation equations.

### State-Value Bellman Equation

Starting from

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s
\right]
$$

Substitute the recursive expression for $G_t$:

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma G_{t+1}
\mid
S_t=s
\right]
$$

Since

$$
\mathbb{E}_\pi
\left[
G_{t+1}
\mid
S_{t+1}
\right]
=
V^\pi(S_{t+1})
$$

We obtain

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
\mid
S_t=s
\right]
$$

For a discrete MDP this can be written explicitly as

$$
V^\pi(s)
=
\sum_a
\pi(a \mid s)
\sum_{s',r}
p(s',r \mid s,a)
\left[
r+\gamma V^\pi(s')
\right]
$$

### Action-Value Bellman Equation

Starting from

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s,\,
A_t=a
\right]
$$

Substitute the recursive expression for $G_t$:

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma G_{t+1}
\mid
S_t=s,\,
A_t=a
\right]
$$

At the next state $S_{t+1}$, the policy chooses the next action according to $\pi$. Therefore,

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma Q^\pi(S_{t+1},A_{t+1})
\mid
S_t=s,\,
A_t=a
\right]
$$

For a discrete action space,

$$
Q^\pi(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma
\sum_{a'}
\pi(a' \mid S_{t+1})
Q^\pi(S_{t+1},a')
\mid
S_t=s,\,
A_t=a
\right]
$$

---

## Bellman Operator

**Definition — Bellman Expectation Operator**

For a policy $\pi$, define the Bellman expectation operator $\mathcal{T}^\pi$ acting on a value function $V$ by

$$
(\mathcal{T}^\pi V)(s)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma V(S_{t+1})
\mid
S_t=s
\right]
$$

The true value function is a fixed point of the Bellman operator:

$$
\mathcal{T}^\pi V^\pi
=
V^\pi
$$

### Contraction Property

For two bounded value functions $V$ and $W$,

$$
\left\|
\mathcal{T}^\pi V
-
\mathcal{T}^\pi W
\right\|_\infty
\leq
\gamma
\left\|
V-W
\right\|_\infty
$$

Since $0 \leq \gamma < 1$, $\mathcal{T}^\pi$ is a contraction under the supremum norm. Repeated Bellman updates therefore converge to the unique fixed point $V^\pi$:

$$
V_{k+1}
=
\mathcal{T}^\pi V_k
\quad
\Longrightarrow
\quad
V_k
\rightarrow
V^\pi
$$

---

## Monte Carlo Estimation

The Bellman equations define value functions as expectations, but these expectations are generally unknown. One approach is to estimate them directly from sampled trajectories.

For an episodic trajectory, the observed return from time $t$ is

$$
G_t
=
\sum_{k=0}^{T-t-1}
\gamma^k R_{t+k+1}
$$

Since

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s
\right]
$$

Given $N$ sampled returns following visits to state $s$,

$$
G^{(1)},
G^{(2)},
\ldots,
G^{(N)}
$$

The value function can be estimated by

$$
V^\pi(s)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
G^{(i)}
$$

**Definition — Monte Carlo Prediction Error**

For a value estimate $V$, define the Monte Carlo prediction error as

$$
e_t^{\mathrm{MC}}
=
G_t-V(S_t)
$$

The corresponding incremental update is

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
G_t-V(S_t)
\right]
$$

The Monte Carlo update moves $V(S_t)$ towards the complete observed return $G_t$. It therefore does not bootstrap from a future value estimate.

However, $G_t$ is not available until the future rewards have been observed. This motivates temporal-difference learning, which replaces the complete return with a bootstrapped target.

---

## Temporal-Difference Learning

Temporal-difference learning updates a value estimate using observed rewards together with an estimate of a future state's value.

### TD Target and TD Error

**Definition — One-Step TD Target**

$$
y_t^{\mathrm{TD}}
=
R_{t+1}
+
\gamma V(S_{t+1})
$$

Because the target depends on another value estimate, TD learning is a bootstrapping method.

**Definition — Temporal-Difference Error**

The TD error is the difference between the TD target and the current value estimate:

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

The one-step TD update is therefore

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha \delta_t
$$

### TD Error as an Advantage Estimate

Recall

$$
A^\pi(s,a)
=
Q^\pi(s,a)
-
V^\pi(s)
$$

Suppose the TD error is formed using the exact value function $V^\pi$. Then

$$
\begin{aligned}
\mathbb{E}_\pi
\left[
\delta_t
\mid
S_t=s,\,
A_t=a
\right]
&=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
-
V^\pi(S_t)
\mid
S_t=s,\,
A_t=a
\right]
\\
&=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
\mid
S_t=s,\,
A_t=a
\right]
-
V^\pi(s)
\\
&=
Q^\pi(s,a)
-
V^\pi(s)
\\
&=
A^\pi(s,a)
\end{aligned}
$$

Thus, with an exact value function, $\delta_t$ is a one-step unbiased estimator of the advantage conditional on $(S_t,A_t)$.

### n-Step Returns

**Definition — n-Step Return**

For a value estimate $V$, the $n$-step return is

$$
G_t^{(n)}
=
\sum_{l=0}^{n-1}
\gamma^l R_{t+l+1}
+
\gamma^n V(S_{t+n})
$$

For $n=1$,

$$
G_t^{(1)}
=
R_{t+1}
+
\gamma V(S_{t+1})
$$

which is the one-step TD target.

As $n$ increases, more observed rewards are used before bootstrapping. In an episodic task, if the return extends to termination,

$$
G_t^{(n)}
=
G_t
$$

and the $n$-step return becomes the Monte Carlo return.

### n-Step Advantage Estimates

Let the value estimate be a learned critic $V_\phi$. Define the corresponding $n$-step advantage estimate by

$$
\hat A_t^{(n)}
=
G_t^{(n)}
-
V_\phi(S_t)
$$

where

$$
G_t^{(n)}
=
\sum_{l=0}^{n-1}
\gamma^l R_{t+l+1}
+
\gamma^n V_\phi(S_{t+n})
$$

and define the TD residual using the same critic:

$$
\delta_t
=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
-
V_\phi(S_t)
$$

For $n=2$,

$$
\begin{aligned}
\delta_t
+
\gamma\delta_{t+1}
&=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
-
V_\phi(S_t)
\\
&\quad+
\gamma R_{t+2}
+
\gamma^2V_\phi(S_{t+2})
-
\gamma V_\phi(S_{t+1})
\\
&=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2V_\phi(S_{t+2})
-
V_\phi(S_t)
\\
&=
G_t^{(2)}
-
V_\phi(S_t)
\end{aligned}
$$

The intermediate value terms cancel in the same way for general $n$, giving the exact identity

$$
\boxed{
\hat A_t^{(n)}
=
\sum_{l=0}^{n-1}
\gamma^l\delta_{t+l}
}
$$

This identity is exact for any $V_\phi$. Whether $\hat A_t^{(n)}$ accurately estimates $A^\pi(S_t,A_t)$ depends on the critic and the amount of bootstrapping.

---

## Lambda Returns and Generalised Advantage Estimation

### Lambda Return

**Definition — Lambda Return**

For $0 \leq \lambda < 1$, the $\lambda$-return is the geometrically weighted average of $n$-step returns:

$$
G_t^\lambda
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
G_t^{(n)}
$$

The weights sum to one:

$$
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
=
1
$$

The $\lambda$-return also satisfies the recursion

$$
G_t^\lambda
=
R_{t+1}
+
\gamma
\left[
(1-\lambda)V(S_{t+1})
+
\lambda G_{t+1}^\lambda
\right]
$$

The value estimate can therefore be updated towards $G_t^\lambda$:

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
G_t^\lambda
-
V(S_t)
\right]
$$

For an episodic task, $\lambda=1$ is understood as the limiting case in which the target becomes the complete Monte Carlo return.

### Generalised Advantage Estimation

**Definition — Generalised Advantage Estimation**

Generalised advantage estimation applies the same geometric weighting to the $n$-step advantage estimates:

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
\hat A_t^{(n)}
$$

Using

$$
\hat A_t^{(n)}
=
\sum_{l=0}^{n-1}
\gamma^l\delta_{t+l}
$$

we obtain

$$
\begin{aligned}
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
&=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
\sum_{l=0}^{n-1}
\gamma^l\delta_{t+l}
\\
&=
\sum_{l=0}^{\infty}
\gamma^l\delta_{t+l}
(1-\lambda)
\sum_{n=l+1}^{\infty}
\lambda^{n-1}
\\
&=
\sum_{l=0}^{\infty}
\gamma^l\delta_{t+l}
(1-\lambda)
\frac{\lambda^l}{1-\lambda}
\\
&=
\boxed{
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
}
\end{aligned}
$$

Thus GAE is an exponentially weighted sum of future TD residuals.

Because the weights in the $\lambda$-return sum to one,

$$
\begin{aligned}
G_t^\lambda
-
V_\phi(S_t)
&=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
\left[
G_t^{(n)}
-
V_\phi(S_t)
\right]
\\
&=
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
\end{aligned}
$$

Hence the exact identity

$$
\boxed{
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
G_t^\lambda
-
V_\phi(S_t)
}
$$

### Limiting Cases

When $\lambda=0$,

$$
\hat A_t^{\mathrm{GAE}(\gamma,0)}
=
\delta_t
$$

so GAE reduces to the one-step TD advantage estimate.

For an episodic task, when $\lambda=1$,

$$
\hat A_t^{\mathrm{GAE}(\gamma,1)}
=
G_t
-
V_\phi(S_t)
$$

so the advantage estimate becomes the Monte Carlo return minus the critic baseline.

Intermediate values of $\lambda$ interpolate between these cases.

---

## Bias-Variance Trade-off

Monte Carlo, one-step TD, $n$-step returns, and GAE differ in how much they rely on sampled rewards versus bootstrapping.

Monte Carlo uses the complete observed return

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2 R_{t+3}
+
\cdots
$$

It does not bootstrap from the current value estimate, so its target does not directly inherit critic approximation error. It instead depends on the full stochastic future trajectory and therefore commonly has high variance.

The $n$-step return

$$
G_t^{(n)}
=
\sum_{l=0}^{n-1}
\gamma^l R_{t+l+1}
+
\gamma^nV(S_{t+n})
$$

interpolates between one-step TD and Monte Carlo. Smaller $n$ bootstraps sooner; larger $n$ uses more sampled rewards before bootstrapping.

GAE provides the analogous interpolation for advantage estimation:

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
$$

Smaller $\lambda$ places more weight on short-horizon TD residuals and therefore relies more strongly on the critic. Larger $\lambda$ incorporates rewards over a longer horizon and generally increases variance while reducing dependence on bootstrapping.

When $V_\phi=V^\pi$, the one-step TD residual is already an unbiased advantage estimator conditional on $(S_t,A_t)$. When $V_\phi\neq V^\pi$, GAE remains an exact weighted sum of TD residuals, but it is only an approximate estimator of the true advantage.

---
