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

---

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

Thus $Q^\pi(s,a)$ measures the expected long-term value of choosing action $a$ in state $s$.

---

## Bellman Expectation Equations

The return can be decomposed recursively.

Starting from

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2 R_{t+3}
+
\cdots,
$$

If we factor out the first reward:

$$
G_t
=
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
=
R_{t+1}
+
\gamma G_{t+1}
$$

This recursive structure of the return gives the Bellman expectation equations.

### State-Value Bellman Equation

Starting from the definition

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

The value of the current state is therefore equal to the expected immediate reward plus the discounted value of the next state.

For a discrete MDP (where both the actions and state is discrete), this can be written explicitly as

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

---

### Action-Value Bellman Equation

Starting from the definition

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

For a discrete-state MDP

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

For a policy $\pi$, define the Bellman expectation operator $\mathcal{T}^\pi$ acting on an arbitrary value function $V$ by

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

The true value function $V^\pi$ satisfies the below and is therefore a fixed point of the Bellman operator.

$$
V^\pi
=
\mathcal{T}^\pi V^\pi
$$

---

### Contraction Property

For two value functions $V$ and $W$, the Bellman expectation operator satisfies

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

Since $0 \leq \gamma < 1$, applying the Bellman operator reduces the maximum distance between two value functions.

Repeated Bellman updates therefore converge towards the unique fixed point $V^\pi$:

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

This provides the mathematical basis for iterative methods that estimate value functions.

---

## Monte Carlo Estimation

The Bellman equations define value functions as expectations, but these expectations are generally unknown.

One approach is to estimate them directly from sampled trajectories.

Suppose a trajectory produces rewards

$$
R_{t+1},
R_{t+2},
\ldots,
R_T
$$

The observed return from time $t$ is

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
+
\gamma^{T-t-1}R_T
$$

Or equivalently,

$$
G_t
=
\sum_{k=0}^{T-t-1}
\gamma^k R_{t+k+1}
$$

Stating the definition

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

The value function can be estimated by the sample mean:

$$
V^\pi(s)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
G^{(i)}
$$

This is a Monte Carlo estimate of the state value.

---

**Definition — Monte Carlo Prediction Error**

$$
\text{Monte Carlo Prediction Error} = G_t - V(S_t)
$$

The expected reward-to-go should converge to the true value function so this difference is the error. 

---

Equivalently, the value estimate can be updated incrementally:

$$
V(s)
\leftarrow
V(s)
+
\alpha
\left[
G_t-V(s)
\right]
$$

Where $\alpha$ is the learning rate.

Monte Carlo estimation of the value function uses the actual sampled future rewards as its target for updating:

$$
\text{Monte Carlo target}
=
G_t
$$

It therefore does not require an estimate of the value of a future state, i.e. to update $V(S_t)$ we do not need $V(S_{t+1})$. 

However, the complete return $G_t$ is not known until all future rewards have been observed. In an episodic problem, this commonly means waiting until the trajectory has progressed to termination before the earlier states can be updated.

This motivates temporal-difference learning: instead of waiting for the complete return, estimate part of the future using the current value function.

## Temporal-Difference Learning

Temporal-difference learning updates a value estimate using the observed immediate reward together with an estimate of the value of the next state.

Unlike Monte Carlo methods, it does not wait for the complete return $G_t$ to be observed.

---

### TD Target

For one-step temporal-difference learning, the TD target is

$$
\text{TD target}
=
R_{t+1}
+
\gamma V(S_{t+1})
$$

The update is

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
\right]
$$

Because the TD target depends on another value estimate, TD learning is a bootstrapping method.

---

### TD Error

**Definition — Temporal-Difference Error**

The temporal-difference error is the difference between the TD target and the current value estimate. 

For one-step TD learning,  

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

The TD update can therefore be written as

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha \delta_t
$$

---

### TD Error as an Advantage Estimate

Recall that the advantage function is

$$
A^\pi(s,a)
=
Q^\pi(s,a)
-
V^\pi(s)
$$

If the value estimate is exact, so that

$$
V(S_t)
=
V^\pi(S_t)
$$

Then the expected TD error conditioned on the current state and action is

$$
\mathbb{E}_\pi
\left[
\delta_t
\mid
S_t=s,\,
A_t=a
\right]
$$

Substituting the definition of $\delta_t$,

$$
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
$$

Since $V^\pi(S_t)=V^\pi(s)$ is fixed under the conditioning,

$$
=
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
$$

By the Bellman equation for the action-value function,

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma V^\pi(S_{t+1})
\mid
S_t=s,\,
A_t=a
\right]
$$

Therefore,

$$
\mathbb{E}_\pi
\left[
\delta_t
\mid
S_t=s,\,
A_t=a
\right]
=
A^\pi(s,a)
$$

Thus, when the value function is accurate, the TD error provides a one-step sample estimate of the advantage.

This is an important connection between value learning and policy-gradient methods.

---

### n-Step Returns

One-step TD uses the target

$$
R_{t+1}
+
\gamma V(S_{t+1})
$$

Monte Carlo uses the full sampled return

$$
G_t.
$$

An $n$-step return lies between these two extremes.

**Definition — n-Step Return**

The $n$-step return from time $t$ is

$$
G_t^{(n)}
=
R_{t+1}
+
\gamma R_{t+2}
+
\cdots
+
\gamma^{n-1}R_{t+n}
+
\gamma^n V(S_{t+n})
$$

The first $n$ rewards are observed directly, while the remaining future return is approximated using the value estimate $V(S_{t+n})$.

For $n=1$,

$$
G_t^{(1)}
=
R_{t+1}
+
\gamma V(S_{t+1})
$$

Which is the ordinary one-step TD target.

For $n=2$,

$$
G_t^{(2)}
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2 V(S_{t+2})
$$

As $n$ increases, the estimate relies on more observed rewards and less bootstrapping.

For an episodic task, if $n$ extends all the way to the end of the episode, then

$$
G_t^{(n)}
=
G_t
$$

and the $n$-step return becomes the Monte Carlo return.

Thus,

$$
\text{1-step TD}
\longrightarrow
\text{n-step returns}
\longrightarrow
\text{Monte Carlo}
$$

Which forms a continuum between heavy bootstrapping and using the full sampled return.

## Lambda Returns

## Bias-Variance Trade-off
