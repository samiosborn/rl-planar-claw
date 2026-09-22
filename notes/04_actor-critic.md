# notes/04_actor-critic.md

# Actor-Critic

## From Policy Gradient to Actor-Critic

Actor-critic methods combine two ideas developed separately in the previous notes: policy-gradient optimisation and temporal-difference value estimation.

* The policy-gradient result tells us how the policy should move if we know the advantage of an action.
* Temporal-difference learning gives us a way to estimate that advantage without waiting for the complete Monte Carlo return.

### Starting Point: The Policy Gradient

With $J(\theta)$ as the policy objective, the policy gradient is

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
A^{\pi_\theta}(S_t,A_t)
\nabla_\theta
\log \pi_\theta(A_t\mid S_t)
\right]
$$

This is an exact mathematical identity when $A^{\pi_\theta}$ is the true advantage function of the current policy.

#### Two Components of the Gradient

The gradient contains two distinct quantities:

$$
\nabla_\theta \log \pi_\theta(A_t\mid S_t)
$$

which describes how the probability of the sampled action changes with the policy parameters, and

$$
A^{\pi_\theta}(S_t,A_t)
$$

which measures how much better or worse the sampled action is than the expected value of being in that state.

The policy-gradient theorem therefore gives the direction in which to change the policy, but it assumes that the advantage is known. In a model-free setting, the true advantage is generally unknown and must instead be estimated.

---

### The Missing Quantity: Advantage Estimation

The definition of the advantage is

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
$$

Neither $Q^\pi(s,a)$ nor $V^\pi(s)$ is generally available exactly, so the policy gradient cannot be evaluated directly.

#### Replacing the True Advantage with an Estimate

The exact policy gradient is therefore approximated using sampled experience:

$$
\nabla_\theta J(\theta)
\approx
\hat g
$$

Replace the true advantage by a sample estimate,

$$
A^\pi(S_t,A_t)
\longrightarrow
\hat A_t
$$

This gives the stochastic gradient estimator

$$
\hat g_t
=
\hat A_t
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
$$

The problem now becomes how to construct $\hat A_t$?

---

#### Monte Carlo Advantage Estimation

The sampled return-to-go $G_t$ is an unbiased sample estimate of $Q^\pi(s,a)$ because

$$
Q^\pi(s,a)
=
\mathbb{E}
\left[
G_t
\mid
S_t=s,A_t=a
\right]
$$

Consider the Monte Carlo advantage estimator

$$
\hat A_t^{\mathrm{MC}}
=
G_t-V^\pi(S_t)
$$

Conditioning on $S_t=s$ and $A_t=a$,

$$
\begin{aligned}
\mathbb{E}
\left[
\hat A_t^{\mathrm{MC}}
\mid
S_t=s,A_t=a
\right]
&=
\mathbb{E}
\left[
G_t-V^\pi(S_t)
\mid
S_t=s,A_t=a
\right]
\\
&=
\mathbb{E}
\left[
G_t
\mid
S_t=s,A_t=a
\right]
-
V^\pi(s)
\\
&=
Q^\pi(s,a)-V^\pi(s)
\\
&=
A^\pi(s,a)
\end{aligned}
$$

Thus, when the true value function is used,

$$
G_t-V^\pi(S_t)
$$

is an unbiased Monte Carlo estimator of the advantage.

The corresponding REINFORCE-style gradient estimator with a baseline is

$$
\hat g_t^{\mathrm{MC}}
=
\left(
G_t-V^\pi(S_t)
\right)
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
$$

The difficulty is that $G_t$ depends on the complete future trajectory, so its randomness can give the policy-gradient estimator high variance.

---

#### Replacing Monte Carlo with Temporal Difference Estimation

Temporal-difference learning suggests another possibility.

Instead of using the complete sampled future return, approximate the future after the next state using a value function:

$$
G_t
\quad\longrightarrow\quad
R_{t+1}
+
\gamma V(S_{t+1})
$$

Subtracting the value of the current state gives

$$
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

This is precisely the one-step TD error,

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

From temporal-difference learning, if $V=V^\pi$,

$$
\mathbb{E}
\left[
\delta_t
\mid
S_t=s,A_t=a
\right]
=
A^\pi(s,a)
$$

Therefore the TD error can be used as a one-step estimate of the advantage:

$$
\hat A_t
=
\delta_t
$$

---

#### Substituting the TD Error into the Policy Gradient

Substituting $ \hat A_t=\delta_t $ into the stochastic policy-gradient estimator gives

$$
\hat g_t
=
\delta_t
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
$$

---

#### Introducing the Learned Critic

In actor-critic, the value function $V$ is a learned, parametrised function. 

$$
V^\pi(s)
\approx
V_\phi(s)
$$

The policy is the **actor**: 

$$
\pi_\theta(a\mid s)
$$

The learned value function is the **critic**: 

$$
V_\phi(s)
$$

