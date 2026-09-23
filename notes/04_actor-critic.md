# notes/04_actor-critic.md

# Actor-Critic

## Policy Gradient

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

### Two Components of the Gradient

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

### Advantage Estimation

The definition of the advantage is

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
$$

#### Replacing the True Advantage with an Estimate

We use a sample estimate of the advantage,

$$
A^\pi(S_t,A_t)
\longrightarrow
\hat A_t
$$

This gives the stochastic gradient estimator

$$
\nabla_\theta J(\theta)
\approx
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

Conditioning on $S_t=s$ and $A_t=a$, we see it is unbiased

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

The difficulty is that $G_t$ depends on the complete future trajectory, which gives the policy-gradient estimator high variance.

---

#### Temporal Difference (TD) Advantage Estimation

Consider the one-step TD error,  

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

If $V=V^\pi$,

$$
\mathbb{E}
\left[
\delta_t
\mid
S_t=s,A_t=a
\right]
= 
Q^\pi(s,a)-V^\pi(s)
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

#### Policy Gradient Estimation using TD Error 

Substituting $ \hat A_t=\delta_t $ into the stochastic policy-gradient estimator gives

$$
\hat g_t
=
\delta_t
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
$$

---

## Actor and Critic Parameterisation

Actor-critic methods maintain two separate parametrised functions: a policy and a value function.

### The Actor

The actor is the policy $ \pi_\theta(a\mid s) $ where $\theta$ denotes the policy parameters.

Its objective is to adjust $\theta$ so as to increase the expected return $ J(\theta) $.

### The Critic

The critic is a learned approximation to the value function of the current policy where $\phi$ denotes the critic parameters.

$$
V^\pi(s)
\approx
V_\phi(s)
$$

The critic estimates the expected return from state $s$ under the policy currently induced by the actor.

### Separate Parameter Sets

The actor adjusts $\theta$ to improve the policy

$$
\pi_\theta(a\mid s)
$$

While the critic adjusts $\phi$ to improve the approximation

$$
V_\phi(s)
\approx
V^{\pi_\theta}(s)
$$

The important coupling is that the critic is estimating the value function of the policy defined by the current actor.

Therefore, as $\theta$ changes, the target value function also changes, which will likely change $\phi$ for it to be optimal. 

---

## Learning the Critic

The critic approximates the state-value function of the current policy

$$
V_\phi(s)
\approx
V^{\pi_\theta}(s)
$$

The critic parameters $\phi$ must therefore be learned from sampled transitions.

### Bootstrapped Value Target

From the Bellman expectation equation,

$$
V^{\pi_\theta}(s)
=
\mathbb{E}_{\pi_\theta}
\left[
R_{t+1}
+
\gamma V^{\pi_\theta}(S_{t+1})
\mid
S_t=s
\right]
$$

For a sampled transition $ (S_t,A_t,R_{t+1},S_{t+1}) $

We construct the one-step TD target for the critic

$$
y_t
=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
$$

The target contains two sources of approximation:

* $R_{t+1}$ and $S_{t+1}$ are a single sample from the environment
* $V_\phi(S_{t+1})$ approximates $V^{\pi_\theta}(S_{t+1})$

---

### Critic Loss

Define the squared value loss

$$
L_V(\phi)
=
\frac{1}{2}
\left(
y_t
-
V_\phi(S_t)
\right)^2
$$

Recall that the TD error is

$$
\delta_t
=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
-
V_\phi(S_t)
$$

Therefore

$$
\delta_t
=
y_t
-
V_\phi(S_t)
$$

Thus

$$
L_V(\phi)
=
\frac{1}{2}\delta_t^2
$$

---

### Gradient of the Critic Loss

For the standard TD update, treat $y_t$ as fixed while differentiating the current prediction.

Apply the chain rule

$$
\begin{aligned}
\nabla_\phi L_V(\phi)
&=
\frac{1}{2}
\cdot
2
\left(
y_t-V_\phi(S_t)
\right)
\nabla_\phi
\left(
y_t-V_\phi(S_t)
\right)
\\
&=
\left(
y_t-V_\phi(S_t)
\right)
\left(
-\nabla_\phi V_\phi(S_t)
\right)
\\
&=
-
\left(
y_t-V_\phi(S_t)
\right)
\nabla_\phi V_\phi(S_t)
\end{aligned}
$$

Using

$$
\delta_t
=
y_t-V_\phi(S_t)
$$

Gives

$$
\nabla_\phi L_V(\phi)
=
-\delta_t
\nabla_\phi V_\phi(S_t)
$$

Standard gradient descent is

$$
\begin{aligned}
\phi
&\leftarrow
\phi
-
\alpha_\phi
\nabla_\phi L_V(\phi)
\end{aligned}
$$

Using our formulation
$$
\begin{aligned}
\phi
&\leftarrow
\phi 
+
\alpha_\phi
\delta_t
\nabla_\phi V_\phi(S_t)
\end{aligned}
$$

This is the function-approximation form of the TD value update.

---

#### Semi-Gradient TD

Note how the target itself depends on $\phi$: 

$$
y_t
=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
$$

If the entire loss were differentiated with respect to $\phi$, then

$$
\nabla_\phi y_t
=
\gamma
\nabla_\phi
V_\phi(S_{t+1})
$$

Therefore

$$
\nabla_\phi
\left(
y_t-V_\phi(S_t)
\right)
=
\gamma\nabla_\phi V_\phi(S_{t+1})
-
\nabla_\phi V_\phi(S_t)
$$

The full gradient of the squared TD error would then be

$$
\nabla_\phi L_V(\phi)
=
\delta_t
\left[
\gamma\nabla_\phi V_\phi(S_{t+1})
-
\nabla_\phi V_\phi(S_t)
\right]
$$

Instead, TD learning holds the target fixed while updating the prediction at $S_t$ so that $ \nabla_\phi y_t = 0 $

This gives the previous - which we call the semi gradient: 

$$
\nabla_\phi L_V(\phi)
=
-\delta_t
\nabla_\phi V_\phi(S_t)
$$

---

## Actor Update

From the policy-gradient estimator,

$$
\hat g_t
=
\hat A_t
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
$$

### TD Error as the Advantage Estimate

Using the 1-step TD error, $ \hat A_t = \delta_t $

Substituting this into the policy-gradient estimator gives

$$
\hat g_t
=
\delta_t
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
$$

---

### Actor Update Rule

The actor performs gradient ascent on the policy objective

$$
\theta
\leftarrow
\theta
+
\alpha_\theta
\hat g_t
$$

Therefore

$$
\theta
\leftarrow
\theta
+
\alpha_\theta
\delta_t
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
$$

where $\alpha_\theta$ is the actor learning rate.

During the actor update, $\delta_t$ is treated as the scalar advantage estimate supplied by the critic. Thus the critic determines the weighting applied to the policy gradient.

The sign of $\delta_t$ determines how the probability of the sampled action changes.

* If $ \delta_t > 0 $, then the observed transition was better than predicted by the critic, so gradient ascent increases the probability of the sampled action.

* If $ \delta_t < 0 $, then the observed transition was worse than predicted, so the probability of the sampled action is decreased.

---

## Estimator Bias and Variance

### Approximate Critic

Define the critic error

$$
\epsilon_\phi(s)
=
V_\phi(s)-V^\pi(s)
$$

Substitute the critic error into the TD error

$$
\delta_t
=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
-
V_\phi(S_t)
$$


$$
\begin{aligned}
\delta_t
&=
R_{t+1}
+
\gamma
\left[
V^\pi(S_{t+1})
+
\epsilon_\phi(S_{t+1})
\right]
-
\left[
V^\pi(S_t)
+
\epsilon_\phi(S_t)
\right]
\\
&=
R_{t+1}
+
\gamma V^\pi(S_{t+1})
-
V^\pi(S_t)
+
\gamma\epsilon_\phi(S_{t+1})
-
\epsilon_\phi(S_t)
\end{aligned}
$$

Therefore

$$
\mathbb{E}[\delta_t\mid S_t,A_t]
=
A^\pi(S_t,A_t)
+
\gamma
\mathbb{E}
\left[
\epsilon_\phi(S_{t+1})
\mid
S_t,A_t
\right]
-
\epsilon_\phi(S_t)
$$

The critic error $ -\epsilon_\phi(S_t) $ depends only on the state and therefore acts as an action-independent baseline.

---

### Bias in the Actor Update

From the baseline result,

$$
\mathbb{E}
\left[
\epsilon_\phi(S_t)
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
\right]
=
0
$$

Therefore the current-state critic error does not itself bias the expected policy gradient (actor update).

The remaining error comes from the bootstrapped next-state value which can depend on the chosen action.

$$
\gamma
\mathbb{E}
\left[
\epsilon_\phi(S_{t+1})
\mid
S_t,A_t
\right]
$$

Thus, in general, when the bootstrapped critic is imperfect,

$$
\mathbb{E}[\hat g_t]
\neq
\nabla_\theta J(\theta)
$$

---

### Baseline Error vs Bootstrapping Error

This distinction is important.

A Monte Carlo estimator using an approximate baseline,

$$
\hat A_t^{\mathrm{MC}}
=
G_t-V_\phi(S_t)
$$

does not introduce bias into the expected policy gradient purely because $V_\phi$ is inaccurate, since $V_\phi(S_t)$ is still action-independent.

By contrast, one-step actor-critic uses

$$
\hat A_t^{\mathrm{TD}}
=
R_{t+1}
+
\gamma V_\phi(S_{t+1})
-
V_\phi(S_t)
$$

The term

$$
V_\phi(S_{t+1})
$$

depends on the next state, whose distribution depends on the current action. Error in this bootstrapped value estimate can therefore bias the actor update.

Actor-critic trades some of the variance of Monte Carlo estimation for bias introduced by bootstrapping from an approximate critic. 

---
