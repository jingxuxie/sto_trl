# Theory Appendix: Stochastic Transitive RL

This appendix formalizes the model-based stochastic TRL operator implemented in
`sto_trl/learners.py`.

## Setup

Let `S` and `A` be finite. For a fixed goal `g`, define the discounted hitting
time value after taking first action `a` in state `s`:

```text
Q^*(s,a,g) = sup_pi E[ gamma^{tau_g} 1{tau_g < infinity} | s_0=s, a_0=a ],
```

where `tau_g = inf{t >= 0 : s_t = g}` and `gamma in (0,1)`. We use the
boundary convention

```text
Q^*(g,a,g) = 1,      V^*(s,g) = max_a Q^*(s,a,g).
```

For any candidate table `Q`, define `V_Q(s,g)=max_a Q(s,a,g)` with
`V_Q(g,g)=1`.

The stochastic TRL implementation uses:

```text
(B Q)(s,a,g) = gamma E_{s'~P(.|s,a)} V_Q(s',g),   s != g
(B Q)(g,a,g) = 1
```

and

```text
(T Q)(s,a,g) = max_{w != s} Q(s,a,w) V_Q(w,g),    s != g
(T Q)(g,a,g) = 1.
```

The `w != s` exclusion is part of the implemented action-value operator. It
prevents the tautological first leg `Q(s,a,s)=1` from copying `V_Q(s,g)` into
every first action regardless of `a`.

The update is

```text
F(Q) = max(BQ, TQ)
```

with the maximum taken pointwise. The default initialization is

```text
Q_0(s,a,g) = gamma P(g | s,a),    s != g
Q_0(g,a,g) = 1.
```

## Theorem 1: Lower-Bound Preservation

Assume `0 <= Q(s,a,g) <= Q^*(s,a,g)` for all `s,a,g`. Then

```text
BQ <= Q^*,     TQ <= Q^*,     F(Q) <= Q^*.
```

Consequently, if `Q_0 <= Q^*`, then every iterate `Q_k = F^k(Q_0)` is also a
pointwise lower bound on `Q^*`.

### Proof

For the Bellman term, optimal hitting values satisfy the Bellman optimality
equation

```text
Q^*(s,a,g) = gamma E_{s'~P(.|s,a)} V^*(s',g),  s != g,
Q^*(g,a,g) = 1.
```

Since `Q <= Q^*`, we have `V_Q(s,g) <= V^*(s,g)` for every `s,g`. Therefore

```text
(BQ)(s,a,g)
  = gamma E[V_Q(s',g)]
 <= gamma E[V^*(s',g)]
  = Q^*(s,a,g)
```

for `s != g`, and the boundary case is equal to one.

For the transitive term, fix `s != g`, first action `a`, and intermediate
state `w != s`. Consider the composite policy that first follows a policy for
goal `w` after taking action `a`, then, if `w` is reached, switches to an
optimal policy for goal `g` from `w`. The discounted value contributed by paths
that reach `w` before switching is

```text
Q^*(s,a,w) V^*(w,g).
```

If `g` is reached before `w`, the true value for goal `g` under the same first
action is only larger than this contribution because it receives earlier
discounted reward. Since `Q^*(s,a,g)` is optimal over all policies for goal
`g`, the composite policy gives the feasible-policy lower bound

```text
Q^*(s,a,w) V^*(w,g) <= Q^*(s,a,g).
```

Using `Q <= Q^*` and `V_Q <= V^*`,

```text
Q(s,a,w) V_Q(w,g)
 <= Q^*(s,a,w) V^*(w,g)
 <= Q^*(s,a,g).
```

Maximizing over `w != s` preserves the inequality, so `TQ <= Q^*`. The pointwise
maximum of two lower bounds is also a lower bound, hence `F(Q) <= Q^*`.

The default one-step initialization is a lower bound because it is the value of
the policy class that hits `g` in exactly one transition and ignores all later
opportunities:

```text
gamma P(g | s,a) <= gamma E[V^*(s',g)] = Q^*(s,a,g).
```

Induction completes the proof.

## Theorem 2: Logarithmic Propagation on Reliable Paths

Consider a deterministic path

```text
s_0 -> s_1 -> ... -> s_H
```

with one available path action at each non-goal state and goal `g=s_H`. Assume
the one-step initialization gives exact one-step values

```text
Q_0(s_i,a_i,s_{i+1}) = gamma
```

and `Q_0(g,a,g)=1`. After `k` stochastic-TRL sweeps, every path segment of
length at most `2^k` has its exact discounted value represented:

```text
Q_k(s_i,a_i,s_j) = gamma^{j-i}
```

whenever `1 <= j-i <= 2^k`.

Therefore a reliable path of length `H` can be represented after
`ceil(log2 H)` transitive sweeps.

### Proof

The claim holds at `k=0` because the initialization contains every length-one
edge. Assume it holds for all segments of length at most `m=2^k`. Consider a
segment from `s_i` to `s_j` with length `L=j-i <= 2m`. Choose an intermediate
state `w=s_l` so that both subsegments have length at most `m`, for example
`l=i+min(m,L)`.

By the induction hypothesis, the first subsegment has value
`gamma^{l-i}` and the second has value `gamma^{j-l}`. The transitive backup can
therefore compose them:

```text
Q_k(s_i,a_i,w) V_k(w,s_j)
  = gamma^{l-i} gamma^{j-l}
  = gamma^{j-i}.
```

Because deterministic path hitting values cannot exceed `gamma^{j-i}`, the
lower-bound result from Theorem 1 implies the value is exact after the update.
Thus the represented segment length doubles each sweep.

## Theorem 3: Shortcut Calibration

Consider a stochastic shortcut MDP with start state `s`, safe first action
`a_safe`, risky first action `a_risky`, and goal `g`.

Assume:

- The safe route reaches `g` deterministically after `H` steps.
- The risky route reaches `g` after `d` steps with probability `p`, and
  otherwise transitions to a trap from which `g` is unreachable.
- The empirical model estimates the risky success probability as `p`.

Then the calibrated risky value is

```text
Q^*(s,a_risky,g) = gamma^d p,
```

while the safe value is

```text
Q^*(s,a_safe,g) = gamma^H.
```

If `gamma^H > gamma^d p`, the calibrated optimal first action is safe. A
realized-positive shortcut estimator that conditions only on successful risky
trajectories can estimate the risky branch near `gamma^d`, overestimating by a
factor of approximately `1/p`.

### Proof

The safe action has deterministic hitting time `H`, so its discounted hitting
value is `gamma^H`. The risky action reaches the goal at hitting time `d` with
probability `p` and never reaches the goal otherwise, giving expected value
`p gamma^d + (1-p) 0 = gamma^d p`.

The stochastic Bellman term uses the empirical transition probabilities and
therefore backs up the probability-weighted value. By Theorem 2, once the safe
route's reliable segments are propagated, stochastic TRL represents the safe
value with logarithmically many transitive sweeps. Since the transitive update
preserves lower bounds by Theorem 1, it cannot turn the risky branch into a
value larger than the calibrated model value.

By contrast, a realized-positive estimator that trains only on successful risky
trajectories observes hitting time `d` conditional on success and can assign
value near `gamma^d`. Relative to the calibrated value `gamma^d p`, this is an
overestimate by roughly `1/p`.

## Notes for the Paper

- The lower-bound theorem assumes an exact empirical model. With finite-sample
  estimation error, the lower bound is with respect to the empirical model, not
  necessarily the true environment.
- The experiments use the empirical-model interpretation: full Bellman is the
  long-sweep reference for the same empirical transition/topology model.
- The continuous-control results add executor error on top of planner error.
  This is why AntMaze seed 2 can trail full Bellman by a small rollout-level
  gap even though the planner values are near-identical.
