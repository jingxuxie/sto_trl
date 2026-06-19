# High-Impact Next Steps for Stochastic TRL

This plan is written for the current state of the repository after the preliminary stochastic-TRL manuscript and result package. The goal is to convert the current evidence into the strongest possible paper while keeping the claim honest: **a stochastic value-propagation / planning operator that isolates value learning from low-level control**, with a credible path toward learned neural modules.

---

## 1. Strategic answer

### Is it valid to isolate value learning from the controller?

Yes. This is a defensible and strategically good framing.

TRL itself is best read as a **value-learning / horizon-propagation method**, not primarily as a paper about inventing a new low-level controller. The title and framing are explicitly about value learning via divide and conquer. In your work, isolating high-level value propagation from execution is therefore not only reasonable, but useful: it lets you test whether the stochastic transitive backup fixes the right problem before spending large compute on controller engineering.

The right framing is:

> We study stochastic transitive **value propagation** for goal-conditioned RL. Continuous-control experiments use topology or learned-controller diagnostics to isolate the high-level value-learning question from low-level execution failures.

Do **not** frame the current result as a full end-to-end neural stochastic TRL agent unless you add such an implementation.

### Do you need harder OGBench tasks?

Eventually, yes, but not as the next bottleneck.

The current evidence already has strong stochastic topology diagnostics on PointMaze and AntMaze teleport tasks. The highest-priority gaps are more reviewer-critical than simply adding harder tasks:

1. cost-normalized propagation comparisons;
2. external baselines beyond Bellman;
3. stochastic calibration plots and phase diagrams;
4. finite-sample transition robustness;
5. a less-privileged learned-module ladder;
6. clearer theory and claim boundaries.

After those are solid, harder OGBench tasks become valuable for scaling and impact.

### Do you need baselines beyond Bellman?

Yes. Bellman baselines are necessary but not sufficient.

Bellman matched and Bellman full are the right baselines for proving the **operator story**:

- matched Bellman shows local propagation is slow;
- full Bellman is the calibrated long-sweep reference;
- stochastic TRL should match full Bellman with fewer transitive sweeps.

But high-impact reviewers will ask how the method compares to existing offline GCRL and stochastic/quasimetric methods. Add at least a subset of:

- original deterministic TRL / realized-path TRL;
- support TRL;
- CRL;
- QRL or CMD-style quasimetric RL;
- TMD;
- MQE if feasible;
- prioritized sweeping / Gauss-Seidel value iteration;
- shortest-path or risk-aware graph planners.

If only one external stochastic baseline is feasible, prioritize **TMD**.

---

## 2. Recommended paper identity

Choose one primary identity. Do not mix them loosely.

### Option A: value-propagation/operator paper

**Recommended primary identity.**

Claim:

> Stochastic Transitive RL is a Bellman-calibrated transitive value-propagation operator for empirical stochastic MDPs. It recovers TRL-style fast long-horizon propagation while avoiding over-composition of lucky stochastic outcomes.

This identity fits your current evidence.

Main evidence:

- tabular shortcut phase diagrams;
- 2D stochastic grid budget curves;
- PointMaze / AntMaze teleport topology diagnostics;
- learned transition and learned high-level head diagnostics;
- cost-normalized propagation curves;
- support-TRL and realized-path TRL failure modes.

Risk:

- reviewers may say it is model-based planning rather than full offline RL.

Response:

- agree, and make that the point: this is the stochastic value-propagation core that neural stochastic TRL should approximate.

### Option B: full offline GCRL algorithm paper

Claim:

> Stochastic TRL is a full neural offline GCRL algorithm that outperforms existing methods on OGBench stochastic tasks.

This is **not yet supported** by the current evidence.

To pursue this identity, you need:

- an end-to-end or near-end-to-end neural critic/value implementation;
- standard OGBench comparisons against CRL, QRL, HIQL, GCIQL, GCIVL, GCBC, TRL, and ideally TMD/MQE;
- multiple environment families beyond teleport topology;
- clean actor/controller training.

This route is higher risk and much more compute-intensive.

### Recommendation

Write the paper as **Option A**, with a strong learned-module appendix and a clear roadmap to Option B.

---

## 3. Claim hierarchy for the paper

Use this hierarchy in the abstract, introduction, and experiments.

### Claim 1: stochastic failure mode

Realized-path or deterministic-support transitive composition can over-compose lucky stochastic outcomes.

Required evidence:

- risky shortcut tabular tasks;
- support-TRL ablation;
- teleport support baseline.

### Claim 2: calibrated stochastic transitive backup

Combining Bellman expectation with transitive composition preserves stochastic calibration better than realized-path composition.

Required evidence:

- predicted risky action values vs true values;
- calibration curves;
- support TRL < stochastic TRL;
- stochastic TRL does not overvalue low-probability shortcut actions.

### Claim 3: fast horizon propagation

Stochastic TRL propagates reliable long-horizon routes in logarithmic-style sweep budgets.

Required evidence:

- safe path lengths 16, 32, 64, 128;
- grid length 127 budget curve;
- success vs sweeps;
- cost-normalized variant.

### Claim 4: OGBench topology relevance

The same phenomenon appears in OGBench stochastic teleport topology diagnostics.

Required evidence:

- PointMaze teleport navigate/stitch;
- AntMaze teleport navigate/stitch;
- matched Bellman vs stochastic TRL vs full Bellman;
- support TRL ablation;
- learned controller robustness.

### Claim 5: learned modules are feasible but not solved end-to-end

Learned transition and high-level policy heads can preserve the signal, but the current paper does not claim a complete neural stochastic TRL agent.

Required evidence:

- learned transition tables;
- raw-observation transition heads;
- tie-policy / previous-action head diagnostics;
- negative raw value-head diagnostic in limitations.

---

## 4. Priority experiments

## P0. Tighten claim language and manuscript boundary

### Goal

Prevent overclaiming and make the current contribution legible.

### Actions

1. Rename continuous-control experiments as:

   - “topology-level OGBench diagnostics”;
   - “learned-controller execution diagnostics”;
   - “learned high-level module diagnostics.”

2. Avoid phrases like:

   - “end-to-end stochastic TRL”;
   - “solves offline GCRL”;
   - “beats neural baselines” unless those baselines are actually run.

3. Use this main claim:

   > Stochastic TRL matches the long-sweep Bellman reference with far fewer value-propagation sweeps in stochastic goal-reaching MDPs, while deterministic support/realized-path transitive methods over-compose lucky stochastic transitions.

### Deliverable

A revised abstract and contribution list that emphasize value propagation.

---

## P1. Cost-normalized propagation comparison

### Why this is critical

A transitive sweep is more expensive than a Bellman sweep. If the paper only compares 6 transitive sweeps to 180 Bellman sweeps, reviewers may object that the comparison is not compute-normalized.

### Add curves with these x-axes

1. sweep count;
2. wall-clock planning time;
3. number of Bellman-equivalent primitive operations;
4. number of transitive candidates evaluated;
5. memory usage if easy to log.

### Methods

- Bellman value iteration;
- stochastic TRL with all waypoints;
- stochastic TRL with top-k waypoint candidates;
- stochastic TRL with random-k candidates;
- prioritized Bellman / prioritized sweeping;
- Gauss-Seidel Bellman if easy.

### Environments

Start with:

- tabular shortcut safe length 128;
- 16x8 stochastic grid shortcut;
- PointMaze topology MDP.

Then optionally AntMaze topology MDP.

### Candidate waypoint ablation

Run:

```text
K in {4, 8, 16, 32, 64, all}
selection in {top_value, random, dataset_supported, shortest_path_midpoint_if_available}
```

### Metrics

```text
success_rate
regret_to_full_bellman
policy_agreement_with_full_bellman
value_mse_to_full_bellman
planning_wall_clock
candidate_count
memory_peak_optional
```

### Paper figure

One main figure:

> success/regret vs compute budget, with Bellman and stochastic TRL curves.

### Acceptance criterion

Stochastic TRL should remain better than matched Bellman under at least one fair compute normalization, or the paper must explicitly say the contribution is sweep-depth/horizon propagation rather than wall-clock speed.

---

## P2. Shortcut phase diagram

### Why this is critical

This is the cleanest proof that the method understands stochastic risk, not just long paths.

### Environment

Risky shortcut MDP:

- safe path length: `H`;
- risky path length: `d`;
- risky success probability: `p`;
- failure enters trap or long recovery chain.

The optimal boundary is approximately:

```text
safe is better when gamma^H > p * gamma^d
```

### Sweep

```text
H in {8, 16, 32, 64, 128}
p_success in {0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80}
gamma in {0.95, 0.98, 0.995}
trap_escape_length in {absorbing, 16, 64}
```

### Methods

- MC positive;
- MC all-goals;
- realized-path TRL raw;
- realized-path TRL log;
- support TRL;
- Bellman matched;
- stochastic TRL;
- Bellman full.

### Metrics

```text
risky_action_rate
success_rate
regret
predicted_safe_value
predicted_risky_value
true_safe_value
true_risky_value
overestimation_ratio_for_risky_action
```

### Paper figure

A heatmap:

- x-axis: shortcut success probability;
- y-axis: safe path length;
- color: risky action rate;
- overlay: true optimal safe/risky boundary.

### Acceptance criterion

Stochastic TRL should follow the true boundary much more closely than MC-positive, support TRL, and realized-path TRL.

---

## P3. Stochastic calibration curves

### Why this is critical

The word “stochastic” in the paper should be supported by calibration evidence, not only rollout success.

### Add plots

1. predicted reachability vs true DP reachability;
2. predicted risky-action value vs true risky-action value;
3. empirical rollout success binned by predicted value;
4. overestimation index by stochastic row;
5. action-rank accuracy vs full Bellman.

### Metrics

```text
ECE-like calibration error
mean overestimation
95th percentile overestimation
risky-row overestimation
teleport-row overestimation
policy agreement with full Bellman
```

### Environments

- risky shortcut;
- stochastic grid shortcut;
- PointMaze topology MDP;
- AntMaze topology MDP if easy.

### Acceptance criterion

Stochastic TRL should be much less overoptimistic than support/realized-path TRL and should approach full Bellman calibration after far fewer sweeps.

---

## P4. External baselines

### Why this is critical

Bellman baselines prove the operator story, but high-impact reviewers will ask whether the result matters relative to existing offline GCRL methods.

### Minimum baseline package

Run these on the smallest environments first:

1. deterministic/support TRL;
2. original TRL-style realized-path backup;
3. CRL or contrastive successor-feature baseline;
4. QRL/CMD-style quasimetric baseline if available;
5. TMD if feasible;
6. prioritized sweeping or Gauss-Seidel Bellman.

### Baseline levels

#### Level 1: tabular/topology baselines

Implement method analogues on the same tabular/topology MDPs. This is fastest and most diagnostic.

#### Level 2: OGBench reference baselines

Use OGBench-provided baselines where possible:

- GCBC;
- GCIVL;
- GCIQL;
- CRL;
- QRL;
- HIQL.

#### Level 3: stochastic/quasimetric baselines

Add TMD and MQE if code/hyperparameters are available.

### Which baseline matters most?

TMD is the most important, because it is closest to the stochastic temporal-distance story.

### Environments for external baselines

Start with:

- risky shortcut;
- 2D stochastic grid;
- PointMaze teleport topology or learned-transition MDP.

Then:

- OGBench PointMaze teleport with learned waypoint executor;
- OGBench AntMaze teleport hard tasks.

### Acceptance criterion

For the operator paper, stochastic TRL does not need to beat every neural baseline end-to-end. It needs to show a unique advantage in **matched propagation budget** and **stochastic calibration**.

---

## P5. Finite-sample transition robustness

### Why this is critical

The algorithm assumes an empirical transition model. Reviewers will ask how sensitive it is to transition-estimation error.

### Experiment

For each stochastic row, estimate transitions from finite samples:

```text
samples_per_row in {1, 2, 5, 10, 20, 50, 100}
Dirichlet_smoothing in {0.0, 0.1, 1.0}
transition_seed in {0, 1, 2, 3, 4}
```

### Environments

- risky shortcut;
- grid shortcut;
- PointMaze topology;
- AntMaze topology hard slice.

### Methods

- Bellman matched;
- stochastic TRL;
- Bellman full under true model;
- Bellman full under estimated model;
- support TRL.

### Metrics

```text
success_rate
regret_to_true_model_optimum
regret_to_estimated_model_full_bellman
transition_l1
transition_kl
policy_agreement_with_true_full_bellman
risky_action_rate
```

### Paper figure

Success/regret vs samples per stochastic row.

### Acceptance criterion

Stochastic TRL should retain its advantage once stochastic rows have modest sample support, and degradation should be explainable by transition-estimation error rather than transitive backup failure.

---

## P6. Privilege ladder for OGBench topology

### Why this is critical

The current OGBench results are strong, but reviewers may worry that topology or task information is too privileged.

### Add a table with these rows

| Setting | Uses env map? | Uses teleport metadata? | Uses task init/goal cells? | Uses offline jumps? | Uses learned transition? | Main/appendix |
|---|---:|---:|---:|---:|---:|---|
| env topology | yes | yes | yes | no | no | appendix only |
| dataset topology | limited/no | no | yes | yes | no | main if honest |
| learned transition table | limited/no | no | yes | yes | yes | main/appendix |
| raw-observation transition MLP | limited/no | no | yes | yes | yes | stronger appendix |
| raw transition + learned high-level policy head | limited/no | no | yes | yes | yes | strongest learned-module diagnostic |
| direct neural end-to-end | no | no | no/standard eval goals | dataset only | yes | future/main only if successful |

### Actions

1. Promote the least-privileged successful setting in the main paper.
2. Move env-topology oracle results to appendix.
3. Explicitly mark task-info usage as evaluation-goal conditioning, not training leakage, if that is accurate.
4. Include negative less-privileged graph/executor checks in limitations.

### Acceptance criterion

A reviewer should be able to tell exactly what information each experiment uses.

---

## P7. Learned high-level module bridge

### Why this is important

You need a bridge between table planning and neural value learning, but you do not need full end-to-end control immediately.

### Current lesson

The raw scalar value head can have low MSE but fail in rollout because small ranking/tie errors around bottlenecks break high-level control. The tie-policy and previous-action heads are positive diagnostics.

### Next experiments

#### P7a. Distill stochastic-TRL Q into a policy set, not a single argmax

Train a high-level policy head to predict all near-optimal actions:

```text
label(a) = 1[Q(s,a,g) >= max_a Q(s,a,g) - epsilon]
```

Vary:

```text
epsilon in {1e-6, 1e-4, 1e-3, 1e-2}
previous_action in {off, on}
```

#### P7b. Ranking loss for action gaps

Use pairwise ranking:

```text
max(0, margin - Q_head(s,a_good,g) + Q_head(s,a_bad,g))
```

Focus on bottleneck and teleporter-adjacent states.

#### P7c. Localized weighted value loss

Weight value errors by decision importance:

```text
weight(s,g) = 1 + alpha * 1[action_gap_small or state_near_teleporter or bottleneck]
```

#### P7d. Neural stochastic-TRL auxiliary loss

Try a small value network trained with:

```text
L = L_Bellman_expectation + lambda * L_transitive_sampled_waypoint + L_action_ranking
```

Use sampled waypoints first, not dense all-waypoint max.

### Acceptance criterion

A learned high-level head should preserve at least 90% of the table-planner success on PointMaze teleport stitch and AntMaze hard slices.

---

## P8. Harder OGBench tasks

### When to run them

Run harder tasks after P1-P6. Otherwise failures will be hard to interpret.

### Priority order

1. PointMaze teleport stitch hard task IDs, already promising.
2. AntMaze teleport hard task IDs 4 and 5, already promising.
3. More AntMaze tasks / all tasks with the same controller.
4. Non-teleport stochastic variants if you can construct them cleanly.
5. Standard OGBench non-teleport long-horizon tasks only after you have external baselines wired.

### What to compare

For topology/value-propagation paper:

- matched Bellman;
- full Bellman;
- support TRL;
- stochastic TRL;
- prioritized Bellman;
- at least one stochastic/quasimetric baseline.

For full offline GCRL paper:

- GCBC;
- GCIVL;
- GCIQL;
- CRL;
- QRL;
- HIQL;
- TRL;
- TMD/MQE if possible.

### Controller protocol

Try to use:

- one controller per environment family, not one per task type;
- multiple controller seeds;
- the same controller across matched Bellman, full Bellman, support TRL, stochastic TRL;
- fixed episode counts and task IDs.

### Acceptance criterion

Harder OGBench tasks should strengthen the generality story, not replace the controlled stochastic evidence.

---

## 5. Theory and proof checklist

### Theorem 1: lower-bound preservation

Prove:

If \(0 \le Q \le Q^*\), then:

```text
B Q <= Q*
T Q <= Q*
max(BQ, TQ) <= Q*
```

Important details:

- define \(Q^*\) as optimal discounted hitting value;
- prove the product term corresponds to a feasible composite policy;
- handle the case where the final goal is hit before the intermediate waypoint;
- state assumptions about absorbing goal convention.

### Theorem 2: deterministic logarithmic propagation

Prove:

In deterministic paths initialized with one-step reachability, transitive sweeps propagate segments of length \(2^k\).

### Theorem 3: stochastic shortcut calibration

Formalize the risky shortcut example:

- safe value: \(\gamma^H\);
- risky value: \(p\gamma^d\) or \(p\gamma^d +\) recovery term;
- realized/support TRL can estimate near \(\gamma^d\);
- stochastic TRL estimates the expectation and chooses safe when \(\gamma^H > p\gamma^d\).

### Theorem 4 or proposition: monotonic convergence

Because the operator is monotone and bounded above by \(Q^*\), establish convergence to a fixed point. Then clarify whether the fixed point equals \(Q^*\) in all finite MDPs or is simply a safe lower bound accelerated by transitive composition.

This is important. Do not overclaim equality unless you have the proof.

---

## 6. Main paper figures and tables

### Figure 1: stochastic failure mode

Risky shortcut cartoon:

- safe route;
- risky teleporter/shortcut;
- realized-path TRL over-composes lucky success;
- stochastic TRL composes calibrated probabilities.

### Figure 2: operator diagram

Show:

```text
Bellman expectation backup + transitive composition backup -> max update
```

### Figure 3: shortcut phase diagram

Risky action rate vs \(p\) and \(H\), with true optimal boundary.

### Figure 4: horizon propagation curve

Success/regret vs sweeps and cost-normalized budget.

### Figure 5: OGBench topology result

PointMaze and AntMaze teleport:

- Bellman matched;
- support TRL;
- stochastic TRL;
- Bellman full.

### Figure 6: calibration plot

Predicted reachability vs true/empirical reachability.

### Table 1: main OGBench topology diagnostics

Use current main hard-task table, but label it clearly as topology / learned-controller diagnostics.

### Table 2: ablations

- no Bellman calibration;
- support TRL;
- stochastic TRL;
- waypoint candidate count;
- transition sample count.

### Table 3: learned-module ladder

- table transition/table Q;
- learned transition/table Q;
- learned transition/tie-policy head;
- previous-action policy head;
- raw scalar value head negative result.

---

## 7. Suggested repo additions

### New scripts

```text
scripts/run_cost_normalized_planning.py
scripts/run_shortcut_phase_diagram.py
scripts/run_transition_sample_robustness.py
scripts/run_calibration_curves.py
scripts/run_external_baseline_tabular.py
scripts/generate_high_impact_figures.py
```

### New result tables

```text
results/paper_tables/cost_normalized_planning.csv
results/paper_tables/shortcut_phase_diagram.csv
results/paper_tables/transition_sample_robustness.csv
results/paper_tables/calibration_metrics.csv
results/paper_tables/external_baselines.csv
```

### New figures

```text
results/figures/cost_normalized_planning.pdf
results/figures/shortcut_phase_diagram.pdf
results/figures/calibration_curves.pdf
results/figures/transition_sample_robustness.pdf
results/figures/ogbench_topology_main.pdf
```

### Verification updates

Extend `scripts/verify_main_claims.py` to check:

- cost-normalized result files exist and match source CSVs;
- stochastic TRL beats matched Bellman under at least one fair compute metric;
- shortcut phase diagram follows true optimal boundary above a threshold;
- support TRL remains worse than stochastic TRL on stochastic rows;
- learned-transition robustness covers specified transition seeds.

---

## 8. Go / no-go criteria

### Continue toward a high-impact submission if

1. Cost-normalized curves still show a meaningful propagation advantage.
2. Shortcut phase diagram matches the true stochastic safe/risky boundary.
3. Stochastic TRL is better calibrated than support/realized-path TRL.
4. The OGBench topology result survives at the least-privileged successful setting.
5. At least one external stochastic/quasimetric baseline is included or carefully discussed.
6. Learned transition or learned high-level policy diagnostics remain positive.

### Weaken the claim if

1. The advantage disappears under wall-clock or operation-normalized comparison.
2. Stochastic TRL only wins because the task is a hand-designed teleport shortcut.
3. TMD/MQE or another baseline solves the same diagnostics equally well.
4. Learned transition errors destroy the advantage at realistic sample counts.
5. The least-privileged OGBench setting fails while only env-topology oracle succeeds.

### Pivot if

1. The operator is not actually lower-bound preserving under the intended stochastic assumptions.
2. The fixed point is systematically too conservative and cannot recover full Bellman on non-toy stochastic MDPs.
3. All learned-module attempts fail and the result cannot be framed cleanly as value propagation.

---

## 9. Immediate task order

### Step 1: manuscript framing pass

Update the abstract, intro, and limitations around:

```text
value propagation, not end-to-end control
stochastic topology diagnostics, not full neural agent
matched sweep budget, not automatic compute superiority
```

### Step 2: cost-normalized planning

Add operation and wall-clock accounting to tabular/grid/PointMaze topology runs.

### Step 3: shortcut phase diagram

Generate the safest and clearest stochastic calibration figure.

### Step 4: calibration metrics

Add predicted-vs-true and overestimation plots.

### Step 5: external baseline package

Start with tabular/topology analogues, then add TMD/CRL/QRL where feasible.

### Step 6: OGBench least-privilege cleanup

Make a privilege ladder table and promote the least-privileged successful setting.

### Step 7: learned high-level head improvement

Improve or clarify the raw-observation policy/value bridge.

### Step 8: final hard-task scaling

Only after the above, run additional OGBench hard tasks or broader task sets.

---

## 10. Suggested final paper framing

### Title options

1. **Stochastic Transitive RL: Calibrated Divide-and-Conquer Value Propagation**
2. **Calibrated Transitive Value Propagation for Stochastic Goal Reaching**
3. **Stochastic Transitive Planning for Offline Goal-Conditioned RL**

### Best abstract angle

> Long-horizon goal-conditioned value learning benefits from transitive composition, but deterministic transitive backups over-compose lucky stochastic transitions. We introduce a Bellman-calibrated stochastic transitive operator that composes reachability probabilities rather than realized support. The operator preserves stochastic calibration while propagating reliable long-horizon routes in logarithmic-style sweep budgets. Controlled stochastic shortcut tasks, stochastic grid corridors, and OGBench teleport topology diagnostics show that stochastic TRL matches long-sweep Bellman references with far fewer propagation sweeps, while support/realized-path TRL and matched Bellman fail in complementary ways.

### Best limitation sentence

> Our continuous-control experiments intentionally isolate high-level value propagation using empirical topology or learned high-level modules with shared low-level executors; they should be read as stochastic value-learning diagnostics rather than a complete end-to-end neural stochastic TRL agent.

---

## 11. Bottom line

You are right to follow TRL’s value-learning isolation. That is the cleanest way to make the stochastic extension scientifically meaningful.

The highest-impact path is not simply “run harder OGBench.” The highest-impact path is:

1. prove and visualize the stochastic failure mode;
2. show the calibrated transitive operator fixes it;
3. make the sweep advantage fair under compute normalization;
4. compare to at least one serious stochastic/quasimetric baseline;
5. keep the OGBench topology diagnostics honest and least-privileged;
6. include learned-module evidence as a bridge, not as an overclaim.
