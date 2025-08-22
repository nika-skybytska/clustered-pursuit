# Multi-UAV Interception Heuristics

**Routing multiple UAVs to intercept moving targets** — this repository implements a small, reproducible experimental framework for the problem where targets move linearly in 2D and UAVs have fixed top speeds. It contains:

* a compact geometric/kinematic model with **closed-form interception** (`model.py`),
* a flexible **random scenario generator** and JSON serde (`generators.py`),
* two heuristic schedulers (global nearest-neighbour and clustered + nearest-neighbour) (`heuristics.py`),
* an experiments notebook (script-style) that generates test suites, visualizes scenarios, runs batched experiments, and saves results.

This README documents the system design, algorithms, the math behind interception, how to run the code, output formats, and practical conclusions from the experiments.

---

## Table of contents

1. [Project overview](#project-overview)
2. [System design and components](#system-design-and-components)
3. [Key mathematics](#key-mathematics)

   * Target and UAV kinematics
   * Closed-form intercept equation and derivation
   * Numerical stability considerations
4. [Algorithms implemented](#algorithms-implemented)

   * Global nearest-neighbour (NN)
   * Clustered NN (k-means + Hungarian assignment + NN per UAV)
   * Complexity and practical properties
5. [Data formats (JSON) and API examples](#data-formats-json-and-api-examples)
6. [How to run experiments (quickstart)](#how-to-run-experiments-quickstart)
7. [Interpreting the results and qualitative conclusions](#interpreting-the-results-and-qualitative-conclusions)
8. [Limitations, caveats and future work](#limitations-caveats-and-future-work)
9. [Dependencies](#dependencies)

---

## Project overview

We address the following problem:

> Given a set of moving targets (points in 2D with constant velocity) and several UAVs with maximum speed, produce routing (sequence of interception events) for each UAV so all targets are intercepted if feasible.

Two heuristics are implemented and compared by two metrics:

* **total route length** — sum of distances traveled by all UAVs (proxy for fuel/time/cost), and
* **makespan** — maximum distance among individual UAV routes (proxy for last finishing UAV).

The framework is intended to be reproducible, modular, and easy to extend for research or classroom experiments.

---

## System design and components

Files / modules:

* **`model.py`**

  * Core dataclasses: `Target`, `UAV`, `Scenario`, and `InterceptSolution`.
  * `intercept_time(...)`: closed-form solver that returns earliest interception time and interception point when a UAV (position at time `t0`, maximum speed `s`) can catch a target moving as `p(t) = p0 + v * t`.
  * Utility math (vector conversions, stable quadratic solver).

* **`generators.py`**

  * `GenParams` dataclass to control generation (n\_targets, n\_uavs, area, speeds, cluster radius, direction concentration, etc.).
  * `generate_scenario(case, params, seed, scenario_id)` produces one scenario for each of the four case kinds: `"random"`, `"pos_clusters"`, `"dir_clusters"`, `"posdir_clusters"`.
  * `generate_suite(...)` creates many scenarios in a reproducible fashion.
  * JSON serde (`scenario_to_json_dict`, `scenarios_to_json_str`, `save_scenarios_json`, `scenario_from_json_dict`, `load_scenarios_json`) so tests can be saved and reloaded.

* **`heuristics.py`**

  * Two schedulers with the same output format (`ScheduleReport` and `InterceptEvent` dataclasses):

    * `schedule_nn(scenario)` — global nearest-neighbour greedy scheduler using a priority queue keyed by UAV availability time.
    * `schedule_clustered_nn(scenario, seed=0)` — clusters targets (sklearn `KMeans`), assigns clusters to UAVs using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), then runs NN inside each cluster.
  * Both schedulers report: `success` (all targets intercepted), `total_distance`, `makespan`, `events` (list of intercepts), and `unreached_targets`.

* **`experiments_notebook.py`** (created as a notebook-like script)

  * Visualizes representative scenarios (2×2 grid).
  * Runs batched experiments (user-settable runs per case), collects metrics, displays styled pandas tables and boxplots, and saves the generated test suite and summary CSVs in `./tests`.

---

## Key mathematics

### Motion model

* Each **target** is a point moving linearly:

  $$
  p(t) = p_0 + v\,t
  $$

  where $p_0 \in \mathbb{R}^2$ is position at $t=0$ and $v\in\mathbb{R}^2$ is constant velocity.

* Each **UAV** is a point that can move with speed at most $s > 0$. The scheduler controls the UAV path; we assume straight-line travel between interception events (i.e., UAV chooses to fly directly to the interception point at max speed).

### Interception equation

Suppose a UAV is at position $u_0$ at time $t_0$ and a target is described by $p(t)$. We want the smallest $t \ge t_0$ satisfying:

$$
\|p(t) - u_0\| = s (t - t_0).
$$

Substitute $p(t) = p_0 + v t$ and define $\Delta = p_0 - u_0$. The equation becomes:

$$
\|\Delta + v t\|^2 = s^2 (t - t_0)^2.
$$

Expanding both sides yields a quadratic in $t$:

$$
A t^2 + B t + C = 0,
$$

with

$$
\begin{aligned}
A &= \|v\|^2 - s^2,\\
B &= 2( \Delta\cdot v + s^2 t_0),\\
C &= \|\Delta\|^2 - s^2 t_0^2.
\end{aligned}
$$

* If $A \neq 0$, use the numerically stable quadratic solver (`_solve_quadratic_stable` in `model.py`) to get real roots and choose the smallest $t \ge t_0$.
* If $A$ is (near) zero the equation is linear and handled separately.
* Immediate co-location (target at UAV at $t_0$) returns an intercept with zero leg time.

This closed-form approach yields exact interception times (subject to floating point precision) and avoids costly numeric root-finding.

### Numerical stability

* The quadratic solver uses a standard trick to compute roots stably (avoid catastrophic cancellation) and is guarded against degenerate coefficients (`A ~ 0`, etc.).
* Tolerances are used to handle floating point edge cases (coincident positions, tiny negative discriminants due to rounding, and so on).

---

## Algorithms implemented

### 1) Global Nearest-Neighbour (`schedule_nn`)

Procedure summary:

1. Keep a **priority queue** (min-heap) of UAVs keyed by their availability times (initially all 0).
2. Pop the UAV that is free the earliest, compute intercept solutions to *every* remaining target (using the closed-form solver), choose the target with earliest intercept time, and assign it to the UAV.
3. Update UAV available time and position, push it back into the queue.
4. Repeat until no targets remain or no UAV can reach any remaining targets.

Properties:

* Simple, greedy and easy to implement.
* Works well when targets are widely distributed and UAVs share the same capabilities.
* Complexity per assignment step: compute intercept for all remaining targets → $O(T)$ per assignment; overall worst-case $O(T^2)$ (where $T$ = number of targets), multiplied by the small factor of UAVs and intercept computation cost.

### 2) Clustered + Nearest-Neighbour (`schedule_clustered_nn`)

Procedure summary:

1. Use **k-means** (via `sklearn.cluster.KMeans`) to form `k = n_uavs` spatial clusters of targets (or `k = min(n_uavs, n_targets)` if fewer targets).
2. Assign clusters to UAVs with a **one-to-one optimal matching** by minimizing UAV-to-cluster centroid distance using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
3. For each cluster assigned to a UAV, run the NN heuristic restricted to targets in that cluster (UAV starts at its own base/initial position).

Properties and motivations:

* Intuition: if targets naturally form spatial groups and you have roughly the same number of UAVs, distributing clusters to UAVs reduces travel overlap and should reduce makespan.
* One-to-one assignment avoids concentrating several clusters on the same UAV (bug fixed during development).
* KMeans provides robust clustering and Hungarian ensures balanced assignment.

---

## Data formats (JSON) and API examples

### Scenario JSON format (example)

Each scenario saved by `generators.save_scenarios_json` is an element in a list. A scenario dict looks like:

```json
{
  "id": "sc_random_001",
  "targets": [
    {"id": "t1", "p0": [100.0, -50.0], "v": [1.2, 0.3]},
    {"id": "t2", "p0": [-30.0, 80.0], "v": [0.0, -2.0]}
  ],
  "uavs": [
    {"id": "u1", "p0": [0.0, 0.0], "speed": 30.0},
    {"id": "u2", "p0": [100.0, 0.0], "speed": 30.0}
  ],
  "metadata": {
    "case": "random",
    "seed": 42,
    "params": { /* GenParams snapshot */ }
  }
}
```

### Example Python usage

```python
from generators import GenParams, generate_scenario, save_scenarios_json
from heuristics import schedule_nn, schedule_clustered_nn

params = GenParams(n_targets=30, n_uavs=4)
sc = generate_scenario("pos_clusters", params, seed=42, scenario_id="example1")

# run NN
rep_nn = schedule_nn(sc)

# run clustered NN
rep_cl = schedule_clustered_nn(sc, seed=42)

print("NN total:", rep_nn.total_distance, "makespan:", rep_nn.makespan)
print("Clustered total:", rep_cl.total_distance, "makespan:", rep_cl.makespan)
```

### `ScheduleReport` (returned by schedulers)

A `ScheduleReport` dataclass contains:

* `success` (bool): whether all targets were intercepted.
* `total_distance` (float): sum of distances flown by UAVs.
* `makespan` (float): maximum distance by any single UAV.
* `events` (list of `InterceptEvent`): each event contains `uav_id, target_id, t_hit, p_hit, leg_time, leg_distance`.
* `unreached_targets` (list of target IDs) for targets that could not be intercepted by any UAV (e.g., faster than UAV and moving away).

---

## How to run experiments (quickstart)

### 1. Install dependencies

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

(Use a virtual environment for reproducibility.)

### 2. Generate scenarios and run a single scheduling instance

Run a Python REPL or create a short script:

```python
from generators import GenParams, generate_scenario
from heuristics import schedule_nn, schedule_clustered_nn

params = GenParams(n_targets=30, n_uavs=4)
sc = generate_scenario("pos_clusters", params, seed=123)
report = schedule_nn(sc)
print(report)
```

### 3. Run the experiments notebook

The provided `experiments_notebook.py` acts as notebook cells. Open it in Jupyter or VS Code and run the cells. It will:

* generate representative visualizations (2×2),
* generate a full suite of scenarios (saved in `./tests/generated_suite.json`),
* run both schedulers for each scenario,
* produce per-case summary tables and boxplots,
* save CSV summaries in `./tests/summaries`.

Adjust `N_PER_CASE` at the top of the notebook to trade off runtime vs statistical confidence.

---

## Interpreting results and qualitative conclusions

> **Important:** do not treat the experimental tables in this repository as universal truth — they are specific to the generator parameters, seed choices, UAV speeds, and the heuristics implemented. The notebook is designed so you can reproduce, vary parameters and investigate deeper.

Some consistent, qualitative patterns typically observed:

* **Clustering helps when targets are spatially grouped** and the number of clusters roughly matches the number of UAVs: clustered NN usually reduces the *makespan* (last UAV finishing earlier) by distributing work geographically. Total distance may or may not improve substantially depending on how clusters are oriented and target directions.

* **Clustering can hurt or be neutral** when clusters don't align with UAV starting positions or when targets inside clusters move in very different directions: assigning one UAV to a spatial cluster does not account for target velocity, and a UAV may have to chase many fast-moving targets inside its cluster, increasing its route length.

* **Global NN is robust but greedy**: it adapts to interceptability (targets unreachable by some UAVs are naturally handled), but it may produce imbalanced load across UAVs if many early-interceptable targets are near one UAV.

* **Unreachable targets** appear when target speeds or movement vectors make interception impossible by available UAVs (e.g., target speed > UAV speed and moving away). Both schedulers properly report these.

Practical recommendation: run experiments with domain-appropriate generator parameters (UAV base positions, speed ranges, cluster radius, von Mises concentration for directions) and compare strategies for your operational regime.

---

## Limitations, caveats and future work

* **No optimality guarantees**: both algorithms are heuristics. There are NP-hard variants of vehicle routing and pursuit-evasion problems that are beyond simple closed-form heuristics.

* **Simplified UAV model**: straight-line constant-speed travel between interceptions (no acceleration/deceleration, no turn-rate limits).

* **No collision / airspace constraints**: UAVs are independent; no no-fly zones, no inter-UAV separation constraints.

* **Clustering only uses initial target positions** (KMeans). Better clustering could incorporate velocity (direction/speed) or interceptability metrics.

* **Scheduling horizon & concurrency**: The schedulers assume each UAV completes a discrete list of legs. Extensions could consider continuous replanning or lookahead heuristics.

**Future work ideas**

* Use richer features in clustering (position + velocity), or cluster in the interceptability metric space.
* Implement more sophisticated multi-agent assignment (e.g., min-sum-of-completion-time) or metaheuristics (genetic algorithms).
* Account for fuel, time windows, or heterogeneous UAV capabilities.
* Simulate continuous replanning where UAVs update intercept decisions in-flight.

---

## Dependencies

* Python 3.8+
* `numpy`
* `scipy`
* `scikit-learn`
* `pandas` (for experiments)
* `matplotlib` (for visualizations)

Install with:

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

---

## Where outputs are saved

* Generated test suite JSON: `./tests/generated_suite.json`
* Per-case CSV summaries (from the experiments notebook): `./tests/summaries/summary_<case>.csv`
* You can modify the notebook to save additional diagnostic outputs (per-scenario logs, per-UAV routes, etc.).

---

## Final notes & reproducibility

* All random generation is seeded via `numpy` RNGs; supply a seed to `generate_scenario` or `generate_suite` for reproducibility.
* The experiments notebook uses explicit seeds to make visualizations and batch runs reproducible by default.

Happy to help you extend this baseline into more advanced planners or to format a set of publication-quality figures from the experiments.
