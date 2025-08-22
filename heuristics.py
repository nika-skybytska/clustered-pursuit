# Two schedulers for UAV–target interception problem:
# - Nearest Neighbour heuristic
# - Clustered + Nearest Neighbour heuristic

from dataclasses import dataclass
from typing import Optional
import heapq
import numpy as np

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from model import Scenario, UAV, Target, intercept_uav_target, InterceptSolution


@dataclass(frozen=True)
class InterceptEvent:
    uav_id: str
    target_id: str
    t_hit: float
    p_hit: tuple[float, float]
    leg_time: float
    leg_distance: float


@dataclass(frozen=True)
class ScheduleReport:
    success: bool
    total_distance: float
    makespan: float
    events: list[InterceptEvent]
    unreached_targets: list[str]


def schedule_nn(scenario: Scenario) -> ScheduleReport:
    """
    Greedy Nearest Neighbour heuristic:
    - At each step, take UAV that becomes available earliest
    - Intercept the target that can be reached soonest
    """
    unassigned = {t.id: t for t in scenario.targets}
    events: list[InterceptEvent] = []

    # Priority queue: (time_available, counter, uav_id, position)
    pq: list[tuple[float, int, str, np.ndarray]] = []
    counter = 0
    for u in scenario.uavs:
        pq.append((0.0, counter, u.id, u.p0.copy()))
        counter += 1
    heapq.heapify(pq)

    # Keep track of cumulative distances per UAV
    dist_accum: dict[str, float] = {u.id: 0.0 for u in scenario.uavs}

    while unassigned and pq:
        t_avail, _, uid, pos = heapq.heappop(pq)
        uav = scenario.uav_by_id(uid)

        # Find best next target for this UAV
        best: Optional[tuple[float, Target, InterceptSolution]] = None
        for tgt in unassigned.values():
            sol = intercept_uav_target(uav, pos, t_avail, tgt)
            if sol is None:
                continue
            if best is None or sol.t_hit < best[0]:
                best = (sol.t_hit, tgt, sol)

        if best is None:
            # UAV cannot reach any remaining targets: skip it
            continue

        _, tgt, sol = best
        # Record event
        ev = InterceptEvent(
            uav_id=uid,
            target_id=tgt.id,
            t_hit=sol.t_hit,
            p_hit=(float(sol.p_hit[0]), float(sol.p_hit[1])),
            leg_time=sol.leg_time,
            leg_distance=sol.leg_distance,
        )
        events.append(ev)
        dist_accum[uid] += sol.leg_distance

        # Update UAV state
        del unassigned[tgt.id]
        heapq.heappush(pq, (sol.t_hit, counter, uid, sol.p_hit))
        counter += 1

    total_distance = sum(dist_accum.values())
    makespan = max(dist_accum.values()) if dist_accum else 0.0
    success = len(unassigned) == 0

    return ScheduleReport(
        success=success,
        total_distance=total_distance,
        makespan=makespan,
        events=events,
        unreached_targets=list(unassigned.keys()),
    )


def schedule_clustered_nn(scenario: Scenario, seed: int = 0) -> ScheduleReport:
    """
    Clustered heuristic:
    - Cluster targets into n_uavs groups by initial positions (k-means using sklearn).
    - Assign clusters to UAVs using an optimal one-to-one assignment (Hungarian algorithm).
    - Within each cluster, run NN heuristic restricted to its UAV.
    """
    nU = len(scenario.uavs)
    if nU == 0:
        return ScheduleReport(False, 0.0, 0.0, [], [t.id for t in scenario.targets])

    # Gather positions
    points = np.array([t.p0 for t in scenario.targets])
    rng = np.random.default_rng(seed)

    # Use sklearn's KMeans for clustering (more robust than a naive implementation).
    # Note: sklearn requires n_samples >= n_clusters.
    if len(points) == 0:
        # no targets
        return ScheduleReport(True, 0.0, 0.0, [], [])
    if len(points) < nU:
        # fewer targets than clusters: fall back to making each target its own cluster
        # and fewer clusters than UAVs; we still proceed by clustering with n_clusters = len(points)
        kmeans_k = len(points)
    else:
        kmeans_k = nU

    km = KMeans(n_clusters=kmeans_k, random_state=seed, n_init=10)
    labels = km.fit_predict(points)
    centroids = km.cluster_centers_

    # If we used fewer clusters than UAVs (because fewer targets), we will assign those clusters
    # to a subset of UAVs. Build distance matrix shape (n_uavs, n_clusters)
    uav_positions = np.array([u.p0 for u in scenario.uavs])
    dist_matrix = np.linalg.norm(
        uav_positions[:, None, :] - centroids[None, :, :], axis=2
    )

    # Optimal one-to-one assignment between UAVs and clusters.
    # linear_sum_assignment returns row indices (uav idx) and col indices (cluster idx)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    # Build cluster -> uav index mapping (for clusters that were assigned)
    assignment = np.full(centroids.shape[0], -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        assignment[c] = r

    events: list[InterceptEvent] = []
    dist_accum: dict[str, float] = {u.id: 0.0 for u in scenario.uavs}
    unreached: list[str] = []

    # For each cluster, run NN for its assigned UAV
    for cid in range(centroids.shape[0]):
        uidx = assignment[cid]
        if uidx < 0:
            # cluster wasn't assigned (shouldn't normally happen), mark targets unreachable
            cluster_targets = [
                scenario.targets[i] for i, lab in enumerate(labels) if lab == cid
            ]
            unreached.extend([t.id for t in cluster_targets])
            continue

        uav = scenario.uavs[uidx]
        # Targets in this cluster
        cluster_targets = [
            scenario.targets[i] for i, lab in enumerate(labels) if lab == cid
        ]
        unassigned = {t.id: t for t in cluster_targets}
        t_avail = 0.0
        pos = uav.p0.copy()

        while unassigned:
            best: Optional[tuple[float, Target, InterceptSolution]] = None
            for tgt in unassigned.values():
                sol = intercept_uav_target(uav, pos, t_avail, tgt)
                if sol is None:
                    continue
                if best is None or sol.t_hit < best[0]:
                    best = (sol.t_hit, tgt, sol)
            if best is None:
                # remaining targets in cluster unreachable
                unreached.extend(list(unassigned.keys()))
                break
            _, tgt, sol = best
            ev = InterceptEvent(
                uav_id=uav.id,
                target_id=tgt.id,
                t_hit=sol.t_hit,
                p_hit=(float(sol.p_hit[0]), float(sol.p_hit[1])),
                leg_time=sol.leg_time,
                leg_distance=sol.leg_distance,
            )
            events.append(ev)
            dist_accum[uav.id] += sol.leg_distance
            pos = sol.p_hit
            t_avail = sol.t_hit
            del unassigned[tgt.id]

    total_distance = sum(dist_accum.values())
    makespan = max(dist_accum.values()) if dist_accum else 0.0
    success = len(unreached) == 0

    return ScheduleReport(
        success=success,
        total_distance=total_distance,
        makespan=makespan,
        events=events,
        unreached_targets=unreached,
    )


if __name__ == "__main__":
    from generators import GenParams, generate_scenario

    params = GenParams(
        n_targets=10,
        n_uavs=2,
        tgt_speed_min=1,
        tgt_speed_max=3,
        uav_speed_min=10,
        uav_speed_max=10,
        scenario_prefix="test",
    )

    sc = generate_scenario("random", params, seed=1, scenario_id="sc1")
    rep1 = schedule_nn(sc)
    print(
        "NN:",
        rep1.success,
        "total_dist=",
        rep1.total_distance,
        "makespan=",
        rep1.makespan,
        "events=",
        len(rep1.events),
    )

    rep2 = schedule_clustered_nn(sc, seed=1)
    print(
        "Clustered NN:",
        rep2.success,
        "total_dist=",
        rep2.total_distance,
        "makespan=",
        rep2.makespan,
        "events=",
        len(rep2.events),
    )
