# generators.py
# Randomized UAV–moving-target scenario generation + serde utilities.


from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Literal, Any
import json
import math
import time

import numpy as np

from model import Target, UAV, Scenario, as_vec2


CaseKind = Literal[
    "random",  # 1) completely random positions and directions
    "pos_clusters",  # 2) clustered positions, diverse directions
    "dir_clusters",  # 3) diverse positions, clustered directions
    "posdir_clusters",  # 4) clustered positions and directions
]


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle for sampling positions (inclusive of min, exclusive of max)."""

    x_min: float = -1000.0
    x_max: float = 1000.0
    y_min: float = -1000.0
    y_max: float = 1000.0

    def sample_uniform(self, rng: np.random.Generator, n: int) -> np.ndarray:
        xs = rng.uniform(self.x_min, self.x_max, size=n)
        ys = rng.uniform(self.y_min, self.y_max, size=n)
        return np.stack([xs, ys], axis=1)

    def clamp_point(self, p: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(p[0], self.x_min, self.x_max),
                np.clip(p[1], self.y_min, self.y_max),
            ],
            dtype=float,
        )

    def width(self) -> float:
        return float(self.x_max - self.x_min)

    def height(self) -> float:
        return float(self.y_max - self.y_min)


@dataclass(frozen=True)
class GenParams:
    """
    Core parameters for scenario generation.

    Notes
    -----
    - All randomness is driven by `seed` via numpy's BitGenerator.
    - For clustered cases, `n_clusters` defaults to `n_uavs` unless set explicitly.
    - Position clusters: points are placed inside discs of radius `cluster_pos_radius`.
    - Direction clusters: angles are drawn from von Mises with concentration `dir_kappa`.
      (kappa=0 ≡ uniform; higher kappa ≡ tighter around the cluster mean angle.)
    """

    # Scenario size
    n_targets: int = 50
    n_uavs: int = 4

    # Spatial domain for targets and UAVs
    area: Box = Box()

    # Target speeds
    tgt_speed_min: float = 5.0
    tgt_speed_max: float = 20.0

    # UAV speeds (either fixed per UAV or sampled)
    uav_speed_min: float = 30.0
    uav_speed_max: float = 30.0  # same as min -> constant speed

    # UAV initial positions
    uav_positions_mode: Literal["fixed", "random"] = "fixed"
    uav_fixed_positions: Optional[list[tuple[float, float]]] = (
        None  # default: all at (0,0)
    )

    # Cluster controls (used by clustered cases)
    n_clusters: Optional[int] = None  # defaults to n_uavs if None
    cluster_pos_radius: float = 150.0  # radius of position discs
    dir_kappa: float = 8.0  # concentration for von Mises (tight when >=5)

    # Scenario ID/labeling
    scenario_prefix: str = "sc"
    tgt_prefix: str = "t"
    uav_prefix: str = "u"


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def _unit_vec_from_angle(theta: np.ndarray) -> np.ndarray:
    """Convert array of angles (rad) to unit vectors shape (n,2)."""
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _sample_speeds(
    rng: np.random.Generator, n: int, vmin: float, vmax: float
) -> np.ndarray:
    if vmax < vmin:
        raise ValueError("tgt_speed_max must be >= tgt_speed_min")
    if math.isclose(vmin, vmax):
        return np.full(n, float(vmin))
    return rng.uniform(vmin, vmax, size=n)


def _split_counts(n: int, k: int) -> list[int]:
    """Nearly-even nonnegative split of n items into k bins that sum to n."""
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def _sample_in_discs(
    rng: np.random.Generator,
    centers: np.ndarray,  # shape (k,2)
    counts: list[int],
    radius: float,
) -> np.ndarray:
    """Sample points inside discs of given radius around centers (uniform by area)."""
    pts = []
    for c, m in zip(centers, counts):
        # Sample radius ~ sqrt(U)*R to be uniform in area; angle ~ U(0,2\pi)
        r = np.sqrt(rng.uniform(0.0, 1.0, size=m)) * radius
        ang = rng.uniform(0.0, 2.0 * np.pi, size=m)
        offset = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        pts.append(c + offset)
    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.vstack(pts)


def _choose_cluster_means_angles(rng: np.random.Generator, k: int) -> np.ndarray:
    """Pick k independent mean directions uniformly on [0, 2pi)."""
    return rng.uniform(0.0, 2.0 * np.pi, size=k)


def _sample_angles_vonmises(
    rng: np.random.Generator, mu: float, kappa: float, n: int
) -> np.ndarray:
    """Sample angles from von Mises(mu, kappa). kappa=0 => uniform."""
    # numpy Generator.vonmises is available and vectorized.
    return rng.vonmises(mu=mu, kappa=kappa, size=n)


def _ensure_uav_positions(
    params: GenParams, rng: np.random.Generator
) -> list[np.ndarray]:
    if params.uav_positions_mode == "fixed":
        if params.uav_fixed_positions is None:
            # Default: all UAVs at origin
            return [np.array([0.0, 0.0], dtype=float) for _ in range(params.n_uavs)]
        if len(params.uav_fixed_positions) < params.n_uavs:
            raise ValueError("uav_fixed_positions shorter than n_uavs")
        return [as_vec2(p) for p in params.uav_fixed_positions[: params.n_uavs]]
    elif params.uav_positions_mode == "random":
        pts = params.area.sample_uniform(rng, params.n_uavs)
        return [pts[i] for i in range(params.n_uavs)]
    else:
        raise ValueError("Unknown uav_positions_mode")


def _sample_uav_speeds(rng: np.random.Generator, params: GenParams) -> list[float]:
    if params.uav_speed_max < params.uav_speed_min:
        raise ValueError("uav_speed_max must be ≥ uav_speed_min")
    if math.isclose(params.uav_speed_min, params.uav_speed_max):
        return [float(params.uav_speed_min)] * params.n_uavs
    vals = rng.uniform(params.uav_speed_min, params.uav_speed_max, size=params.n_uavs)
    return [float(x) for x in vals]


def _mk_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i+1}" for i in range(n)]


def generate_scenario(
    case: CaseKind,
    params: GenParams = GenParams(),
    seed: Optional[int] = None,
    scenario_id: Optional[str] = None,
) -> Scenario:
    """
    Generate a single Scenario for the requested case.

    Parameters
    ----------
    case : CaseKind
        One of "random", "pos_clusters", "dir_clusters", "posdir_clusters".
    params : GenParams
        Parameter object for sizes, ranges, clustering radii, etc.
    seed : Optional[int]
        Reproducibility seed. If None, uses entropy.
    scenario_id : Optional[str]
        If None, an id is composed from params.scenario_prefix + timestamp.

    Returns
    -------
    Scenario
    """
    rng = _rng(seed)
    sid = (
        scenario_id
        or f"{params.scenario_prefix}_{case}_{int(time.time()*1000)%10_000_000}"
    )

    # Targets: positions and directions
    nT = int(params.n_targets)

    # Decide per-case sampling strategies
    if case == "random":
        # Positions uniform in the area; angles uniform; speeds uniform in [min,max]
        p0s = params.area.sample_uniform(rng, nT)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=nT)
        dirs = _unit_vec_from_angle(angles)

    else:
        # Clustered variants share the same wiring with different toggles
        k = int(params.n_clusters or params.n_uavs or 1)
        k = max(1, min(k, nT))
        counts = _split_counts(nT, k)

        # Position cluster centers and samples
        if case in ("pos_clusters", "posdir_clusters"):
            centers = params.area.sample_uniform(rng, k)
            p0s = _sample_in_discs(rng, centers, counts, params.cluster_pos_radius)
            # guard to keep inside area (optional)
            p0s = np.stack([params.area.clamp_point(p) for p in p0s], axis=0)
        else:
            # positions not clustered
            p0s = params.area.sample_uniform(rng, nT)

        # Direction clusters
        if case in ("dir_clusters", "posdir_clusters"):
            mean_angles = _choose_cluster_means_angles(rng, k)
            angles_list: list[np.ndarray] = []
            for mu, m in zip(mean_angles, counts):
                if params.dir_kappa <= 0:
                    # kappa <= 0 => uniform fallback
                    ang = rng.uniform(0.0, 2.0 * np.pi, size=m)
                else:
                    ang = _sample_angles_vonmises(
                        rng, mu=mu, kappa=params.dir_kappa, n=m
                    )
                angles_list.append(ang)
            angles = (
                np.concatenate(angles_list, axis=0) if angles_list else np.empty((0,))
            )
            dirs = _unit_vec_from_angle(angles)
        else:
            # directions uniform
            angles = rng.uniform(0.0, 2.0 * np.pi, size=nT)
            dirs = _unit_vec_from_angle(angles)

    # Target speeds and velocities
    tgt_speeds = _sample_speeds(rng, nT, params.tgt_speed_min, params.tgt_speed_max)
    vels = (dirs.T * tgt_speeds).T  # shape (nT,2)

    # UAVs
    uav_positions = _ensure_uav_positions(params, rng)
    uav_speeds = _sample_uav_speeds(rng, params)

    # Build objects
    t_ids = _mk_ids(params.tgt_prefix, nT)
    u_ids = _mk_ids(params.uav_prefix, params.n_uavs)

    targets = [
        Target(id=t_ids[i], p0=p0s[i].astype(float), v=vels[i].astype(float))
        for i in range(nT)
    ]
    uavs = [
        UAV(id=u_ids[i], p0=uav_positions[i].astype(float), speed=float(uav_speeds[i]))
        for i in range(params.n_uavs)
    ]

    # Metadata
    meta: dict[str, Any] = {
        "generator": "generators.generate_scenario",
        "case": case,
        "seed": None if seed is None else int(seed),
        "params": _params_to_metadata(params),
        "counts": {"n_targets": nT, "n_uavs": params.n_uavs},
        "clusters": {
            "n_clusters": (
                int(params.n_clusters or params.n_uavs or 1) if case != "random" else 0
            ),
            "cluster_pos_radius": (
                params.cluster_pos_radius if case != "random" else 0.0
            ),
            "dir_kappa": (
                params.dir_kappa if case in ("dir_clusters", "posdir_clusters") else 0.0
            ),
        },
        "timestamp_unix": int(time.time()),
    }

    return Scenario(id=sid, targets=targets, uavs=uavs, metadata=meta)


def generate_suite(
    cases: Iterable[CaseKind] = (
        "random",
        "pos_clusters",
        "dir_clusters",
        "posdir_clusters",
    ),
    n_per_case: int = 5,
    base_params: GenParams = GenParams(),
    seed: Optional[int] = None,
    id_prefix: Optional[str] = None,
) -> list[Scenario]:
    """
    Produce a list of scenarios covering multiple cases with deterministic reseeding.

    The i-th scenario for a case uses seed = (seed + hash(case) + i) if a seed is provided.
    """
    rng_master = _rng(seed)
    scenarios: list[Scenario] = []
    for case in cases:
        for i in range(n_per_case):
            # Derive a child seed to keep runs reproducible yet independent
            child_seed = (
                None if seed is None else int(rng_master.integers(0, 2**31 - 1))
            )
            sid = None
            if id_prefix:
                sid = f"{id_prefix}_{case}_{i+1}"
            sc = generate_scenario(
                case=case,
                params=base_params,
                seed=child_seed,
                scenario_id=sid,
            )
            scenarios.append(sc)
    return scenarios


def scenario_to_json_dict(sc: Scenario) -> dict[str, Any]:
    """
    Convert Scenario -> plain JSON-serializable dict (arrays become lists).
    """
    return {
        "id": sc.id,
        "targets": [
            {"id": t.id, "p0": _arr2list(t.p0), "v": _arr2list(t.v)} for t in sc.targets
        ],
        "uavs": [
            {"id": u.id, "p0": _arr2list(u.p0), "speed": float(u.speed)}
            for u in sc.uavs
        ],
        "metadata": _jsonify_metadata(sc.metadata),
    }


def scenarios_to_json_str(scenarios: list[Scenario], indent: Optional[int] = 2) -> str:
    """
    Convert list of Scenarios -> JSON string. Each scenario becomes an element in an array.
    """
    payload = [scenario_to_json_dict(sc) for sc in scenarios]
    return json.dumps(payload, indent=indent)


def save_scenarios_json(
    path: str, scenarios: list[Scenario], indent: Optional[int] = 2
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(scenarios_to_json_str(scenarios, indent=indent))


def scenario_from_json_dict(d: dict[str, Any]) -> Scenario:
    """
    Convert JSON dict -> Scenario. Validates presence of required fields.
    """
    sid = str(d["id"])
    targets_json = d.get("targets", [])
    uavs_json = d.get("uavs", [])
    metadata = d.get("metadata", {})

    targets = [
        Target(
            id=str(t["id"]),
            p0=np.array(t["p0"], dtype=float),
            v=np.array(t["v"], dtype=float),
        )
        for t in targets_json
    ]
    uavs = [
        UAV(id=str(u["id"]), p0=np.array(u["p0"], dtype=float), speed=float(u["speed"]))
        for u in uavs_json
    ]
    return Scenario(id=sid, targets=targets, uavs=uavs, metadata=metadata)


def scenarios_from_json_str(s: str) -> list[Scenario]:
    arr = json.loads(s)
    if not isinstance(arr, list):
        raise ValueError("Top-level JSON must be a list of scenarios")
    return [scenario_from_json_dict(d) for d in arr]


def load_scenarios_json(path: str) -> list[Scenario]:
    with open(path, "r", encoding="utf-8") as f:
        return scenarios_from_json_str(f.read())


def _arr2list(a: np.ndarray) -> list[float]:
    a = np.asarray(a, dtype=float)
    if a.shape != (2,):
        raise ValueError(f"Expected shape (2,), got {a.shape}")
    return [float(a[0]), float(a[1])]


def _jsonify_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """Ensure metadata is JSON-serializable (convert numpy scalars, etc.)."""

    def _clean(x: Any) -> Any:
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): _clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_clean(v) for v in x]
        return x

    return _clean(md)


def _params_to_metadata(p: GenParams) -> dict[str, Any]:
    # dataclass -> dict, ensure Box is flattened
    d = asdict(p)
    # Expand Box for readability
    if isinstance(p.area, Box):
        d["area"] = {
            "x_min": p.area.x_min,
            "x_max": p.area.x_max,
            "y_min": p.area.y_min,
            "y_max": p.area.y_max,
        }
    return d


if __name__ == "__main__":
    # Smoke tests to ensure shapes, serde, and reproducibility
    base = GenParams(
        n_targets=30,
        n_uavs=3,
        area=Box(-500, 500, -500, 500),
        tgt_speed_min=5,
        tgt_speed_max=15,
        uav_speed_min=25,
        uav_speed_max=35,
        uav_positions_mode="fixed",
        uav_fixed_positions=[(0.0, 0.0), (100.0, 0.0), (-100.0, 0.0)],
        n_clusters=None,  # defaults to n_uavs
        cluster_pos_radius=120.0,
        dir_kappa=6.0,
        scenario_prefix="demo",
    )

    seed = 42
    sc1 = generate_scenario("random", base, seed=seed, scenario_id="demo_random")
    sc2 = generate_scenario("pos_clusters", base, seed=seed, scenario_id="demo_pos")
    sc3 = generate_scenario("dir_clusters", base, seed=seed, scenario_id="demo_dir")
    sc4 = generate_scenario(
        "posdir_clusters", base, seed=seed, scenario_id="demo_posdir"
    )

    blob = scenarios_to_json_str([sc1, sc2, sc3, sc4], indent=2)
    rec = scenarios_from_json_str(blob)

    assert len(rec) == 4
    assert rec[0].id == "demo_random"
    assert len(rec[0].targets) == base.n_targets
    assert len(rec[0].uavs) == base.n_uavs

    print("generators.py smoke tests passed.")
