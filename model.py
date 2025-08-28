# Core geometry, kinematics, and closed-form interception for moving targets in 2D.

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


def as_vec2(x: tuple[float, float] | list[float] | np.ndarray) -> np.ndarray:
    """Convert input to a (2,) float64 numpy array (no copy if already suitable)."""
    a = np.asarray(x, dtype=float)
    if a.shape != (2,):
        raise ValueError(f"Expected shape (2,), got {a.shape}")
    return a


@dataclass(frozen=True)
class Entity2D:
    """Base class for 2D point entities with an identifier and initial position."""

    id: str
    p0: np.ndarray  # initial position at t = 0

    def __post_init__(self):
        object.__setattr__(self, "p0", as_vec2(self.p0))


@dataclass(frozen=True)
class Target(Entity2D):
    """Target moving linearly with constant velocity."""

    v: np.ndarray  # velocity (vx, vy)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "v", as_vec2(self.v))

    def position(self, t: float) -> np.ndarray:
        """Position of the target at time t."""
        return self.p0 + self.v * t


@dataclass(frozen=True)
class UAV(Entity2D):
    """UAV with a maximum speed. UAV motion is controlled by the scheduler."""

    speed: float  # max speed (scalar, > 0)

    def __post_init__(self):
        super().__post_init__()
        if self.speed <= 0:
            raise ValueError("UAV.speed must be positive")


@dataclass(frozen=True)
class Scenario:
    """A collection of targets and UAVs with light convenience methods."""

    id: str
    targets: list[Target] = field(default_factory=list)
    uavs: list[UAV] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def target_by_id(self, tid: str) -> Target:
        for t in self.targets:
            if t.id == tid:
                return t
        raise KeyError(f"Target id '{tid}' not found")

    def uav_by_id(self, uid: str) -> UAV:
        for u in self.uavs:
            if u.id == uid:
                return u
        raise KeyError(f"UAV id '{uid}' not found")


@dataclass(frozen=True)
class InterceptSolution:
    """Solution of the intercept equation for a UAV and a moving target."""

    t_hit: float  # absolute time of interception (>= t0)
    p_hit: np.ndarray  # position of interception
    leg_time: float  # travel time from (t0, u0) to hit (equals t_hit - t0)
    leg_distance: float  # distance traveled (equals speed * leg_time)


def _solve_quadratic_stable(
    A: float, B: float, C: float
) -> Optional[tuple[float, float]]:
    """
    Numerically stable quadratic solver for A*t^2 + B*t + C = 0.
    Returns roots as a tuple (r1, r2) without sorting, or None if discriminant < 0.
    """
    D = B * B - 4.0 * A * C
    if D < 0.0:
        return None
    sqrtD = np.sqrt(D)
    # When A == 0, caller should use linear path; here assume |A|>0.
    # Use the 'q' trick for stability.
    q = -0.5 * (B + np.sign(B) * sqrtD) if B != 0.0 else -0.5 * sqrtD
    # If q == 0, both roots are 0 or ill-conditioned; fall back to direct formula.
    if q != 0.0:
        r1 = q / A
        r2 = C / q
    else:
        r1 = (-B + sqrtD) / (2.0 * A)
        r2 = (-B - sqrtD) / (2.0 * A)
    return (r1, r2)


def intercept_time(
    u0: np.ndarray,
    t0: float,
    p0: np.ndarray,
    v: np.ndarray,
    speed: float,
    eps: float = 1e-9,
) -> Optional[InterceptSolution]:
    """
    Compute the earliest intercept between a UAV and a linearly moving target.

    We solve for the smallest t >= t0 such that
        || (p0 + v*t) - u0 || = speed * (t - t0)

    Parameters
    ----------
    u0 : np.ndarray
        UAV position at time t0.
    t0 : float
        Current time when UAV becomes free.
    p0 : np.ndarray
        Target's initial position at time 0.
    v : np.ndarray
        Target's constant velocity.
    speed : float
        UAV max speed (positive).
    eps : float
        Small tolerance for comparisons.

    Returns
    -------
    InterceptSolution or None
        The interception details, or None if unreachable (no real root >= t0).
    """
    if speed <= 0:
        raise ValueError("speed must be positive")

    u0 = as_vec2(u0)
    p0 = as_vec2(p0)
    v = as_vec2(v)
    t0 = float(t0)

    # Immediate hit if already co-located at t0.
    # Check whether the target is at u0 at t0: p0 + v*t0 == u0
    if np.linalg.norm((p0 + v * t0) - u0) < eps:
        p_hit = p0 + v * t0
        return InterceptSolution(t_hit=t0, p_hit=p_hit, leg_time=0.0, leg_distance=0.0)

    # Build quadratic A t^2 + B t + C = 0 following:
    # ||Delta + v t||^2 = s^2 (t - t0)^2
    # where Delta = p0 - u0
    Delta = p0 - u0
    v2 = np.dot(v, v)
    s2 = speed * speed
    A = v2 - s2
    B = 2.0 * (np.dot(Delta, v) + s2 * t0)
    C = np.dot(Delta, Delta) - s2 * t0 * t0

    # Handle A ~ 0: linear equation B t + C = 0
    if abs(A) <= 1e-14:
        if abs(B) <= 1e-14:
            # Degenerate: A ~ 0 and B ~ 0 -> either no solution or infinite solutions
            # If C ~ 0 as well, the equality holds for all t; choose earliest feasible t >= t0.
            if abs(C) <= 1e-14:
                # The equality holds; earliest is t0 (we already checked co-location).
                # If not co-located, the only way equality holds is pathological; return None.
                return None
            else:
                return None
        t = -C / B
        if t + eps < t0:
            return None
        # Compute hit
        t_hit = max(t, t0)  # numeric guard
        p_hit = p0 + v * t_hit
        leg_time = t_hit - t0
        if leg_time < 0:
            return None
        leg_distance = speed * leg_time
        return InterceptSolution(
            t_hit=t_hit, p_hit=p_hit, leg_time=leg_time, leg_distance=leg_distance
        )

    # Proper quadratic case
    roots = _solve_quadratic_stable(A, B, C)
    if roots is None:
        return None

    # Select the smallest root >= t0 (with tolerance)
    candidates = [r for r in roots if r + eps >= t0]
    if not candidates:
        return None
    t_hit = min(candidates)
    if t_hit < t0:
        t_hit = t0  # numeric guard

    p_hit = p0 + v * t_hit
    leg_time = t_hit - t0
    if leg_time < -eps:
        return None
    leg_time = max(0.0, leg_time)
    leg_distance = speed * leg_time
    return InterceptSolution(
        t_hit=t_hit, p_hit=p_hit, leg_time=leg_time, leg_distance=leg_distance
    )


def intercept_uav_target(
    uav: UAV, uav_pos_at_t0: np.ndarray, t0: float, target: Target, eps: float = 1e-9
) -> Optional[InterceptSolution]:
    """
    Wrapper that uses UAV's speed and a Target object.
    `uav_pos_at_t0` is the UAV's current position (e.g., last interception point).
    """
    return intercept_time(uav_pos_at_t0, t0, target.p0, target.v, uav.speed, eps=eps)


if __name__ == "__main__":
    # A few quick smoke tests
    rng = np.random.default_rng(0)

    # 1) Stationary target ahead, fast UAV
    u = UAV(id="u1", p0=np.array([0.0, 0.0]), speed=10.0)
    tgt = Target(id="t1", p0=np.array([100.0, 0.0]), v=np.array([0.0, 0.0]))
    sol = intercept_uav_target(u, u.p0, 0.0, tgt)
    assert sol is not None
    assert np.isclose(sol.t_hit, 10.0)
    assert np.isclose(sol.leg_distance, 100.0)

    # 2) Target moving towards UAV
    tgt2 = Target(id="t2", p0=np.array([100.0, 0.0]), v=np.array([-5.0, 0.0]))
    sol2 = intercept_uav_target(u, u.p0, 0.0, tgt2)
    assert sol2 is not None
    # Expected t satisfies |100 - 5 t| = 10 t  => 100 = 15 t  => t \approx 6.6667
    assert abs(sol2.t_hit - (100.0 / 15.0)) < 1e-9

    # 3) Target faster and moving away, unreachable
    u_slow = UAV(id="u2", p0=np.array([0.0, 0.0]), speed=5.0)
    tgt_fast = Target(id="t3", p0=np.array([100.0, 0.0]), v=np.array([10.0, 0.0]))
    sol3 = intercept_uav_target(u_slow, u_slow.p0, 0.0, tgt_fast)
    assert sol3 is None

    # 4) Immediate co-location at t0
    tgt4 = Target(id="t4", p0=np.array([0.0, 0.0]), v=np.array([1.0, 0.0]))
    sol4 = intercept_uav_target(u, u.p0, 0.0, tgt4)
    assert (
        sol4 is not None
        and np.isclose(sol4.t_hit, 0.0)
        and np.isclose(sol4.leg_distance, 0.0)
    )

    print("model.py smoke tests passed.")
