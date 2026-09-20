from __future__ import annotations

r"""Scale-covariant, dimensionally consistent operational information certificates.

Why this module exists
----------------------

Every V68 "geometry" certificate that is used as a *level* metric is built from

.. math::

   F(M) \;=\; \frac{\operatorname{rank}(M)}{n}\,
              \frac{\lambda_{\min}^{+}(M)}{\overline{\lambda^{+}}(M)},

implemented as
:func:`wideband_info_geometry.experiments_v58.v58_candidate_planner._finite_frame_lower_bound`.

``F`` is **homogeneous of degree zero**: for every :math:`c>0`,
:math:`F(cM)=F(M)` exactly.  It is a *conditioning* (isotropy) number.  It
carries no units and it cannot represent how much Fisher information a schedule
actually delivers.  Two consequences were measured on the shipped V68 source
(see ``docs/v69_ten_cycle_research_log.md``, cycles 1-3):

* ``sampling_stability_retention`` :math:`=F(\sum_i w_i J_i)/F(\sum_i J_i)` is
  identically ``1`` whenever the operational derating :math:`w_i` is common to
  all actions, *and* it saturates at ``1`` in 62.5% of a 216-point
  world/schedule sweep - including worlds in which **every** action has
  :math:`w_i=0`, i.e. the schedule returns no usable information at all.
* ``F`` is not monotone in the Loewner order.  Deleting actions lowers the rank
  of the retained matrix, which can *raise* :math:`\lambda_{\min}^{+}/
  \overline{\lambda^{+}}` towards 1, so
  ``single_action_erasure_stability`` can *increase* when information is
  destroyed.

This module replaces those level metrics with quantities that

1. are computed from the **Schur-complemented, nuisance-marginalized** position
   information - the object the project already declares to be the scientific
   estimand - rather than from a 3x3 proxy;
2. carry **SI units of metres**, so a "minimum effect of interest" can be
   stated operationally instead of in dimensionless certificate units;
3. are **degree** :math:`-1/2` **homogeneous** in the information, hence strictly
   responsive to uniform power/denial degradation;
4. are **Loewner monotone**, hence never reward the destruction of information.

Mathematical core
-----------------

Let :math:`\mathcal S` be a schedule and :math:`J_0\succ 0` the normalized
prior information.  Each registered ``ActionFactor`` already contains two
different Fisher matrices:

* :math:`J_i^{\mathrm{comp}}`, the complete-data information conditional on
  the measurement model, and
* :math:`J_i^{\mathrm{obs}}`, the observed-data information after detection
  and association uncertainty.

The observed matrix already contains the detection probability and the
direction-dependent association-mixture kernel.  Multiplying it by those
retentions a second time would count missing information twice.  V69 therefore
uses the matrices directly:

.. math::

   J_{\mathrm{comp}} \;=\; J_0+\sum_{i\in\mathcal S}J_i^{\mathrm{comp}},
   \qquad
   J_{\mathrm{obs}} \;=\; J_0+\sum_{i\in\mathcal S}J_i^{\mathrm{obs}},

where :math:`\operatorname{Schur}_3` marginalizes velocity **and** every sensor
nuisance coordinate (clock bias, frequency bias, effective receive-chain delay,
calibration path), exactly as
:func:`...physics_v58.information.position_task_information_and_covariance`
does.  The SI position covariance is
:math:`P(w)=\Sigma\,J_{\mathrm{eff}}(w)^{-1}\,\Sigma` with
:math:`\Sigma=\operatorname{diag}(\text{position scales})`, so
:math:`P(w)` has units :math:`\mathrm{m}^2`.

**Certificate 1 - operational degradation radius** (metres):

.. math::

   \rho_{\mathrm{miss}}(\mathcal S)
   \;=\; d_{\mathrm{BW}}\!\big(P_{\mathrm{obs}},P_{\mathrm{comp}}\big),
   \qquad
   d_{\mathrm{BW}}(A,B)^2=\operatorname{tr}A+\operatorname{tr}B
   -2\operatorname{tr}\big[(A^{1/2}BA^{1/2})^{1/2}\big].

:math:`d_{\mathrm{BW}}` is the Bures-Wasserstein metric on the SPD cone; it is
*exactly* the 2-Wasserstein distance between the centred Gaussian posteriors
:math:`\mathcal N(0,P_{\mathrm{obs}})` and
:math:`\mathcal N(0,P_{\mathrm{comp}})`.  It measures the covariance
displacement attributable to modeled missing observations.  It is not a
full-posterior distance because posterior means are not part of this local
Fisher calculation.

*Theorem V69.1 (uniform-denial response).*  With :math:`J_0=0` and
:math:`w_i\equiv w\in(0,1]`, :math:`P(w)=P(\mathbf 1)/w` and

.. math::

   \rho_{\mathrm{op}} \;=\; \big(w^{-1/2}-1\big)\,
                             \sqrt{\operatorname{tr}P(\mathbf 1)} \;>\;0
   \quad\text{for } w<1 .

The V68 certificate returns exactly :math:`0` on the same input.  Proved in
``docs/v69_derivations.md`` Sec. 2; regression-tested in
``tests_v40/test_v69_operational_information.py``.

**Certificate 2 - attainable localisation radii** (metres):

.. math::

   r_D=\det\!\big(P(w)\big)^{1/6},\quad
   r_A=\sqrt{\tfrac13\operatorname{tr}P(w)},\quad
   r_E=\sqrt{\lambda_{\max}\big(P(w)\big)} .

These are the D-, A- and E-optimal criteria of optimal experimental design
written as lengths.  :math:`r_D` is the geometric-mean semi-axis of the
posterior 1-sigma ellipsoid; :math:`r_E` is its longest semi-axis.

**Certificate 2b - operational information gain** (nats per axis).  Added in
research cycle 8 after the bounded ladder showed the metre-valued radii are
*right-censored at the prior*: once a schedule is fully denied, :math:`P(w)`
equals the prior covariance and no posterior statistic can distinguish degrees
of denial.  Measured on the untreated V69 grid, 30.6% of
``operational_d_radius_m`` values sit exactly at the 260 m prior ceiling, and
the A4 ladder Spearman coefficient fell to 0.354 for that reason alone.  The
uncensored estimand must live on the information side:

.. math::

   G\;=\;\frac{1}{2d}\Big[\log\det\big(J_{\mathrm{eff}}\big)
                           -\log\det\big(J_{\mathrm{prior}}\big)\Big],
   \qquad d=3 ,

with :math:`J_{\mathrm{prior}}` the Schur-marginalized prior.  :math:`G` is the
Gaussian information gain per position axis; :math:`e^{-G}` is the factor by
which the geometric-mean error radius shrinks, so a minimum effect of
:math:`0.1` nats is a 9.5% change in localisation accuracy.  :math:`G` is
Loewner nondecreasing and unbounded above,
and **not** censored: as a schedule is denied, :math:`G\to 0` continuously
rather than pinning at a ceiling.  The denial cost
:math:`\Delta G=G(\mathbf 1)-G(w)\ge 0` is the uncensored companion of
:math:`\rho_{\mathrm{op}}`.

**Certificate 3 - erasure fragility** (metres):

.. math::

   \varphi(\mathcal S)\;=\;\max_{i\in\mathcal S}
   \Big[r_D\big(\mathcal S\setminus\{i\}\big)-r_D(\mathcal S)\Big]\;\ge\;0 .

Non-negativity is a theorem, not a convention: removing an action removes a PSD
term, so :math:`J` decreases in the Loewner order, :math:`P` increases
(operator antitonicity of :math:`X\mapsto X^{-1}`), and :math:`\det` is Loewner
monotone, hence :math:`r_D` cannot decrease.  The V68 analogue can and does
decrease.

Units discipline
----------------

Every public function documents its units.  ``*_m`` is metres, ``*_m2`` is
square metres, ``*_ratio`` is dimensionless.  Nothing in this module returns an
unlabelled unit-interval score.

License / provenance of adopted mathematics
-------------------------------------------

Bures-Wasserstein geometry, Loewner order facts and alphabetic optimality are
classical results used as mathematics, not as code; no third-party source is
vendored.  See ``docs/v69_prior_art_and_novelty_matrix.md`` for the citation
ledger.
"""

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.information import (
    action_factor_cache_key,
    prior_information_normalized,
    schur_effective_information,
    stable_inverse_spd,
)
from wideband_info_geometry.physics_v58.measurement import (
    ActionFactor,
    RadarAction,
    StateLayout,
    build_action_factor,
)
from wideband_info_geometry.physics_v58.schema import PhysicalWorld

__all__ = [
    "OperationalInformationCertificate",
    "V69_CERTIFICATE_OUTCOMES",
    "V69_CERTIFICATE_UNITS",
    "a_optimal_radius_m",
    "bures_wasserstein_distance_m",
    "d_optimal_radius_m",
    "e_optimal_radius_m",
    "erasure_fragility_m",
    "information_gain_nats",
    "operational_information_certificate",
    "operational_weights_from_physics",
    "position_covariance_m2",
    "spd_sqrt",
    "uniform_denial_reference_m",
]

_EPS = 1e-300
#: Position-block dimension of the target state.
_POSITION_DIM = 3


# ---------------------------------------------------------------------------
# SPD primitives
# ---------------------------------------------------------------------------


def _sym(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    return 0.5 * (m + m.T)


def spd_sqrt(matrix: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    """Principal square root of a symmetric PSD matrix.

    Units: if ``matrix`` is in :math:`\\mathrm{m}^2` the result is in metres.
    Eigenvalues are clipped at ``floor`` (default ``0``) so a numerically
    indefinite input from round-off is projected onto the PSD cone rather than
    producing NaNs.
    """

    symmetric = _sym(matrix)
    if not np.all(np.isfinite(symmetric)):
        raise ValueError("SPD square root requires a finite matrix")
    values, vectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(np.min(values)) < -1e-10 * scale:
        raise ValueError(
            "SPD square root received a materially indefinite matrix: "
            f"lambda_min={float(np.min(values)):.6g}, scale={scale:.6g}"
        )
    values = np.maximum(values, float(floor))
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def bures_wasserstein_distance_m(covariance_a_m2: np.ndarray, covariance_b_m2: np.ndarray) -> float:
    r"""Bures-Wasserstein distance between two PSD covariances.

    ``d_BW(A,B)^2 = tr A + tr B - 2 tr[(A^{1/2} B A^{1/2})^{1/2}]``

    Equals the 2-Wasserstein distance between ``N(0, A)`` and ``N(0, B)``.

    Units: inputs :math:`\mathrm{m}^2`, output metres.

    Numerically the cross term is evaluated through the eigenvalues of
    :math:`A^{1/2}BA^{1/2}`, which is PSD and similar to :math:`AB`, so the
    trace of its square root equals the sum of the singular values of
    :math:`A^{1/2}B^{1/2}`.  Negative round-off in the radicand is clipped to
    zero, which can only shrink the reported distance (conservative).
    """

    a = _sym(covariance_a_m2)
    b = _sym(covariance_b_m2)
    if a.shape != b.shape:
        raise ValueError(f"covariance shape mismatch: {a.shape} vs {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    root_a = spd_sqrt(a)
    root_b = spd_sqrt(b)
    # Orthogonal Procrustes form of the Bures metric:
    #   d_BW(A,B) = min_Q ||A^(1/2) - B^(1/2) Q||_F.
    # Evaluating the aligned residual directly avoids catastrophic cancellation
    # in tr(A)+tr(B)-2*cross when A and B are equal or nearly equal.
    u, _singular_values, vt = np.linalg.svd(root_b.T @ root_a, full_matrices=False)
    alignment = u @ vt
    return float(np.linalg.norm(root_a - root_b @ alignment, ord="fro"))


def d_optimal_radius_m(covariance_m2: np.ndarray) -> float:
    r"""``det(P)^{1/(2d)}`` - geometric-mean semi-axis of the 1-sigma ellipsoid.

    Units: metres.  Degree :math:`-1/2` homogeneous in the information matrix.
    Returns ``inf`` for a singular (unobservable) covariance-inverse, which is
    the physically correct value: an unobserved direction has unbounded error.
    """

    covariance = _sym(covariance_m2)
    if not np.all(np.isfinite(covariance)):
        return float("inf")
    values = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(np.min(values)) < -1e-10 * scale:
        raise ValueError("D-optimal radius requires a positive-semidefinite covariance")
    values = np.maximum(values, 0.0)
    if np.any(values == 0.0):
        return 0.0
    return float(np.exp(0.5 * np.mean(np.log(values))))


def a_optimal_radius_m(covariance_m2: np.ndarray) -> float:
    """``sqrt(tr(P)/d)`` - RMS position error over the three axes.  Units: metres."""

    p = _sym(covariance_m2)
    if not np.all(np.isfinite(p)):
        return float("inf")
    return float(math.sqrt(max(float(np.trace(p)), 0.0) / max(p.shape[0], 1)))


def e_optimal_radius_m(covariance_m2: np.ndarray) -> float:
    """``sqrt(lambda_max(P))`` - worst-axis 1-sigma error.  Units: metres."""

    p = _sym(covariance_m2)
    if not np.all(np.isfinite(p)):
        return float("inf")
    return float(math.sqrt(max(float(np.max(np.linalg.eigvalsh(p))), 0.0)))


def information_gain_nats(
    posterior_covariance_m2: np.ndarray,
    prior_covariance_m2: np.ndarray,
) -> float:
    r"""Gaussian information gain per position axis, nats.

    ``G = (1/(2d)) [log det(P_prior) - log det(P_post)]``.  Equivalently
    ``G = (1/(2d)) [log det J_post - log det J_prior]``.  The factor ``1/2``
    is the Gaussian entropy factor.  ``exp(-G)`` is the
    factor by which the geometric-mean 1-sigma radius shrinks, so ``G`` is a
    dimensionless but *interpretable* accuracy measure that, unlike the radii,
    is not censored by the prior: a fully denied schedule gives ``G = 0``
    exactly and approaches it continuously.

    Returns ``0.0`` when the posterior is not better than the prior and
    ``inf`` when the posterior is singular in the prior's support.
    """

    post = _sym(posterior_covariance_m2)
    prior = _sym(prior_covariance_m2)
    if not (np.all(np.isfinite(post)) and np.all(np.isfinite(prior))):
        return 0.0
    dimension = max(post.shape[0], 1)
    prior_diagonal_scale = np.sqrt(np.maximum(np.diag(prior), 0.0))
    if np.any(prior_diagonal_scale <= 0.0):
        return 0.0
    scale_outer = np.outer(prior_diagonal_scale, prior_diagonal_scale)
    equilibrated_prior = _sym(prior / scale_outer)
    equilibrated_post = _sym(post / scale_outer)
    try:
        prior_cholesky = np.linalg.cholesky(equilibrated_prior)
    except np.linalg.LinAlgError:
        return 0.0
    try:
        post_cholesky = np.linalg.cholesky(equilibrated_post)
    except np.linalg.LinAlgError:
        return float("inf")
    # First equilibrate every coordinate by the prior marginal standard
    # deviation, then evaluate each log determinant from its Cholesky diagonal.
    # Positive diagonal coordinate changes (metres to kilometres, seconds to
    # nanoseconds, and so on) leave the ratio and its backward-error certificate
    # unchanged.  For A = L L^T,
    #
    #   (log det prior - log det post)/2
    #       = sum(log diag L_prior) - sum(log diag L_post).
    #
    # The tolerance is a backward-error bound, not a fixed scientific margin:
    # eigensolvers and triangular solves perturb an SPD matrix by O(kappa*eps).
    # Inputs too ill-conditioned to support a useful bound are rejected rather
    # than silently censored at zero.
    gain = float(
        np.sum(np.log(np.diag(prior_cholesky)))
        - np.sum(np.log(np.diag(post_cholesky)))
    ) / dimension
    equilibrated_prior_eigenvalues = np.linalg.eigvalsh(equilibrated_prior)
    equilibrated_post_eigenvalues = np.linalg.eigvalsh(equilibrated_post)
    condition_number = max(
        float(
            np.max(equilibrated_prior_eigenvalues)
            / np.min(equilibrated_prior_eigenvalues)
        ),
        float(
            np.max(equilibrated_post_eigenvalues)
            / np.min(equilibrated_post_eigenvalues)
        ),
    )
    relative_roundoff = np.finfo(float).eps * condition_number
    # Reject only when the backward-error estimate can move a spectral mode by
    # at least 0.1%.  The former 1e-7 cutoff rejected valid, highly informative
    # radar geometries even though their Cholesky log determinant retained many
    # useful digits.
    if not math.isfinite(relative_roundoff) or relative_roundoff > 1.0e-3:
        raise ValueError(
            "information gain is not numerically certifiable for the declared "
            f"covariances; condition*eps={relative_roundoff:.6g}"
        )
    # A determinant ratio assembled from two independently computed covariance
    # matrices has a coordinate-invariant numerical null band.  ``sqrt(eps)``
    # is the standard boundary between resolvable first-order change and
    # cancellation-dominated change; it is a numerical convention, not a
    # scientific effect-size threshold.  Material degradation remains an error.
    numerical_tolerance = max(
        float(math.sqrt(np.finfo(float).eps)),
        8.0 * relative_roundoff,
    )
    if gain < -numerical_tolerance:
        raise ValueError(
            "posterior covariance is materially worse than the declared prior; "
            f"computed information gain {gain:.6g} nat/axis exceeds numerical "
            f"tolerance {numerical_tolerance:.6g}"
        )
    return float(max(gain, 0.0))


def uniform_denial_reference_m(nominal_covariance_m2: np.ndarray, weight: float) -> float:
    r"""Closed form of :math:`\rho_{\mathrm{op}}` under uniform denial.

    Theorem V69.1: with no prior and :math:`w_i\equiv w>0`,
    :math:`\rho_{\mathrm{op}}=(w^{-1/2}-1)\sqrt{\operatorname{tr}P_{\mathrm{nom}}}`.
    Exposed so the closed form can be regression-tested against the numerical
    Bures-Wasserstein evaluation.  Units: metres.
    """

    w = float(weight)
    if w <= 0.0:
        return float("inf")
    trace = max(float(np.trace(_sym(nominal_covariance_m2))), 0.0)
    return float(abs(w**-0.5 - 1.0) * math.sqrt(trace))


# ---------------------------------------------------------------------------
# Pipeline-integrated evaluation
# ---------------------------------------------------------------------------


def operational_weights_from_physics(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    *,
    factor_cache: dict | None = None,
) -> tuple[dict[int, float], dict[int, ActionFactor]]:
    r"""Per-action observed/complete information retention diagnostics.

    ``ActionFactor.observed_information_normalized`` already includes detection
    probability and the association-mixture kernel.  The returned retention is

    ``trace(J_observed) / trace(J_complete)``,

    computed on the registered normalized state.  It is diagnostic only; the
    certificate never multiplies ``J_observed`` by this number again.
    """

    weights: dict[int, float] = {}
    factors: dict[int, ActionFactor] = {}
    for action in actions:
        key = action_factor_cache_key(world, action.action_id, "nominal")
        if factor_cache is not None and key in factor_cache:
            factor = factor_cache[key]
        else:
            factor = build_action_factor(world, action, None)
            if factor_cache is not None:
                factor_cache[key] = factor
        complete_trace = float(np.trace(factor.complete_information_normalized))
        observed_trace = float(np.trace(factor.observed_information_normalized))
        weight = observed_trace / max(complete_trace, 1e-300)
        weights[int(action.action_id)] = float(np.clip(weight, 0.0, 1.0))
        factors[int(action.action_id)] = factor
    return weights, factors


def position_covariance_m2(
    world: PhysicalWorld,
    factors: Mapping[int, ActionFactor],
    weights: Mapping[int, float],
    *,
    include_prior: bool = True,
    layout: StateLayout | None = None,
    information_kind: str = "observed",
) -> np.ndarray:
    r"""SI position covariance after nuisance marginalization.  Units: m^2.

    ``J(w) = [J_0] + sum_i w_i J_i`` on the full normalized state, then
    :math:`\operatorname{Schur}_3` down to the position block, then the
    normalized-to-SI scale similarity.  ``include_prior=False`` removes
    :math:`J_0`, which is what makes the exact degree :math:`-1/2` homogeneity
    theorem testable (with a prior the map is only asymptotically homogeneous).
    """

    layout = layout or StateLayout.from_world(world)
    dimension = layout.dimension
    if include_prior:
        joint = np.asarray(prior_information_normalized(world, layout), dtype=float).copy()
    else:
        joint = np.zeros((dimension, dimension), dtype=float)
    matrix_attribute = {
        "observed": "observed_information_normalized",
        "complete": "complete_information_normalized",
    }.get(str(information_kind))
    if matrix_attribute is None:
        raise ValueError(
            "information_kind must be 'observed' or 'complete', "
            f"received {information_kind!r}"
        )
    for action_id, factor in factors.items():
        w = float(weights.get(int(action_id), 1.0))
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"invalid information multiplier for action {action_id}: {w}")
        if w == 0.0:
            continue
        joint = joint + w * np.asarray(getattr(factor, matrix_attribute), dtype=float)
    joint = _sym(joint)
    effective, _ = schur_effective_information(joint, _POSITION_DIM)
    scales = layout.scales()[:_POSITION_DIM]
    eigenvalues = np.linalg.eigvalsh(_sym(effective))
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) <= 1e-12 * scale:
        if include_prior:
            raise ValueError(
                "prior-regularized effective position information is not positive definite"
            )
        return np.full((_POSITION_DIM, _POSITION_DIM), np.inf)
    covariance_normalized = stable_inverse_spd(
        effective, name="v69_effective_position_information"
    )
    return np.diag(scales) @ _sym(covariance_normalized) @ np.diag(scales)


def erasure_fragility_m(
    world: PhysicalWorld,
    factors: Mapping[int, ActionFactor],
    weights: Mapping[int, float],
    *,
    include_prior: bool = True,
    layout: StateLayout | None = None,
) -> tuple[float, int]:
    r"""Worst single-action erasure cost in metres, and the critical action id.

    :math:`\varphi=\max_i [r_D(\mathcal S\setminus\{i\})-r_D(\mathcal S)]\ge 0`.
    Non-negativity is guaranteed by Loewner monotonicity of
    :math:`\det` composed with operator antitonicity of the inverse; it is
    asserted at runtime with a small numerical tolerance so a violation is a
    loud failure rather than a silent one.
    """

    layout = layout or StateLayout.from_world(world)
    base = d_optimal_radius_m(
        position_covariance_m2(world, factors, weights, include_prior=include_prior, layout=layout)
    )
    worst = 0.0
    worst_id = -1
    for action_id in sorted(factors):
        reduced = {k: v for k, v in factors.items() if k != action_id}
        radius = d_optimal_radius_m(
            position_covariance_m2(world, reduced, weights, include_prior=include_prior, layout=layout)
        )
        delta = radius - base
        if delta > worst:
            worst = float(delta)
            worst_id = int(action_id)
    if worst < -1e-6 * max(base, 1.0):  # pragma: no cover - theorem violation
        raise AssertionError(
            "Loewner monotonicity violated: erasure decreased the D-optimal radius "
            f"by {worst:.6g} m (base {base:.6g} m)"
        )
    return float(max(worst, 0.0)), worst_id


@dataclass(frozen=True)
class OperationalInformationCertificate:
    """Unit-carrying operational certificate for one (world, schedule).

    Field names are suffixed with their units.  ``*_ratio`` fields are
    dimensionless and are reported only alongside the metre-valued quantity
    they are derived from, never as a standalone level metric.
    """

    action_ids: tuple[int, ...]
    #: Nominal (w=1) localisation radii, metres.
    nominal_d_radius_m: float
    nominal_a_radius_m: float
    nominal_e_radius_m: float
    #: Operational (w from the link budget) localisation radii, metres.
    operational_d_radius_m: float
    operational_a_radius_m: float
    operational_e_radius_m: float
    #: Certificate 1: Bures-Wasserstein displacement of the posterior, metres.
    operational_degradation_radius_m: float
    #: Certificate 2b: average per-axis log-precision gain, nats.  Uncensored.
    nominal_information_gain_nats: float
    operational_information_gain_nats: float
    #: Denial cost in nats: nominal minus operational.  Uncensored, >= 0.
    information_gain_deficit_nats: float
    #: Certificate 3: worst single-action erasure cost, metres.
    erasure_fragility_m: float
    critical_action_id: int
    #: Dimensionless companions (reported, never used alone as a level metric).
    operational_degradation_ratio: float
    minimum_operational_weight: float
    mean_operational_weight: float
    dead_action_fraction: float
    #: Anisotropy of the operational posterior (dimensionless, diagnostic only).
    operational_condition_ratio: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["action_ids"] = list(self.action_ids)
        return payload


#: Names exported into the intervention-engine outcome namespace.  Every entry
#: is metre-valued except the three explicitly dimensionless diagnostics.
V69_CERTIFICATE_OUTCOMES: tuple[str, ...] = (
    "nominal_d_radius_m",
    "nominal_a_radius_m",
    "nominal_e_radius_m",
    "operational_d_radius_m",
    "operational_a_radius_m",
    "operational_e_radius_m",
    "operational_degradation_radius_m",
    "nominal_information_gain_nats",
    "operational_information_gain_nats",
    "information_gain_deficit_nats",
    "erasure_fragility_m",
    "operational_degradation_ratio",
    "minimum_operational_weight",
    "mean_operational_weight",
    "dead_action_fraction",
    "operational_condition_ratio",
)

V69_CERTIFICATE_UNITS: Mapping[str, str] = {
    "nominal_d_radius_m": "m",
    "nominal_a_radius_m": "m",
    "nominal_e_radius_m": "m",
    "operational_d_radius_m": "m",
    "operational_a_radius_m": "m",
    "operational_e_radius_m": "m",
    "operational_degradation_radius_m": "m",
    "nominal_information_gain_nats": "nat",
    "operational_information_gain_nats": "nat",
    "information_gain_deficit_nats": "nat",
    "erasure_fragility_m": "m",
    "operational_degradation_ratio": "1",
    "minimum_operational_weight": "1",
    "mean_operational_weight": "1",
    "dead_action_fraction": "1",
    "operational_condition_ratio": "1",
}


def operational_information_certificate(
    world: PhysicalWorld,
    action_ids: Iterable[int],
    *,
    include_prior: bool = True,
    factor_cache: dict | None = None,
    compute_fragility: bool = True,
) -> OperationalInformationCertificate:
    """Evaluate the full V69 certificate for one schedule.

    This is the function the normal pipeline calls: it takes the *registered
    physical world* and a schedule, rebuilds the registered ``ActionFactor``
    objects (shared cache with the rest of the pipeline), derives the
    operational derating from their metadata, and returns metre-valued
    certificates computed through the same Schur marginalization the exact
    objective uses.
    """

    ids = tuple(sorted(dict.fromkeys(int(v) for v in action_ids)))
    by_id = {int(a.action_id): a for a in world.actions}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"unknown action identifiers: {missing}")
    actions = [by_id[i] for i in ids]
    layout = StateLayout.from_world(world)
    retentions, factors = operational_weights_from_physics(
        world, actions, factor_cache=factor_cache
    )
    unit = {i: 1.0 for i in ids}

    p_nom = position_covariance_m2(
        world,
        factors,
        unit,
        include_prior=include_prior,
        layout=layout,
        information_kind="complete",
    )
    p_op = position_covariance_m2(
        world,
        factors,
        unit,
        include_prior=include_prior,
        layout=layout,
        information_kind="observed",
    )
    p_prior = position_covariance_m2(world, {}, {}, include_prior=True, layout=layout)

    nominal_gain = information_gain_nats(p_nom, p_prior)
    operational_gain = information_gain_nats(p_op, p_prior)

    nominal_d = d_optimal_radius_m(p_nom)
    operational_d = d_optimal_radius_m(p_op)
    rho = bures_wasserstein_distance_m(p_op, p_nom)

    if compute_fragility:
        fragility, critical = erasure_fragility_m(
            world, factors, unit, include_prior=include_prior, layout=layout
        )
    else:
        fragility, critical = float("nan"), -1

    weight_values = np.asarray([retentions[i] for i in ids], dtype=float)
    eig = np.linalg.eigvalsh(_sym(p_op))
    condition = float(np.max(eig) / max(float(np.min(eig)), _EPS)) if np.all(np.isfinite(eig)) else float("inf")

    return OperationalInformationCertificate(
        action_ids=ids,
        nominal_d_radius_m=nominal_d,
        nominal_a_radius_m=a_optimal_radius_m(p_nom),
        nominal_e_radius_m=e_optimal_radius_m(p_nom),
        operational_d_radius_m=operational_d,
        operational_a_radius_m=a_optimal_radius_m(p_op),
        operational_e_radius_m=e_optimal_radius_m(p_op),
        operational_degradation_radius_m=rho,
        nominal_information_gain_nats=nominal_gain,
        operational_information_gain_nats=operational_gain,
        information_gain_deficit_nats=float(max(nominal_gain - operational_gain, 0.0)),
        erasure_fragility_m=fragility,
        critical_action_id=critical,
        operational_degradation_ratio=float(operational_d / max(nominal_d, _EPS)),
        minimum_operational_weight=float(np.min(weight_values)) if weight_values.size else 1.0,
        mean_operational_weight=float(np.mean(weight_values)) if weight_values.size else 1.0,
        dead_action_fraction=float(np.mean(weight_values <= 1e-12)) if weight_values.size else 0.0,
        operational_condition_ratio=condition,
    )
