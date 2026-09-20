from __future__ import annotations

"""Authoritative dynamic factor-graph belief for the unified runtime."""

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from wideband_info_geometry.small_symmetric_linalg import (
    platform_solve,
    platform_symmetric_eigvalsh,
    symmetric_pinv,
)

from wideband_info_geometry.physics_v58.dynamic_factor_graph import (
    CommunicationSnapshot,
    DynamicFactor,
    DynamicFactorGraph,
    DynamicVariable,
    FixedLagMarginalization,
    FactorKind,
    GraphSolution,
    RobustKernel,
    VariableKey,
    VariableKind,
)
from wideband_info_geometry.physics_v58.effective_delay import (
    automatic_free_calibration_actions,
)
from wideband_info_geometry.physics_v58.information import (
    normalized_process_covariance,
    normalized_state_transition,
    prior_information_normalized,
)
from wideband_info_geometry.physics_v58.measurement import (
    ActionFactor,
    StateLayout,
    build_action_factor,
)
from wideband_info_geometry.physics_v58.schema import Jammer, PhysicalWorld, RadarAction

from .contracts import BeliefRecord, FactorInformationMode, RuntimeConfig


def _sym(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    return 0.5 * (value + value.T)


def _spectral_threshold(eigenvalues: np.ndarray, relative: float) -> float:
    """Return a scale-relative eigentolerance without a unit-sized floor.

    Normalized radar information may contain physically valid modes far below
    one while a separate gauge mode is exactly zero.  Using
    ``max(max_abs_eigenvalue, 1) * rcond`` erases those modes solely because of
    their numerical units.  The threshold must scale with the matrix itself.
    """

    values = np.asarray(eigenvalues, dtype=float)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return max(scale * float(relative), np.finfo(float).tiny)


def _validated_symmetric_eigendecomposition(
    matrix: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a validated eigensystem for a finite symmetric matrix.

    NumPy 2.3's divide-and-conquer path can stop converging after repeated
    decompositions of growing, highly conditioned fixed-lag covariances.  A
    positive scalar normalization preserves inertia and eigenspaces.  SciPy's
    MRRR driver is used first and the classical QR driver is an independent
    fallback; either result must satisfy residual and orthogonality bounds.
    """

    symmetric = _sym(matrix)
    if not np.all(np.isfinite(symmetric)):
        raise np.linalg.LinAlgError(f"{name} is non-finite")
    normalization = (
        float(np.max(np.abs(symmetric))) if symmetric.size else 0.0
    )
    if normalization == 0.0:
        return np.zeros(symmetric.shape[0]), np.eye(symmetric.shape[0])
    normalized = symmetric / normalization
    failures: list[str] = []
    for driver in ("evr", "ev"):
        try:
            eigenvalues, eigenvectors = scipy_eigh(
                normalized,
                driver=driver,
                check_finite=False,
                overwrite_a=False,
            )
            residual = float(
                np.linalg.norm(
                    normalized @ eigenvectors
                    - eigenvectors * eigenvalues,
                    ord="fro",
                )
                / max(1.0, float(np.linalg.norm(normalized, ord="fro")))
            )
            orthogonality = float(
                np.linalg.norm(
                    eigenvectors.T @ eigenvectors
                    - np.eye(eigenvectors.shape[1]),
                    ord="fro",
                )
            )
            tolerance = max(
                1.0e-10,
                1024.0
                * max(symmetric.shape[0], 1)
                * np.finfo(float).eps,
            )
            if (
                not np.all(np.isfinite(eigenvalues))
                or not np.all(np.isfinite(eigenvectors))
                or residual > tolerance
                or orthogonality > tolerance
            ):
                raise np.linalg.LinAlgError(
                    "unacceptable eigensystem "
                    f"residual={residual:.3e}, "
                    f"orthogonality={orthogonality:.3e}"
                )
            return eigenvalues * normalization, eigenvectors
        except np.linalg.LinAlgError as exc:
            failures.append(f"{driver}:{exc}")
    raise np.linalg.LinAlgError(
        f"{name} symmetric eigensolvers failed: {'; '.join(failures)}"
    )


def _psd_part(
    matrix: np.ndarray,
    floor: float = 0.0,
    *,
    relative_tolerance: float = 1e-10,
) -> np.ndarray:
    symmetric = _sym(matrix)
    if floor <= 0.0:
        try:
            np.linalg.cholesky(symmetric)
            return symmetric
        except np.linalg.LinAlgError:
            pass
    eigenvalues, eigenvectors = _validated_symmetric_eigendecomposition(
        symmetric,
        name="PSD projection input",
    )
    tolerance = _spectral_threshold(eigenvalues, relative_tolerance)
    if float(eigenvalues.min()) < -tolerance:
        raise np.linalg.LinAlgError(
            "materially indefinite covariance encountered; PSD projection is "
            "permitted only for roundoff-scale negative eigenvalues"
        )
    return _sym((eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T)


def _stable_inverse(matrix: np.ndarray, *, rcond: float = 1e-12) -> np.ndarray:
    """Invert SPD information without truncating small identifiable modes."""

    symmetric = _sym(matrix)
    identity = np.eye(symmetric.shape[0], dtype=float)
    try:
        cholesky = np.linalg.cholesky(symmetric)
        inverse = platform_solve(
            cholesky.T,
            platform_solve(cholesky, identity),
        )
        if np.all(np.isfinite(inverse)):
            return _sym(inverse)
    except np.linalg.LinAlgError:
        pass
    eigenvalues, eigenvectors = _validated_symmetric_eigendecomposition(
        symmetric,
        name="inverse input",
    )
    threshold = _spectral_threshold(eigenvalues, rcond)
    if float(eigenvalues.min()) < -threshold:
        raise np.linalg.LinAlgError(
            "matrix is materially indefinite; refusing a positive-mode-only "
            "pseudoinverse"
        )
    retained = eigenvalues > threshold
    if not np.any(retained):
        return np.zeros_like(symmetric)
    inverse = (
        (eigenvectors[:, retained] / eigenvalues[retained])
        @ eigenvectors[:, retained].T
    )
    return _sym(inverse)


def _whitened_jacobian(covariance: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Return ``W`` such that ``W.T @ W = H.T @ R^-1 @ H``.

    Keeping the measurement contribution in square-root form avoids explicitly
    multiplying by a possibly ill-conditioned covariance inverse.
    """

    symmetric = _sym(covariance)
    matrix = np.asarray(jacobian, dtype=float)
    try:
        cholesky = np.linalg.cholesky(symmetric)
        return platform_solve(cholesky, matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = _validated_symmetric_eigendecomposition(
            symmetric,
            name="factor covariance",
        )
        threshold = _spectral_threshold(eigenvalues, 1e-12)
        if float(eigenvalues.min()) < -threshold:
            raise np.linalg.LinAlgError(
                "cannot whiten a materially indefinite factor covariance"
            )
        retained = eigenvalues > threshold
        if not np.any(retained):
            return np.zeros((0, matrix.shape[1]), dtype=float)
        whitener = (
            eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
        ).T
        return whitener @ matrix


def _solve_measurement_covariance(
    covariance: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Solve ``R x = b`` after unit-equivariant diagonal equilibration.

    Radar measurement vectors mix quantities such as seconds, hertz and
    radians.  A physically well-conditioned covariance can therefore have raw
    diagonal entries separated by many orders of magnitude.  Testing pivots
    against the largest *unscaled* entry incorrectly labels that unit choice as
    singular.  Writing ``R = D C D``, where ``D = sqrt(diag(R))`` and ``C`` is
    the correlation matrix, gives

        x = D^{-1} C^{-1} D^{-1} b.

    The solve is then invariant to independent positive rescalings of each
    measurement coordinate and retains every strictly positive information
    mode.  No ridge or artificial measurement noise is introduced.
    """

    symmetric = _sym(covariance)
    value = np.asarray(right, dtype=float)
    if symmetric.ndim != 2 or symmetric.shape[0] != symmetric.shape[1]:
        raise ValueError("measurement covariance must be square")
    if value.shape[0] != symmetric.shape[0]:
        raise ValueError("measurement covariance right-hand side mismatch")
    scale = np.sqrt(np.diag(symmetric))
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise np.linalg.LinAlgError(
            "measurement covariance has a nonpositive diagonal"
        )
    correlation = symmetric / scale[:, None] / scale[None, :]
    scaled_right = (
        value / scale
        if value.ndim == 1
        else value / scale[:, None]
    )
    scaled_solution = platform_solve(correlation, scaled_right)
    return (
        scaled_solution / scale
        if scaled_solution.ndim == 1
        else scaled_solution / scale[:, None]
    )


def _covariance_support_factor(covariance: np.ndarray) -> np.ndarray:
    """Factor a numerically PSD covariance without inventing information.

    The eigensolver's backward error is proportional to matrix order and
    spectral scale.  Negative modes inside that bound are representation
    roundoff of a covariance already constructed from an SPD information
    factor; larger negative modes remain fail-closed.
    """

    symmetric = _sym(covariance)
    if not np.all(np.isfinite(symmetric)):
        raise np.linalg.LinAlgError("covariance support factor is non-finite")
    denominator = max(
        float(np.linalg.norm(symmetric, ord="fro")),
        np.finfo(float).tiny,
    )
    try:
        lower = np.linalg.cholesky(symmetric)
        residual = float(
            np.linalg.norm(lower @ lower.T - symmetric, ord="fro")
            / denominator
        )
        if np.isfinite(residual) and residual <= 1.0e-10:
            return lower
    except np.linalg.LinAlgError:
        pass
    eigenvalues, eigenvectors = _validated_symmetric_eigendecomposition(
        symmetric,
        name="covariance support factor",
    )
    scale = max(
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0,
        np.finfo(float).tiny,
    )
    backward_error = (
        64.0
        * max(symmetric.shape[0], 1)
        * np.finfo(float).eps
        * scale
    )
    if float(eigenvalues.min()) < -backward_error:
        raise np.linalg.LinAlgError(
            "covariance has a negative mode outside the eigensolver backward-error bound"
        )
    retained = eigenvalues > 0.0
    if not np.any(retained):
        return np.zeros((symmetric.shape[0], 0), dtype=float)
    return eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])


def _canonical_rows(
    information: np.ndarray,
    information_vector: np.ndarray,
    *,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return H,y with H.T H=J and H.T y=eta on the support of J."""

    matrix = _sym(information)
    eta = np.asarray(information_vector, dtype=float).reshape(-1)
    if matrix.shape != (eta.size, eta.size):
        raise ValueError("canonical information and vector dimensions disagree")
    try:
        lower = np.linalg.cholesky(matrix)
        rows = lower.T
        measurement = platform_solve(lower, eta)
        eigenvalues = platform_symmetric_eigvalsh(matrix)
        return rows, measurement, {
            "rank": float(matrix.shape[0]),
            "minimum_retained_eigenvalue": float(eigenvalues.min()),
            "information_vector_projection_residual": 0.0,
        }
    except np.linalg.LinAlgError:
        pass
    eigenvalues, eigenvectors = _validated_symmetric_eigendecomposition(
        matrix,
        name="canonical information",
    )
    threshold = _spectral_threshold(eigenvalues, float(eigenvalue_floor))
    keep = eigenvalues > threshold
    if not np.any(keep):
        return np.zeros((0, eta.size)), np.zeros(0), {
            "rank": 0.0,
            "minimum_retained_eigenvalue": 0.0,
            "information_vector_projection_residual": float(np.linalg.norm(eta)),
        }
    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    rows = np.sqrt(values)[:, None] * vectors.T
    projected_eta = vectors @ (vectors.T @ eta)
    measurement = (vectors.T @ projected_eta) / np.sqrt(values)
    residual = float(
        np.linalg.norm(eta - projected_eta)
        / max(float(np.linalg.norm(eta)), np.finfo(float).tiny)
    )
    return rows, measurement, {
        "rank": float(values.size),
        "minimum_retained_eigenvalue": float(values.min()),
        "information_vector_projection_residual": residual,
    }


def _split_rows(rows: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return rows[:, :target_dim], rows[:, target_dim:]


def make_scoped_information_factor(
    *,
    factor_id: str,
    variables: Sequence[DynamicVariable],
    information: np.ndarray,
    information_vector: np.ndarray,
    epoch: int,
    arrival_epoch: int,
    provenance_ids: Sequence[str],
    fidelity_level: str,
    metadata: Mapping[str, Any] | None = None,
    eigenvalue_floor: float = 1e-12,
) -> DynamicFactor | None:
    """Canonical factor over an ordered, arbitrary-dimensional boundary.

    This constructs a candidate factor only. A graph-issued marginalization
    must still validate/commit it atomically; construction conveys no authority.
    The variable units define the information coordinates; no unit conversion
    or unregistered coordinate change is performed here.
    """
    variables = tuple(variables)
    keys = tuple(variable.key for variable in variables)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("scoped information requires unique ordered variables")
    if (type(epoch) is not int or type(arrival_epoch) is not int
            or epoch < 0 or arrival_epoch < epoch):
        raise ValueError("scoped factor epoch/arrival contract is invalid")
    dimension = sum(variable.dimension for variable in variables)
    matrix = np.asarray(information, dtype=float)
    eta = np.asarray(information_vector, dtype=float).reshape(-1)
    if matrix.shape != (dimension, dimension) or eta.shape != (dimension,):
        raise ValueError("scoped information dimension disagrees with boundary")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(eta)):
        raise ValueError("scoped information must be finite")
    matrix = _psd_part(matrix)
    rows, measurement, diagnostics = _canonical_rows(matrix, eta,
        eigenvalue_floor=eigenvalue_floor)
    if rows.shape[0] == 0:
        if np.any(matrix != 0.0) or np.any(eta != 0.0):
            raise ValueError("nonzero information cannot become an empty separator")
        return None
    jacobians = []
    offset = 0
    for variable in variables:
        jacobians.append(rows[:, offset:offset+variable.dimension])
        offset += variable.dimension
    return DynamicFactor(
        factor_id=factor_id, kind=FactorKind.PRIOR, observation_epoch=epoch,
        arrival_epoch=arrival_epoch, variables=keys, jacobians=tuple(jacobians),
        measurement=measurement, covariance=np.eye(rows.shape[0]), participants=(),
        provenance_ids=tuple(provenance_ids), correlation_group=None,
        robust_kernel=RobustKernel.NONE, association_probabilities=(1.0,),
        missing=False, fidelity_level=fidelity_level,
        metadata={**dict(metadata or {}), **diagnostics,
                  "canonical_information_factor": True,
                  "boundary_units": tuple(tuple(variable.units) for variable in variables),
                  "association_or_detection_weight_applied_again": False})


def make_joint_information_factor(
    *,
    factor_id: str,
    kind: FactorKind,
    epoch: int,
    target_key: VariableKey,
    nuisance_key: VariableKey,
    information: np.ndarray,
    information_vector: np.ndarray | None = None,
    participants: Sequence[str] = (),
    provenance_ids: Sequence[str],
    fidelity_level: str,
    metadata: Mapping[str, Any] | None = None,
    robust_kernel: RobustKernel = RobustKernel.NONE,
    eigenvalue_floor: float = 1e-12,
) -> DynamicFactor | None:
    matrix = _psd_part(information)
    eta = (
        np.zeros(matrix.shape[0], dtype=float)
        if information_vector is None
        else np.asarray(information_vector, dtype=float).reshape(-1)
    )
    rows, measurement, diagnostics = _canonical_rows(
        matrix,
        eta,
        eigenvalue_floor=eigenvalue_floor,
    )
    if rows.shape[0] == 0:
        return None
    target_rows, nuisance_rows = _split_rows(rows, 6)
    return DynamicFactor(
        factor_id=str(factor_id),
        kind=kind,
        observation_epoch=int(epoch),
        arrival_epoch=int(epoch),
        variables=(target_key, nuisance_key),
        jacobians=(target_rows, nuisance_rows),
        measurement=measurement,
        covariance=np.eye(rows.shape[0], dtype=float),
        participants=tuple(sorted(set(str(value) for value in participants))),
        provenance_ids=tuple(str(value) for value in provenance_ids),
        correlation_group=None,
        robust_kernel=robust_kernel,
        robust_scale=2.5,
        association_probabilities=(1.0,),
        missing=False,
        fidelity_level=str(fidelity_level),
        metadata={
            **dict(metadata or {}),
            **diagnostics,
            "canonical_information_factor": True,
            "association_or_detection_weight_applied_again": False,
        },
    )


def _automatic_calibration_information(
    world: PhysicalWorld,
    action: RadarAction,
) -> np.ndarray:
    layout = StateLayout.from_world(world)
    total = np.zeros((layout.dimension, layout.dimension), dtype=float)
    for calibration in automatic_free_calibration_actions(world, action):
        total += build_action_factor(world, calibration, None).observed_information_normalized
    return _sym(total)


def selected_action_information(
    world: PhysicalWorld,
    action: RadarAction,
    action_factor: ActionFactor,
    mode: FactorInformationMode,
) -> np.ndarray:
    r"""Return the information a *selected* action contributes in ``mode``.

    Let :math:`H_w=R^{-1/2}H` be the whitened normalized Jacobian, :math:`K` the
    association-mixture information kernel (:math:`0\preceq K\preceq I`),
    :math:`P_D` the detection probability and :math:`J_{\rm cal}` the
    information of any simultaneous free calibration rows.  Then

    .. math::

       J_{\rm complete}   &= H_w^{T}H_w + J_{\rm cal},\\
       J_{\rm obs}        &= P_D\,H_w^{T}KH_w + J_{\rm cal},\\
       J_{\rm cond\mid D} &= H_w^{T}KH_w + J_{\rm cal}.

    ``EXPECTED_OBSERVED`` is the correct planning quantity (the expectation over
    the latent detection indicator).  ``CONDITIONAL_DETECTED`` is the correct
    quantity for a factor that is only created *because* a detection occurred;
    the detection expectation is then realized by the presence/absence of the
    factor itself, so ``P_D`` must not appear a second time.

    The previous implementation obtained :math:`J_{\rm cond\mid D}` as
    :math:`(J_{\rm obs}-J_{\rm cal})/P_D + J_{\rm cal}`.  That is algebraically
    equivalent but numerically invalid: the subtraction cancels to
    :math:`O(\varepsilon\|J_{\rm cal}\|)` whenever :math:`P_D\|H_w^TKH_w\|\ll
    \|J_{\rm cal}\|`, the division then amplifies that rounding noise by
    :math:`1/P_D`, and at :math:`P_D=0` the ``max(P_D,10^{-12})`` guard multiplies
    it by :math:`10^{12}` before a PSD clip silently destroys the calibration
    block.  The exact matrix is now carried by the ``ActionFactor`` itself, so no
    subtraction, no division and no "conservative fallback" is required, and
    :math:`J_{\rm cond\mid D}\preceq J_{\rm complete}` holds by construction from
    :math:`K\preceq I`.
    """

    if mode is FactorInformationMode.COMPLETE_DATA:
        return _psd_part(action_factor.complete_information_normalized)
    if mode is FactorInformationMode.EXPECTED_OBSERVED:
        return _psd_part(action_factor.observed_information_normalized)
    conditional = action_factor.conditional_detected_information_normalized
    if conditional is None:
        raise ValueError(
            "conditional-on-detection information is unavailable for this "
            "ActionFactor; rebuild it with build_action_factor so the "
            "detection expectation is not applied twice"
        )
    return _psd_part(conditional)


def make_operational_action_factor(
    world: PhysicalWorld,
    action: RadarAction,
    action_factor: ActionFactor,
    *,
    epoch: int,
    arrival_epoch: int,
    participants: Sequence[str],
    mode: FactorInformationMode,
    linearization_state_normalized: np.ndarray,
    measurement_residual: np.ndarray | None,
    provenance_id: str,
    factor_id: str,
    fidelity_level: str,
    robust_kernel: RobustKernel,
    eigenvalue_floor: float,
    communication_snapshot_epoch: int | None = None,
    information_scale: float = 1.0,
) -> DynamicFactor | None:
    layout = StateLayout.from_world(world)
    x0 = np.asarray(linearization_state_normalized, dtype=float).reshape(-1)
    if x0.shape != (layout.dimension,):
        raise ValueError("linearization state has the wrong normalized dimension")
    scale = float(information_scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("information_scale must be a finite probability in [0, 1]")
    information = scale * selected_action_information(world, action, action_factor, mode)
    complete = _psd_part(action_factor.complete_information_normalized)
    eta = information @ x0
    score_projection_residual = 0.0
    factor_semantics = "expected_information_counterfactual"
    score_mapping = "none_expected_information_only"
    score_bartlett_identity_exact = False
    if measurement_residual is not None:
        residual = np.asarray(measurement_residual, dtype=float).reshape(-1)
        if (
            residual.shape != action_factor.prediction.shape
            or not np.all(np.isfinite(residual))
        ):
            raise ValueError("realized measurement residual is invalid")
        h_normalized = action_factor.jacobian_physical * layout.scales()[None, :]
        complete_score = h_normalized.T @ _solve_measurement_covariance(
            action_factor.covariance,
            residual,
        )
        exact_complete_gaussian_score = bool(
            scale == 1.0
            and np.allclose(
                information,
                complete,
                rtol=1.0e-10,
                atol=max(eigenvalue_floor, 1.0e-12),
            )
        )
        # The observed/conditional information kernel already incorporates
        # detection and association ambiguity.  The simulator does not expose
        # the corresponding exact latent-mixture responsibilities, so the
        # realized residual cannot be converted into an exact nonlinear
        # mixture score.  We instead use the local Fisher-scoring map
        #
        #   s_selected = J_selected J_complete^+ s_complete.
        #
        # For a local perturbation delta, E[s_complete] = J_complete delta;
        # because range(J_selected) is contained in range(J_complete),
        # E[s_selected] = J_selected delta.  Thus first-order mean sensitivity
        # is preserved.  In general, however,
        # Cov(s_selected) = J_selected J_complete^+ J_selected != J_selected,
        # so this object is a Gaussian information pseudo-factor, not an exact
        # observed-data mixture likelihood and not evidence of global MAP
        # optimality.  The explicit metadata below makes that boundary
        # machine-readable.
        if exact_complete_gaussian_score:
            score = complete_score
            score_projection_residual = 0.0
            score_mapping = "exact_complete_Gaussian_score"
            score_bartlett_identity_exact = True
        else:
            complete_pinv = symmetric_pinv(complete, rcond=1e-12)
            score = information @ complete_pinv @ complete_score
            score_projection_residual = float(
                np.linalg.norm(
                    complete_score
                    - complete @ complete_pinv @ complete_score
                )
                / max(
                    float(np.linalg.norm(complete_score)),
                    np.finfo(float).tiny,
                )
            )
            score_mapping = (
                "J_selected @ J_complete^+ @ complete_Gaussian_score"
            )
        eta = eta + score
        factor_semantics = "local_Fisher_scoring_Gaussian_pseudofactor"
    kind = (
        FactorKind.CALIBRATION
        if action.kind.startswith("calibration")
        else FactorKind.MEASUREMENT
    )
    factor = make_joint_information_factor(
        factor_id=factor_id,
        kind=kind,
        epoch=epoch,
        target_key=VariableKey(VariableKind.TARGET, "primary", int(epoch)),
        nuisance_key=VariableKey(VariableKind.NUISANCE, "network", int(epoch)),
        information=information,
        information_vector=eta,
        participants=participants,
        provenance_ids=(provenance_id,),
        fidelity_level=fidelity_level,
        metadata={
            "action_id": int(action.action_id),
            "action_kind": action.kind,
            "measurement_names": list(action_factor.measurement_names),
            "information_mode": mode.value,
            "information_scale": scale,
            "information_scale_semantics": (
                "expected_transport_delivery_probability"
                if scale < 1.0
                else "conditional_on_delivered_product"
            ),
            "detection_probability": float(
                action_factor.link_budget.detection_probability
            ),
            "association_information_retention": float(
                action_factor.metadata.get("association_information_retention", 1.0)
            ),
            "score_projection_residual": score_projection_residual,
            "source_action_factor_model": action_factor.metadata.get("model_level"),
            "factor_semantics": factor_semantics,
            "frozen_linearization": True,
            "expected_information_counterfactual": measurement_residual is None,
            "exact_observed_data_mixture_likelihood_factor": False,
            "global_nonlinear_MAP_claimed": False,
            "score_mapping": score_mapping,
            "score_mapping_preserves_first_order_mean_sensitivity": (
                measurement_residual is not None
            ),
            "score_Bartlett_identity_exact": score_bartlett_identity_exact,
            # The packet route is established when the observation is created.
            # A later topology change must not retroactively invalidate a
            # packet already admitted to that route.  Arrival time still gates
            # when the factor may enter the solve.
            "communication_snapshot_epoch": int(
                epoch
                if communication_snapshot_epoch is None
                else communication_snapshot_epoch
            ),
        },
        robust_kernel=robust_kernel,
        eigenvalue_floor=eigenvalue_floor,
    )
    if factor is None:
        return None
    # Canonical construction fixes observation_epoch. Arrival can be later.
    return DynamicFactor(
        factor_id=factor.factor_id,
        kind=factor.kind,
        observation_epoch=factor.observation_epoch,
        arrival_epoch=int(arrival_epoch),
        variables=factor.variables,
        jacobians=factor.jacobians,
        measurement=factor.measurement,
        covariance=factor.covariance,
        participants=factor.participants,
        provenance_ids=factor.provenance_ids,
        correlation_group=factor.correlation_group,
        robust_kernel=factor.robust_kernel,
        robust_scale=factor.robust_scale,
        association_probabilities=factor.association_probabilities,
        missing=factor.missing,
        fidelity_level=factor.fidelity_level,
        metadata=factor.metadata,
    )


@dataclass(frozen=True)
class CounterfactualResult:
    epoch: int
    joint_information: np.ndarray
    joint_covariance: np.ndarray
    position_covariance_m2: np.ndarray
    velocity_covariance_m2_s2: np.ndarray
    added_factor_ids: tuple[str, ...]
    status: str


class GraphBeliefEngine:
    """One online belief: a fixed-lag dynamic factor graph and nothing else."""

    def __init__(self, world: PhysicalWorld, config: RuntimeConfig) -> None:
        world.validate()
        config.validate()
        self.world = world
        self.config = config
        self.layout = StateLayout.from_world(world)
        self.graph = DynamicFactorGraph(
            fixed_lag_steps=config.fixed_lag_steps,
            correlation_weight_policy="equal_share",
        )
        self.current_epoch = -1
        self.current_solution: GraphSolution | None = None
        self.fixed_lag_marginalizations: list[FixedLagMarginalization] = []
        self.pre_marginalization_refresh_solves = 0

    def _add_epoch_variables(self, epoch: int) -> None:
        target = DynamicVariable(
            key=VariableKey(VariableKind.TARGET, "primary", int(epoch)),
            dimension=6,
            prior_mean=np.zeros(6),
            prior_covariance=np.eye(6),
            units=(
                "normalized_m",
                "normalized_m",
                "normalized_m",
                "normalized_m_s",
                "normalized_m_s",
                "normalized_m_s",
            ),
            anchored=True,
            prior_enabled=False,
        )
        nuisance_dimension = self.layout.dimension - 6
        nuisance = DynamicVariable(
            key=VariableKey(VariableKind.NUISANCE, "network", int(epoch)),
            dimension=nuisance_dimension,
            prior_mean=np.zeros(nuisance_dimension),
            prior_covariance=np.eye(nuisance_dimension),
            units=tuple("normalized_clock_frequency_or_delay" for _ in range(nuisance_dimension)),
            gauge_group="network_time_phase_reference",
            anchored=epoch == 0,
            prior_enabled=False,
        )
        self.graph.add_variable(target)
        self.graph.add_variable(nuisance)

    def _add_initial_prior(self) -> None:
        prior = prior_information_normalized(self.world, self.layout)
        factor = make_joint_information_factor(
            factor_id="authoritative_joint_prior:0",
            kind=FactorKind.PRIOR,
            epoch=0,
            target_key=VariableKey(VariableKind.TARGET, "primary", 0),
            nuisance_key=VariableKey(VariableKind.NUISANCE, "network", 0),
            information=prior,
            information_vector=np.zeros(self.layout.dimension),
            provenance_ids=(f"prior:{self.world.world_id}:0",),
            fidelity_level="V69_joint_target_nuisance_prior",
            metadata={
                "source": "prior_information_normalized",
                "block_prior_duplication": False,
                "shared_nuisance_covariance_preserved": True,
            },
            eigenvalue_floor=self.config.information_eigenvalue_floor,
        )
        if factor is None:
            raise RuntimeError("authoritative joint prior has zero rank")
        self.graph.add_factor(factor)

    def _add_joint_transition(
        self,
        reference_world: PhysicalWorld,
        epoch: int,
        dt_s: float,
        *,
        process_covariance_addition_normalized: np.ndarray | None = None,
        process_covariance_addition_audit: Mapping[str, Any] | None = None,
    ) -> None:
        transition = normalized_state_transition(self.layout, dt_s)
        process = _sym(normalized_process_covariance(reference_world, self.layout, dt_s))
        base_process_trace = float(np.trace(process))
        addition_trace = 0.0
        if process_covariance_addition_normalized is not None:
            addition = np.asarray(
                process_covariance_addition_normalized,
                dtype=float,
            )
            expected_shape = (self.layout.dimension, self.layout.dimension)
            if addition.shape != expected_shape:
                raise ValueError(
                    "process covariance addition must match the joint state dimension"
                )
            if not np.all(np.isfinite(addition)):
                raise ValueError("process covariance addition must be finite")
            tolerance = max(
                float(np.max(np.abs(addition))),
                np.finfo(float).tiny,
            ) * 1.0e-10
            if not np.allclose(addition, addition.T, rtol=0.0, atol=tolerance):
                raise ValueError("process covariance addition must be symmetric")
            addition = _sym(addition)
            if float(np.min(platform_symmetric_eigvalsh(addition))) < -tolerance:
                raise ValueError("process covariance addition must be positive semidefinite")
            addition_trace = float(np.trace(addition))
            process = process + addition
        eigenvalues = platform_symmetric_eigvalsh(process)
        floor = max(
            self.config.process_noise_floor,
            max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-12,
        )
        process = _psd_part(process, floor=floor)
        dimension = self.layout.dimension
        identity = np.eye(dimension)
        previous_target = VariableKey(VariableKind.TARGET, "primary", epoch - 1)
        previous_nuisance = VariableKey(VariableKind.NUISANCE, "network", epoch - 1)
        current_target = VariableKey(VariableKind.TARGET, "primary", epoch)
        current_nuisance = VariableKey(VariableKind.NUISANCE, "network", epoch)
        factor = DynamicFactor(
            factor_id=f"authoritative_joint_transition:{epoch - 1}->{epoch}",
            kind=FactorKind.TRANSITION,
            observation_epoch=int(epoch),
            arrival_epoch=int(epoch),
            variables=(previous_target, previous_nuisance, current_target, current_nuisance),
            jacobians=(
                -transition[:, :6],
                -transition[:, 6:],
                identity[:, :6],
                identity[:, 6:],
            ),
            measurement=np.zeros(dimension),
            covariance=process,
            participants=(),
            provenance_ids=(f"process:{self.world.world_id}:{epoch - 1}:{epoch}",),
            correlation_group=None,
            robust_kernel=RobustKernel.NONE,
            association_probabilities=(1.0,),
            fidelity_level="normalized_constant_velocity_and_shared_nuisance_random_walk",
            metadata={
                "full_joint_process_covariance": True,
                "target_nuisance_transition_split_only_at_variable_boundary": True,
                "process_noise_floor": floor,
                "base_process_covariance_trace": base_process_trace,
                "registered_process_covariance_addition_applied": bool(
                    process_covariance_addition_normalized is not None
                ),
                "registered_process_covariance_addition_trace": addition_trace,
                "registered_process_covariance_addition_audit": dict(
                    process_covariance_addition_audit or {}
                ),
            },
        )
        self.graph.add_factor(factor)

    def _marginalize_expired_epoch(self, next_epoch: int) -> None:
        if self.current_solution is None:
            raise RuntimeError("fixed-lag marginalization requires a solved graph")
        eliminated = int(next_epoch) - self.config.fixed_lag_steps
        if eliminated < 0:
            return
        if (any(key.epoch == eliminated for key in self.graph.variables)
                and not self.graph.solution_is_current(self.current_solution)):
            # Branch creation or a subsequently queued factor changes the
            # graph/solution binding. A real solve on this graph is necessary
            # before staging, and is an additional counted inference call.
            # Increment before the call so a failed refresh remains visible.
            self.pre_marginalization_refresh_solves += 1
            self.solve()
        boundary_epoch = eliminated + 1
        requested_boundary = (
            VariableKey(VariableKind.TARGET, "primary", boundary_epoch),
            VariableKey(VariableKind.NUISANCE, "network", boundary_epoch),
        )
        result = self.graph.marginalize_epoch_to_boundary(
            eliminate_epoch=eliminated, boundary_keys=requested_boundary,
            solved_epoch=self.current_solution.epoch, solution=self.current_solution,
            arrival_epoch=int(next_epoch))
        if result.status not in {"passed", "no_expired_variables"}:
            raise RuntimeError("fixed-lag Schur marginalization failed closed")
        if result.expired_undelivered_factor_ids:
            raise RuntimeError(
                "delayed evidence expired from the fixed-lag window before arrival; "
                "increase fixed_lag_steps or tighten the arrival bound: "
                f"{list(result.expired_undelivered_factor_ids)}")
        if result.status == "no_expired_variables":
            self.fixed_lag_marginalizations.append(result)
            return
        factor = make_scoped_information_factor(
            factor_id=f"fixed_lag_marginal_message:{eliminated}->{boundary_epoch}",
            variables=tuple(self.graph.variables[key] for key in result.boundary_keys),
            information=result.information, information_vector=result.information_vector,
            epoch=min(key.epoch for key in result.boundary_keys),
            arrival_epoch=int(next_epoch),
            provenance_ids=(result.separator_provenance_id,),
            fidelity_level="frozen_weighted_joint_Schur_boundary_message",
            metadata=result.separator_metadata,
            eigenvalue_floor=self.config.information_eigenvalue_floor)
        self.graph.commit_marginalization(result, factor)
        self.fixed_lag_marginalizations.append(result)

    def prepare_epoch(
        self,
        epoch: int,
        reference_world: PhysicalWorld,
        communication: CommunicationSnapshot,
        *,
        dt_s: float | None = None,
        process_covariance_addition_normalized: np.ndarray | None = None,
        process_covariance_addition_audit: Mapping[str, Any] | None = None,
    ) -> GraphSolution:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if epoch > self.current_epoch + 1:
            raise ValueError("belief epochs must be prepared sequentially")
        if process_covariance_addition_normalized is not None and (
            epoch == 0 or epoch <= self.current_epoch
        ):
            raise ValueError(
                "process covariance addition requires a newly prepared noninitial epoch"
            )
        if epoch > self.current_epoch and self.current_epoch >= 0:
            self._marginalize_expired_epoch(epoch)
        if epoch not in self.graph.communication:
            self.graph.add_communication_snapshot(communication)
        if epoch > self.current_epoch:
            self._add_epoch_variables(epoch)
            if epoch == 0:
                self._add_initial_prior()
            else:
                self._add_joint_transition(
                    reference_world,
                    epoch,
                    self.config.dt_s if dt_s is None else float(dt_s),
                    process_covariance_addition_normalized=(
                        process_covariance_addition_normalized
                    ),
                    process_covariance_addition_audit=(
                        process_covariance_addition_audit
                    ),
                )
            self.current_epoch = epoch
        self.current_solution = self.graph.solve(
            epoch,
            irls_iterations=2,
            diagnostics=self.config.graph_diagnostics,
        )
        return self.current_solution

    def add_factor(self, factor: DynamicFactor | None) -> None:
        if factor is not None:
            self.graph.add_factor(factor)

    def replace_factor(self, factor: DynamicFactor) -> None:
        self.graph.replace_factor(factor)

    def factor(self, factor_id: str) -> DynamicFactor:
        return self.graph.factor(factor_id)

    def solve(self) -> GraphSolution:
        if self.current_epoch < 0:
            raise RuntimeError("belief has not been initialized")
        self.current_solution = self.graph.solve(
            self.current_epoch,
            irls_iterations=4,
            diagnostics=self.config.graph_diagnostics,
        )
        return self.current_solution

    def _current_tokens(self, epoch: int | None = None) -> tuple[str, str]:
        active_epoch = self.current_epoch if epoch is None else int(epoch)
        return (
            VariableKey(VariableKind.TARGET, "primary", active_epoch).token(),
            VariableKey(VariableKind.NUISANCE, "network", active_epoch).token(),
        )

    def current_state_vector(self, solution: GraphSolution | None = None) -> np.ndarray:
        active = solution or self.current_solution
        if active is None:
            raise RuntimeError("belief solution is unavailable")
        target_token, nuisance_token = self._current_tokens(active.epoch)
        return np.concatenate((active.estimates[target_token], active.estimates[nuisance_token]))

    def current_state_covariance(self, solution: GraphSolution | None = None) -> np.ndarray:
        active = solution or self.current_solution
        if active is None:
            raise RuntimeError("belief solution is unavailable")
        target_token, nuisance_token = self._current_tokens(active.epoch)
        target_slice = slice(*active.variable_slices[target_token])
        nuisance_slice = slice(*active.variable_slices[nuisance_token])
        indices = np.r_[np.arange(target_slice.start, target_slice.stop), np.arange(nuisance_slice.start, nuisance_slice.stop)]
        return _sym(active.joint_covariance[np.ix_(indices, indices)])

    def _physical_covariances_from_joint(
        self,
        covariance: np.ndarray,
        variable_slices: Mapping[str, tuple[int, int]],
        epoch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_token = VariableKey(VariableKind.TARGET, "primary", epoch).token()
        target_slice = slice(*variable_slices[target_token])
        target_covariance = _sym(covariance[target_slice, target_slice])
        scales = self.layout.scales()[:6]
        physical = _sym(np.diag(scales) @ target_covariance @ np.diag(scales))
        return physical[:3, :3], physical[3:6, 3:6]

    def physical_covariances(
        self,
        solution: GraphSolution | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        active = solution or self.current_solution
        if active is None:
            raise RuntimeError("belief solution is unavailable")
        return self._physical_covariances_from_joint(
            active.joint_covariance,
            active.variable_slices,
            active.epoch,
        )

    def counterfactual(
        self,
        factors: Sequence[DynamicFactor],
        *,
        base_solution: GraphSolution | None = None,
    ) -> CounterfactualResult:
        base = base_solution or self.current_solution
        if base is None:
            raise RuntimeError("counterfactual requires a solved belief")
        self.graph.validate_owned_separators()
        information = np.asarray(base.joint_information, dtype=float).copy()
        base_covariance = _sym(
            np.asarray(base.joint_covariance, dtype=float).copy()
        )
        active_tokens = set(base.variable_slices)
        added: list[str] = []
        whitened_rows: list[np.ndarray] = []
        existing_provenance = {
            provenance
            for factor in self.graph.factors
            if factor.factor_id in set(base.accepted_factor_ids)
            for provenance in self.graph._fusion_provenance_lineages(factor)
        }
        accepted_hypothetical: list[DynamicFactor] = []
        seen_exact: set[tuple[str, ...]] = set()
        accepted_provenance: list[tuple[set[str], str | None]] = []
        for factor in factors:
            if factor.robust_kernel is not RobustKernel.NONE:
                raise ValueError("counterfactual expected-information factors cannot use data-dependent robust weights")
            if factor.arrival_epoch > base.epoch:
                raise ValueError("counterfactual factor is not available in the current decision epoch")
            if any(key.token() not in active_tokens for key in factor.variables):
                raise ValueError("counterfactual factor references a variable outside the active fixed lag")
            reason = self.graph.evidence_admission_reason(factor)
            if reason is not None:
                raise ValueError(f"counterfactual evidence rejected: {reason}")
            provenance_key = tuple(sorted(set(factor.provenance_ids)))
            if provenance_key in seen_exact:
                continue
            provenance_set = self.graph._fusion_provenance_lineages(factor)
            if existing_provenance.intersection(provenance_set):
                raise ValueError("counterfactual factor reuses realized evidence provenance")
            snapshot = self.graph.communication.get(base.epoch)
            if factor.participants and (
                snapshot is None or not snapshot.participants_connected(factor.participants)
            ):
                raise ValueError("counterfactual factor is not operationally available")
            overlapping = [
                (roots, group)
                for roots, group in accepted_provenance
                if roots.intersection(provenance_set)
            ]
            if any(
                factor.correlation_group is None
                or group is None
                or group != factor.correlation_group
                for _, group in overlapping
            ):
                raise ValueError(
                    "counterfactual factors overlap in raw provenance without "
                    "one declared correlation group"
                )
            seen_exact.add(provenance_key)
            accepted_hypothetical.append(factor)
            accepted_provenance.append((provenance_set, factor.correlation_group))

        ordered_hypothetical = sorted(
            accepted_hypothetical,
            key=lambda item: (
                item.arrival_epoch,
                item.observation_epoch,
                item.factor_id,
            ),
        )
        group_counts: dict[str, int] = {}
        for factor in ordered_hypothetical:
            if factor.correlation_group:
                group_counts[factor.correlation_group] = (
                    group_counts.get(factor.correlation_group, 0) + 1
                )
        for factor in ordered_hypothetical:
            weight = self.graph._association_weight(factor)
            if factor.correlation_group:
                group = factor.correlation_group
                if self.graph.correlation_weight_policy == "equal_share":
                    weight /= group_counts[group]
                else:
                    first_factor_id = min(
                        item.factor_id
                        for item in ordered_hypothetical
                        if item.correlation_group == group
                    )
                    weight *= float(factor.factor_id == first_factor_id)
            added.append(factor.factor_id)
            if weight <= 0.0:
                continue
            local = np.zeros((factor.measurement.size, information.shape[0]), dtype=float)
            for key, jacobian in zip(factor.variables, factor.jacobians):
                sl = slice(*base.variable_slices[key.token()])
                local[:, sl] = jacobian
            whitened = math.sqrt(weight) * _whitened_jacobian(
                factor.covariance, local
            )
            information += whitened.T @ whitened
            whitened_rows.append(whitened)
        information = _sym(information)
        if whitened_rows:
            # Let P = S S.T on the support of the authoritative covariance and
            # DeltaJ = W.T W for all independent hypothetical observations.
            # The exact posterior is
            #
            #   P+ = S (I + S.T DeltaJ S)^-1 S.T.
            #
            # This contraction is PSD and no larger than P by construction.  It
            # avoids the catastrophic cancellation in P - P H.T S^-1 H P that
            # appears for weak clock/delay modes at condition numbers near 1e12.
            support = _covariance_support_factor(base_covariance)
            if support.shape[1] == 0:
                covariance = np.zeros_like(base_covariance)
            else:
                stacked = np.vstack(whitened_rows)
                projected = stacked @ support
                contraction_information = _sym(
                    np.eye(support.shape[1], dtype=float)
                    + projected.T @ projected
                )
                contraction_cholesky = np.linalg.cholesky(
                    contraction_information
                )
                posterior_root = platform_solve(
                    contraction_cholesky,
                    support.T,
                ).T
                covariance = _sym(posterior_root @ posterior_root.T)
        else:
            covariance = base_covariance
        position, velocity = self._physical_covariances_from_joint(
            covariance,
            base.variable_slices,
            base.epoch,
        )
        eigenvalues = platform_symmetric_eigvalsh(information)
        threshold = _spectral_threshold(eigenvalues, 1e-10)
        status = "passed" if float(eigenvalues.min()) >= -threshold and np.all(np.isfinite(covariance)) else "failed_closed"
        return CounterfactualResult(
            epoch=base.epoch,
            joint_information=information,
            joint_covariance=covariance,
            position_covariance_m2=position,
            velocity_covariance_m2_s2=velocity,
            added_factor_ids=tuple(added),
            status=status,
        )

    def fork(self) -> "GraphBeliefEngine":
        """Create an isolated branch with the exact authoritative graph state."""

        branch = object.__new__(GraphBeliefEngine)
        branch.world = self.world
        branch.config = self.config
        branch.layout = self.layout
        branch.graph = self.graph.fork()
        branch.current_epoch = self.current_epoch
        branch.current_solution = self.current_solution
        branch.fixed_lag_marginalizations = list(
            self.fixed_lag_marginalizations
        )
        branch.pre_marginalization_refresh_solves = self.pre_marginalization_refresh_solves
        return branch

    def counterfactual_at_epoch(
        self,
        factors: Sequence[DynamicFactor],
        *,
        evaluation_epoch: int,
        reference_world: PhysicalWorld,
        communication_template: CommunicationSnapshot,
    ) -> CounterfactualResult:
        """Evaluate delayed evidence on an isolated future graph branch.

        Every candidate is propagated through the same joint transition model.
        Factors retain their observation epoch and declared arrival epoch, so a
        delayed packet informs the historical state and reaches the evaluation
        state only through graph transitions.  The current topology is repeated
        solely as a planning assumption for future *transition* epochs; each
        observation factor is checked against its explicitly recorded capture
        topology through ``communication_snapshot_epoch``.
        """

        if self.current_solution is None or self.current_epoch < 0:
            raise RuntimeError("counterfactual requires a solved belief")
        target_epoch = int(evaluation_epoch)
        if target_epoch < self.current_epoch:
            raise ValueError("counterfactual evaluation epoch cannot be in the past")
        if target_epoch - self.current_epoch >= self.config.fixed_lag_steps:
            raise ValueError("counterfactual evaluation exceeds active graph window")
        if target_epoch == self.current_epoch:
            return self.counterfactual(factors)

        branch = self.fork()
        for factor in factors:
            branch.add_factor(factor)
        for epoch in range(self.current_epoch + 1, target_epoch + 1):
            snapshot = CommunicationSnapshot(
                epoch=int(epoch),
                node_ids=communication_template.node_ids,
                edges=communication_template.edges,
            )
            branch.prepare_epoch(
                epoch,
                reference_world,
                snapshot,
                dt_s=self.config.dt_s,
            )
        solution = branch.solve()
        position, velocity = branch.physical_covariances(solution)
        accepted = set(solution.accepted_factor_ids)
        return CounterfactualResult(
            epoch=solution.epoch,
            joint_information=solution.joint_information,
            joint_covariance=solution.joint_covariance,
            position_covariance_m2=position,
            velocity_covariance_m2_s2=velocity,
            added_factor_ids=tuple(
                factor.factor_id for factor in factors if factor.factor_id in accepted
            ),
            status=solution.status,
        )

    def record(self, solution: GraphSolution | None = None) -> BeliefRecord:
        active = solution or self.current_solution
        if active is None:
            raise RuntimeError("belief solution is unavailable")
        target_token, nuisance_token = self._current_tokens(active.epoch)
        position, velocity = self.physical_covariances(active)
        return BeliefRecord(
            epoch=active.epoch,
            status=active.status,
            target_mean_normalized=active.estimates[target_token],
            nuisance_mean_normalized=active.estimates[nuisance_token],
            position_covariance_m2=position,
            velocity_covariance_m2_s2=velocity,
            accepted_factor_ids=active.accepted_factor_ids,
            rejected_factors=tuple((item.factor_id, item.reason) for item in active.rejected_factors),
            robust_weights=dict(active.robust_weights),
            information_trace=float(np.trace(active.joint_information)),
            covariance_trace=float(np.trace(active.joint_covariance)),
            exact_batch_incremental_residual=float(active.exact_batch_incremental_residual),
            metadata={
                "authoritative_backend": "DynamicFactorGraph",
                "pre_marginalization_refresh_solves": self.pre_marginalization_refresh_solves,
                "separate_controller_covariance_exists": False,
                "joint_target_nuisance_state_dimension": self.layout.dimension,
                "factor_graph_diagnostics_enabled": self.config.graph_diagnostics,
                "fixed_lag_marginalization_count": len(
                    self.fixed_lag_marginalizations
                ),
                "expired_unfused_factor_ids": [
                    factor_id
                    for item in self.fixed_lag_marginalizations
                    for factor_id in item.expired_unfused_factor_ids
                ],
                "expired_undelivered_factor_ids": [
                    factor_id
                    for item in self.fixed_lag_marginalizations
                    for factor_id in item.expired_undelivered_factor_ids
                ],
            },
        )
