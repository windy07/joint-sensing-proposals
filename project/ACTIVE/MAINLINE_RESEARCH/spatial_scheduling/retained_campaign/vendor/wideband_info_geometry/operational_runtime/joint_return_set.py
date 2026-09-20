"""A declared affine single-Bernoulli return-set law, not the legacy receiver.

The reference density is that of a uniformly shuffled ordered packet.  The
payload stores a canonical sorted representative of the unlabelled set.  Its
density on the sorted Lebesgue domain (or the FISST set density) is m! times
the ordered density returned by ``factor_branches``.  This common factor may
cancel in posterior branch weights; it must not be omitted when reporting
an absolute density in a different reference measure.

All probabilities, affine maps and covariances are frozen at capture.  State
dependent detection, nonlinear extraction, gating, singular Gaussian density
and equivalence to the legacy R2 capture model are outside this module.
The caller supplies an existence hypothesis for inference; the payload never
contains the simulator's existence or target/clutter labels.  This module
constructs likelihood alternatives and cannot publish a posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import ClassVar

import numpy as np

from .covariance_coordinates import (
    CovarianceModelError,
    checked_covariance,
    gaussian_draw,
    gaussian_log_density,
)


class ReturnSetModelError(ValueError):
    """Malformed or unsupported declared likelihood, never zero evidence."""


def _integer(value, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ReturnSetModelError(name + '_INVALID_INTEGER')
    return value


def _text(value, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ReturnSetModelError(name + '_INVALID_TEXT')
    return value


def _float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ReturnSetModelError(name + '_NOT_REAL')
    out = float(value)
    if not math.isfinite(out):
        raise ReturnSetModelError(name + '_NONFINITE')
    # A single canonical representation for signed zero in content digests.
    return 0.0 if out == 0.0 else out


def _vector(values, name: str) -> tuple[float, ...]:
    try:
        return tuple(_float(v, name) for v in values)
    except TypeError as exc:
        raise ReturnSetModelError(name + '_NOT_VECTOR') from exc


def _matrix(values, rows: int, columns: int, name: str):
    try:
        out = tuple(_vector(row, name) for row in values)
    except TypeError as exc:
        raise ReturnSetModelError(name + '_NOT_MATRIX') from exc
    if len(out) != rows or any(len(row) != columns for row in out):
        raise ReturnSetModelError(name + '_SHAPE')
    return out


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _spd(values, dimension: int, name: str):
    out = _matrix(values, dimension, dimension, name)
    # No silent covariance symmetrization or PSD clipping in the declared law.
    if any(out[i][j] != out[j][i] for i in range(dimension)
           for j in range(dimension)):
        raise ReturnSetModelError(name + '_NOT_SYMMETRIC')
    try:
        checked_covariance(np.asarray(out), require_spd=True)
    except (CovarianceModelError, np.linalg.LinAlgError) as exc:
        raise ReturnSetModelError(name + '_UNSUPPORTED_SPD') from exc
    return out


@dataclass(frozen=True)
class JointReturnSetPayload:
    action_id: int
    observation_epoch: int
    provenance_id: str
    law_digest: str
    measurements: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        _integer(self.action_id, 'ACTION_ID')
        _integer(self.observation_epoch, 'OBSERVATION_EPOCH')
        _text(self.provenance_id, 'PROVENANCE_ID')
        if (type(self.law_digest) is not str or len(self.law_digest) != 64
                or any(c not in '0123456789abcdef' for c in self.law_digest)):
            raise ReturnSetModelError('LAW_DIGEST_INVALID')
        try:
            rows = tuple(_vector(row, 'MEASUREMENT') for row in self.measurements)
        except TypeError as exc:
            raise ReturnSetModelError('MEASUREMENTS_INVALID') from exc
        if rows and (not rows[0] or any(len(row) != len(rows[0]) for row in rows)):
            raise ReturnSetModelError('MEASUREMENTS_INCONSISTENT_DIMENSION')
        # Never deduplicate: this is a finite packet, with canonical ordering.
        object.__setattr__(self, 'measurements', tuple(sorted(rows)))

    @property
    def digest(self) -> str:
        return _digest({'schema': 'joint-return-set-payload.v1',
                        'action_id': self.action_id,
                        'observation_epoch': self.observation_epoch,
                        'provenance_id': self.provenance_id,
                        'law_digest': self.law_digest,
                        'measurements': self.measurements})


@dataclass(frozen=True)
class AffineReturnBranch:
    """``exp(log_constant) * Normal(y; H state, R)`` in ordered convention.

    Empty rows mean a scalar likelihood.  Normalization constants from all
    Gaussian rows in H/y/R must be retained by the authority.  A branch is an
    inference alternative; association_index is never simulator truth.
    """
    event: str
    association_index: int | None
    log_constant: float
    H: tuple[tuple[float, ...], ...]
    y: tuple[float, ...]
    R: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if self.event not in ('absent', 'missed', 'associated'):
            raise ReturnSetModelError('BRANCH_EVENT_INVALID')
        if self.event == 'associated':
            _integer(self.association_index, 'ASSOCIATION_INDEX')
        elif self.association_index is not None:
            raise ReturnSetModelError('NONASSOCIATED_INDEX')
        object.__setattr__(self, 'log_constant', _float(self.log_constant, 'LOG_CONSTANT'))
        y = _vector(self.y, 'BRANCH_Y')
        try:
            h = tuple(_vector(row, 'BRANCH_H') for row in self.H)
        except TypeError as exc:
            raise ReturnSetModelError('BRANCH_H_INVALID') from exc
        if len(h) != len(y) or (h and (not h[0] or any(len(v) != len(h[0]) for v in h))):
            raise ReturnSetModelError('BRANCH_H_SHAPE')
        r = _matrix(self.R, len(y), len(y), 'BRANCH_R')
        object.__setattr__(self, 'H', h)
        object.__setattr__(self, 'y', y)
        object.__setattr__(self, 'R', r)


@dataclass(frozen=True)
class FrozenAffineReturnLaw:
    """One capture context and one explicit finite-count generative law.

    ``capture_units[k][j]`` declares the physical unit of coordinate j of key
    k.  H and G map flattened capture coordinates into measurement_units;
    covariance entry ij has the product of measurement units i and j.  Names
    are declarations, not automatic dimensional inference or SI conversion.

    rho is accepted only within 1e-12 of a probability vector's unit total.
    Its exposed tuple is the normalized effective PMF used by both sampling
    and likelihood, and its digest binds that tuple.  Zero entries remain
    exactly zero.  No Poisson/PPP assumption is implicit in a finite rho.
    """
    action_id: int
    observation_epoch: int
    capture_keys: tuple[str, ...]
    capture_dimensions: tuple[int, ...]
    capture_units: tuple[tuple[str, ...], ...]
    measurement_names: tuple[str, ...]
    measurement_units: tuple[str, ...]
    H: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    R: tuple[tuple[float, ...], ...]
    G: tuple[tuple[float, ...], ...]
    g: tuple[float, ...]
    Rc: tuple[tuple[float, ...], ...]
    p_D: float
    rho: tuple[float, ...]

    reference_measure: ClassVar[str] = 'uniformly_shuffled_ordered_packet'
    schema: ClassVar[str] = 'frozen-affine-return-law.v1'

    def __post_init__(self):
        _integer(self.action_id, 'ACTION_ID')
        _integer(self.observation_epoch, 'OBSERVATION_EPOCH')
        keys = tuple(_text(k, 'CAPTURE_KEY') for k in self.capture_keys)
        dims = tuple(_integer(k, 'CAPTURE_DIMENSION', positive=True)
                     for k in self.capture_dimensions)
        if not keys or len(keys) != len(dims) or len(set(keys)) != len(keys):
            raise ReturnSetModelError('CAPTURE_KEYS_DIMENSIONS_INVALID')
        units = tuple(tuple(_text(u, 'CAPTURE_UNIT') for u in row)
                      for row in self.capture_units)
        if len(units) != len(dims) or any(len(row) != d for row, d in zip(units, dims)):
            raise ReturnSetModelError('CAPTURE_UNITS_SHAPE')
        names = tuple(_text(k, 'MEASUREMENT_NAME') for k in self.measurement_names)
        mu = tuple(_text(k, 'MEASUREMENT_UNIT') for k in self.measurement_units)
        if not names or len(set(names)) != len(names) or len(names) != len(mu):
            raise ReturnSetModelError('MEASUREMENT_NAMES_UNITS_INVALID')
        d, n = len(names), sum(dims)
        for field, value in [('capture_keys', keys), ('capture_dimensions', dims),
                             ('capture_units', units), ('measurement_names', names),
                             ('measurement_units', mu)]:
            object.__setattr__(self, field, value)
        for field in ('H', 'G'):
            object.__setattr__(self, field, _matrix(getattr(self, field), d, n, field))
        for field in ('b', 'g'):
            value = _vector(getattr(self, field), field)
            if len(value) != d:
                raise ReturnSetModelError(field + '_SHAPE')
            object.__setattr__(self, field, value)
        for field in ('R', 'Rc'):
            object.__setattr__(self, field, _spd(getattr(self, field), d, field))
        pd = _float(self.p_D, 'P_D')
        if not 0.0 <= pd <= 1.0:
            raise ReturnSetModelError('P_D_OUT_OF_RANGE')
        object.__setattr__(self, 'p_D', pd)
        rho = _vector(self.rho, 'RHO')
        total = math.fsum(rho)
        if not rho or any(p < 0.0 for p in rho) or abs(total - 1.0) > 1e-12:
            raise ReturnSetModelError('RHO_NOT_PROBABILITY_VECTOR')
        normalized = [p / total for p in rho]
        pivot = max(range(len(normalized)), key=normalized.__getitem__)
        normalized[pivot] = 1.0 - math.fsum(p for i, p in enumerate(normalized) if i != pivot)
        if normalized[pivot] <= 0.0:
            raise ReturnSetModelError('RHO_NORMALIZATION_RANGE')
        object.__setattr__(self, 'rho', tuple(normalized))

    @property
    def digest(self) -> str:
        fields = ('action_id', 'observation_epoch', 'capture_keys', 'capture_dimensions',
                  'capture_units', 'measurement_names', 'measurement_units', 'H', 'b',
                  'R', 'G', 'g', 'Rc', 'p_D', 'rho')
        return _digest({'schema': self.schema, 'reference_measure': self.reference_measure,
                        **{name: getattr(self, name) for name in fields}})

    def validate_payload(self, payload: JointReturnSetPayload) -> None:
        if type(payload) is not JointReturnSetPayload:
            raise ReturnSetModelError('PAYLOAD_TYPE')
        if (payload.action_id != self.action_id
                or payload.observation_epoch != self.observation_epoch
                or payload.law_digest != self.digest):
            raise ReturnSetModelError('PAYLOAD_CAPTURE_CONTEXT_MISMATCH')
        if any(len(row) != len(self.measurement_names) for row in payload.measurements):
            raise ReturnSetModelError('PAYLOAD_MEASUREMENT_DIMENSION')
        if len(payload.measurements) > len(self.rho):
            raise ReturnSetModelError('PAYLOAD_CARDINALITY_OUTSIDE_LAW')

    def sample(self, rng, state, exists: bool, *, provenance_id: str) -> JointReturnSetPayload:
        """Sample the declared law; no legacy same-seed compatibility promise.

        The random call order is count choice; detection uniform iff exists;
        target Gaussian iff detected; count-many independent clutter Gaussians.
        Sorting removes the target identity; no extra random permutation is
        needed for the quotient set.  Randomness and truth stay in capture.
        """
        if type(exists) is not bool:
            raise ReturnSetModelError('EXISTENCE_HYPOTHESIS_NOT_BOOL')
        _text(provenance_id, 'PROVENANCE_ID')
        x = np.asarray(_vector(state, 'CAPTURE_STATE'))
        if x.shape != (sum(self.capture_dimensions),):
            raise ReturnSetModelError('CAPTURE_STATE_DIMENSION')
        # Validate all deterministic means before consuming the random stream.
        target_mean = np.asarray(self.H) @ x + np.asarray(self.b)
        background_mean = np.asarray(self.G) @ x + np.asarray(self.g)
        if not (np.all(np.isfinite(target_mean)) and np.all(np.isfinite(background_mean))):
            raise ReturnSetModelError('CAPTURE_MEAN_OVERFLOW')
        count = int(rng.choice(len(self.rho), p=np.asarray(self.rho)))
        detected = bool(exists and float(rng.random()) < self.p_D)
        measurements = []
        if detected:
            primary = gaussian_draw(rng, target_mean, np.asarray(self.R))
            measurements.append(tuple(float(v) for v in primary))
            center = primary
        else:
            center = background_mean
        for _ in range(count):
            value = gaussian_draw(rng, center, np.asarray(self.Rc))
            measurements.append(tuple(float(v) for v in value))
        return JointReturnSetPayload(self.action_id, self.observation_epoch,
                                     provenance_id, self.digest, tuple(measurements))

    def factor_branches(self, payload: JointReturnSetPayload, exists: bool) -> tuple[AffineReturnBranch, ...]:
        """Enumerate the entire conditional likelihood for one E hypothesis.

        Structural zero terms are omitted.  An empty tuple means zero model
        evidence under this E hypothesis, not numeric failure.  Unsupported
        numeric values raise; no underflow-based pruning is performed.  The
        caller supplies parent/existence weights and a hard bank budget.
        """
        if type(exists) is not bool:
            raise ReturnSetModelError('EXISTENCE_HYPOTHESIS_NOT_BOOL')
        self.validate_payload(payload)
        z, d = payload.measurements, len(self.measurement_names)
        m = len(z)
        rho_m = self.rho[m] if m < len(self.rho) else 0.0
        branches = []
        miss_probability = 1.0 - self.p_D if exists else 1.0
        if rho_m > 0.0 and miss_probability > 0.0:
            h = tuple(row for _ in range(m) for row in self.G)
            y = tuple(zj[i] - self.g[i] for zj in z for i in range(d))
            r = tuple(tuple(self.Rc[i % d][j % d] if i // d == j // d else 0.0
                            for j in range(m * d)) for i in range(m * d))
            branches.append(AffineReturnBranch('missed' if exists else 'absent', None,
                            math.log(rho_m) + math.log(miss_probability), h, y, r))
        rho_previous = self.rho[m - 1] if 1 <= m <= len(self.rho) else 0.0
        if exists and self.p_D > 0.0 and rho_previous > 0.0:
            base = math.log(self.p_D) + math.log(rho_previous) - math.log(m)
            for j, zj in enumerate(z):
                terms = [base]
                for k, zk in enumerate(z):
                    if k != j:
                        residual = np.asarray(zk) - np.asarray(zj)
                        terms.append(gaussian_log_density(residual, np.asarray(self.Rc)))
                constant = math.fsum(terms)
                if not math.isfinite(constant):
                    raise ReturnSetModelError('BRANCH_LOG_CONSTANT_NUMERIC_RANGE')
                branches.append(AffineReturnBranch('associated', j, constant, self.H,
                                tuple(zj[i] - self.b[i] for i in range(d)), self.R))
        return tuple(branches)
