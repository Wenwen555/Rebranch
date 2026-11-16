"""Execution trace divergence localization and signed failure signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rebranch.exec.trace import CallRecord, Trace


@dataclass(frozen=True)
class FailureSignature:
    """Hashable failure signature preserving divergence location and deviation signs."""

    lineno: int
    occurrence: int
    failure_type: str
    signs: tuple[tuple[str, int], ...]


def _signed_deviations(
    obs_trace: Mapping[str, float],
    obs_nom: Mapping[str, float],
    obs_scale: Mapping[str, float],
) -> dict[str, float]:
    devs: dict[str, float] = {}
    for key, val in obs_trace.items():
        scale = float(obs_scale.get(key, 1.0))
        if scale <= 0.0:
            scale = 1.0
        devs[key] = (float(val) - float(obs_nom.get(key, 0.0))) / scale
    return devs


def first_divergence(
    trace: Trace,
    nominal_trace: Trace,
    obs_scale: Mapping[str, float],
    delta: float = 3.0,
) -> tuple[int, dict[str, float]]:
    """Find the earliest call index where `trace` diverges from `nominal_trace`.

    Returns `(call_index, signed_normalized_deviations)`.
    """
    if not trace.calls:
        return 0, {}

    n_nom = len(nominal_trace.calls)
    if n_nom == 0:
        devs = _signed_deviations(trace.calls[0].obs, {}, obs_scale)
        return 0, devs

    n_common = min(len(trace.calls), n_nom)
    best_idx = 0
    best_devs: dict[str, float] = {}
    best_mag = -1.0

    for idx in range(n_common):
        c_t: CallRecord = trace.calls[idx]
        c_n: CallRecord = nominal_trace.calls[idx]
        devs = _signed_deviations(c_t.obs, c_n.obs, obs_scale)
        mag = max((abs(v) for v in devs.values()), default=0.0)
        if mag > best_mag:
            best_mag = mag
            best_idx = idx
            best_devs = devs
        if (
            mag >= delta
            or (c_t.name, c_t.lineno, c_t.occurrence)
            != (c_n.name, c_n.lineno, c_n.occurrence)
        ):
            return idx, devs

    if len(trace.calls) > n_common:
        idx = n_common
        devs = _signed_deviations(
            trace.calls[idx].obs, nominal_trace.calls[-1].obs, obs_scale
        )
        return idx, devs

    if trace.failure_call is not None and 0 <= trace.failure_call < len(trace.calls):
        if best_mag <= 1e-6:
            ref_idx = min(trace.failure_call, n_nom - 1)
            devs = _signed_deviations(
                trace.calls[trace.failure_call].obs,
                nominal_trace.calls[ref_idx].obs,
                obs_scale,
            )
            return trace.failure_call, devs

    return best_idx, best_devs


def top_deviation_signs(
    deviations: Mapping[str, float], top_k: int = 2
) -> tuple[tuple[str, int], ...]:
    """Extract top-k most-deviating snapshot fields and their signs (+1 / -1)."""
    ranked = sorted(
        ((k, float(v)) for k, v in deviations.items() if abs(float(v)) > 1e-6),
        key=lambda kv: (-abs(kv[1]), kv[0]),
    )
    return tuple((k, 1 if v > 0.0 else -1) for k, v in ranked[:top_k])


def failure_signature(
    trace: Trace,
    nominal_trace: Trace,
    obs_scale: Mapping[str, float],
    delta: float = 3.0,
) -> FailureSignature:
    """Compute a `FailureSignature` for a failing trace relative to `nominal_trace`."""
    call_idx, devs = first_divergence(trace, nominal_trace, obs_scale, delta=delta)
    if trace.calls and 0 <= call_idx < len(trace.calls):
        call = trace.calls[call_idx]
        lineno = int(call.lineno)
        occurrence = int(call.occurrence)
    else:
        lineno = 0
        occurrence = 0

    failure_type = (
        trace.failure
        or trace.safety_violation
        or trace.error
        or ("success" if trace.success else "unknown")
    )
    signs = top_deviation_signs(devs, top_k=2)
    return FailureSignature(
        lineno=lineno,
        occurrence=occurrence,
        failure_type=str(failure_type),
        signs=signs,
    )
