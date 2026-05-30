from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np


_EPS = 1e-12


def _to_1d_float(a: Sequence[float]) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


def volatility_series(best_so_far: Sequence[float], eps: float = _EPS) -> np.ndarray:
    """
    Cumulative volatility ("jumpiness") of best-so-far improvements.

    Let b_t be best-so-far (monotone). Δ_t = b_t - b_{t-1} >= 0.
    Volatility(t) = max_{k<=t} Δ_k / (sum_{k<=t} Δ_k + eps)

    Returns an array v_t aligned with best_so_far, with v_0 = 0.
    Lower => improvements spread out more smoothly.
    """
    b = _to_1d_float(best_so_far)
    if len(b) == 0:
        return np.array([], dtype=float)
    if len(b) == 1:
        return np.array([0.0], dtype=float)

    deltas = np.diff(b, prepend=b[0])
    deltas[0] = 0.0
    deltas = np.maximum(deltas, 0.0)

    cum_sum = np.cumsum(deltas)
    cum_max = np.maximum.accumulate(deltas)
    v = np.zeros_like(b, dtype=float)
    v[1:] = cum_max[1:] / (cum_sum[1:] + eps)
    return v


def smoothness_series(signal: Sequence[float], eps: float = _EPS) -> np.ndarray:
    """
    Cumulative normalized total variation (NormTV).

    TV_t = sum_{k=1..t} |x_k - x_{k-1}|
    Net_t = |x_t - x_0|
    NormTV_t = TV_t / (Net_t + eps)

    Interpretation:
      ~1 => mostly monotone / smooth
      >>1 => oscillatory / jagged
    """
    x = _to_1d_float(signal)
    if len(x) == 0:
        return np.array([], dtype=float)
    if len(x) == 1:
        return np.array([0.0], dtype=float)

    dx = np.abs(np.diff(x, prepend=x[0]))
    dx[0] = 0.0
    tv = np.cumsum(dx)
    net = np.abs(x - x[0])
    return tv / (net + eps)

@dataclass(frozen=True)
class GateSignature:
    """A minimal, stable signature that captures circuit *structure* (not continuous angles)."""
    kind: str
    control: Optional[int] = None
    target: Optional[int] = None


def _gate_signature(gate) -> GateSignature:
    """
    Generic signature extractor.
    Works even if Gate has different field names; extend if your Gate class differs.
    """
    kind = getattr(gate, "name", None) or getattr(gate, "kind", None) or gate.__class__.__name__
    control = getattr(gate, "control", None)
    target = getattr(gate, "target", None)
    return GateSignature(kind=str(kind), control=control, target=target)


def decode_gate_signatures_grid(
    solution: np.ndarray,
    gate_set,
    n_qubits: int,
    n_layers: int,
    actions_to_gates_fn: Callable,
    identity_gate,
) -> np.ndarray:
    """
    Decodes a solution vector to an (n_layers, n_qubits) grid of GateSignature.
    This is intentionally structure-only.

    - solution: flat vector length n_qubits*n_layers
    - actions_to_gates_fn: typically src.utils.actions_to_gates
    - identity_gate: typically src.circuit.I
    """
    # Match prepare_circuit behavior: clip to valid gate selector range
    repaired = np.clip(solution, 0, len(gate_set) - 0.01)
    mat = repaired.reshape(n_qubits, n_layers)

    grid = np.empty((n_layers, n_qubits), dtype=object)

    for layer_idx in range(n_layers):
        candidate_layer = mat[:, layer_idx]
        layer_dict, _, _ = actions_to_gates_fn(candidate_layer, gate_set)

        # Fill missing wires with identity
        for w in range(n_qubits):
            gate = layer_dict.get(w, identity_gate)
            grid[layer_idx, w] = _gate_signature(gate)

    return grid


def phenotypic_instability_series(
    representative_solutions: Sequence[np.ndarray],
    gate_set,
    n_qubits: int,
    n_layers: int,
    actions_to_gates_fn: Callable,
    identity_gate,
) -> np.ndarray:
    """
    Returns D_t = normalized Hamming distance of decoded structure between t and t-1:
      D_t = (1/(n_qubits*n_layers)) * count(structure changes)

    D_0 = 0.
    """
    if len(representative_solutions) == 0:
        return np.array([], dtype=float)
    if len(representative_solutions) == 1:
        return np.array([0.0], dtype=float)

    prev = decode_gate_signatures_grid(
        representative_solutions[0],
        gate_set=gate_set,
        n_qubits=n_qubits,
        n_layers=n_layers,
        actions_to_gates_fn=actions_to_gates_fn,
        identity_gate=identity_gate,
    )

    D = np.zeros(len(representative_solutions), dtype=float)
    denom = float(n_qubits * n_layers)

    for t in range(1, len(representative_solutions)):
        cur = decode_gate_signatures_grid(
            representative_solutions[t],
            gate_set=gate_set,
            n_qubits=n_qubits,
            n_layers=n_layers,
            actions_to_gates_fn=actions_to_gates_fn,
            identity_gate=identity_gate,
        )
        changed = np.not_equal(cur, prev).sum()
        D[t] = changed / denom
        prev = cur

    return D

# Convenience wrapper: compute all requested metrics at once
def compute_convergence_metrics(
    best_fit: Sequence[float],
    stability_signal: Sequence[float],
    representative_solutions: Sequence[np.ndarray],
    gate_set,
    n_qubits: int,
    n_layers: int,
    actions_to_gates_fn: Callable,
    identity_gate,
) -> dict[str, np.ndarray]:
    """
    Returns metric name -> series aligned with steps.

    - best_fit: best-so-far objective series (monotone)
    - stability_signal: non-monotone series, e.g. QD_score per step
    - representative_solutions: one genotype per step (e.g. argmax solution in batch)
    """
    return {
        "metric_volatility": volatility_series(best_fit),
        "metric_smoothness": smoothness_series(stability_signal),
        "metric_pheno_instability": phenotypic_instability_series(
            representative_solutions,
            gate_set=gate_set,
            n_qubits=n_qubits,
            n_layers=n_layers,
            actions_to_gates_fn=actions_to_gates_fn,
            identity_gate=identity_gate,
        ),
    }
