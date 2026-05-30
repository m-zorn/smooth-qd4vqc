import numpy as np
from typing import List

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from src.circuit import Gate, I
from src.quality_metrics import QualityMetric
from src.utils import QDParams
import threading
from src.circuit import CNOT, RX, RY, RZ, X
from src.utils import map_to_choice, remap_to_2pi

_SMOOTH_GATE_TAU: float = 0.50  # higher = smoother / more random; lower = more discrete

def _set_smooth_gate_tau(tau: float) -> None:
    """Sets global temperature used by smooth_actions_to_gates()."""
    global _SMOOTH_GATE_TAU
    _SMOOTH_GATE_TAU = float(tau)

def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def smooth_actions_to_gates(actions: List[float], gate_set: List[Gate]):
    converted = {}
    gate_choices = []
    gate_params = []

    tau = max(float(_SMOOTH_GATE_TAU), 1e-6)
    idxs = np.arange(len(gate_set), dtype=float)

    for wire, action in enumerate(actions):
        # Use rounding to 4 decimals to match 'Hard' encoder semantics and deterministic version
        a = round(abs(float(action)), 4)
        
        # Keep within valid range. prepare_circuit() already clips, but be defensive.
        a = min(max(a, 0.0), len(gate_set) - 1e-6)
        
        sign = -1.0 if float(action) < 0 else 1.0

        # Fractional part for parameters / target selection.
        frac = a - np.floor(a)

        # Distance-based logits: closer indices get higher probability.
        logits = -((a - idxs) / tau) ** 2
        probs = _stable_softmax(logits)

        # Sample gate index (stochastic gate choice; continuous in distribution w.r.t. a).
        g_idx = int(np.random.choice(len(gate_set), p=probs))
        gate = gate_set[g_idx]

        if gate == CNOT:
            # FIX: Removed /10.0 scaling which restricted targets to low indices
            target = map_to_choice(frac, len(actions))
            gate = X if target == wire else CNOT(wire, target)

        elif gate in [RX, RY, RZ]:
            angle = sign * remap_to_2pi(frac)
            gate = gate(angle)

        # else: fixed operator (H, S, T, I, ...)
        converted[wire] = gate
        gate_choices.append(g_idx)
        gate_params.append(frac)

    return converted, np.asarray(gate_choices, dtype=int), np.asarray(gate_params, dtype=float)


class SmoothSchedulerWrapper:
    def __init__(self, scheduler: Scheduler, tau0: float, tau_min: float, tau_decay: float):
        self._scheduler = scheduler
        self._tau = float(tau0)
        self._tau_min = float(tau_min)
        self._tau_decay = float(tau_decay)
        self._itr = 0
        _set_smooth_gate_tau(self._tau)

    def ask(self):
        return self._scheduler.ask()

    def tell(self, objective_batch, measure_batch):
        out = self._scheduler.tell(objective_batch, measure_batch)
        self._itr += 1
        # exponential annealing
        self._tau = max(self._tau_min, self._tau * self._tau_decay)
        _set_smooth_gate_tau(self._tau)
        return out

    def __getattr__(self, name):
        # Forward everything else (e.g., stats) to underlying scheduler.
        return getattr(self._scheduler, name)


def smooth_qd_optimizer(solution_dim: int, metrics: list[QualityMetric], params: QDParams):
    # pyribs optimization archive for annealed exploring
    optimization_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[metric.range[1] for metric in metrics],
        ranges=[metric.range for metric in metrics],
        learning_rate=params.lr,
        threshold_min=0.0
    )

    # pyribs result archive for saving the elites
    result_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[metric.range[1] for metric in metrics],
        ranges=[metric.range for metric in metrics],
    )

    # population of emitters each with their own maintained distribution
    emitters = [
        EvolutionStrategyEmitter(
            optimization_archive,
            x0=params.x0 if params.x0 is not None else np.zeros(solution_dim),
            sigma0=params.s0,
            ranker=params.rankers,
            selection_rule=params.selection_rule,
            restart_rule=params.restart_rule,
            batch_size=params.batch_size,
        )
        for _ in range(params.num_emitters)
    ]

    base_scheduler = Scheduler(optimization_archive, emitters, result_archive=result_archive)

    # Annealing hyperparams (optional fields on params; safe if not present)
    tau0 = getattr(params, "tau0", 0.75)
    tau_min = getattr(params, "tau_min", 0.10)
    tau_decay = getattr(params, "tau_decay", 0.99)

    scheduler = SmoothSchedulerWrapper(base_scheduler, tau0=tau0, tau_min=tau_min, tau_decay=tau_decay)
    return scheduler, optimization_archive, result_archive


from typing import Optional

# --- globals for deterministic smooth decoding ---
# Use thread-local storage to prevent race conditions during parallel evaluations
_smooth_local = threading.local()
_smooth_local.layer_idx = 0

_SMOOTH_GUMBELS: Optional[np.ndarray] = None  # shape: (n_layers, n_wires, |GS|)
_SMOOTH_CRN_SEED: Optional[int] = None        # if None -> derive once from np.random

def _set_smooth_layer_idx(idx: int) -> None:
    _smooth_local.layer_idx = int(idx)

def _reset_smooth_layer_idx() -> None:
    _smooth_local.layer_idx = 0

def _set_smooth_crn_seed(seed: Optional[int]) -> None:
    """
    Set the seed used to generate the *fixed* Gumbel table.
    If seed is None, we derive a seed once from the current np.random state.
    Changing the seed invalidates the cached table.
    """
    global _SMOOTH_CRN_SEED, _SMOOTH_GUMBELS
    _SMOOTH_CRN_SEED = None if seed is None else int(seed)
    _SMOOTH_GUMBELS = None  # force regeneration

def _ensure_smooth_gumbels(n_layers: int, n_wires: int, n_gates: int) -> None:
    """
    Ensure we have a fixed CRN table of Gumbel noises for (layer, wire, gate_index).
    This makes gate sampling deterministic for the whole run.
    """
    global _SMOOTH_GUMBELS, _SMOOTH_CRN_SEED

    shape = (int(n_layers), int(n_wires), int(n_gates))
    if _SMOOTH_GUMBELS is not None and _SMOOTH_GUMBELS.shape == shape:
        return

    # If no explicit seed was provided, derive a seed ONCE from current np.random state.
    # (seed_all(run_id) affects np.random, so you get per-run variability if you don't set smooth_crn_seed)
    if _SMOOTH_CRN_SEED is None:
        _SMOOTH_CRN_SEED = int(np.random.randint(0, 2**31 - 1))

    rng = np.random.default_rng(_SMOOTH_CRN_SEED)
    u = rng.random(shape)
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    _SMOOTH_GUMBELS = -np.log(-np.log(u))


def deterministic_smooth_actions_to_gates(actions: List[float], gate_set: List[Gate]):
    converted = {}
    gate_choices = []
    gate_params = []

    tau = max(float(_SMOOTH_GATE_TAU), 1e-6)  # uses your existing global temperature
    idxs = np.arange(len(gate_set), dtype=float)

    # Use thread-local layer index
    layer_idx = getattr(_smooth_local, 'layer_idx', 0)

    # Defensive: if someone calls this without prepare_circuit_smooth, ensure at least current layer exists.
    if _SMOOTH_GUMBELS is None or layer_idx >= _SMOOTH_GUMBELS.shape[0] or _SMOOTH_GUMBELS.shape[1] != len(actions) or _SMOOTH_GUMBELS.shape[2] != len(gate_set):
        _ensure_smooth_gumbels(n_layers=layer_idx + 1, n_wires=len(actions), n_gates=len(gate_set))

    for wire, action in enumerate(actions):
        a0 = float(action)
        sign = -1.0 if a0 < 0 else 1.0

        # mimic original "round then split" behavior without string parsing:
        a = round(abs(a0), 4)

        # keep within valid range
        a = min(max(a, 0.0), len(gate_set) - 1e-6)

        frac = a - np.floor(a)  # "decimal" part used for param/target (same role as original)

        # Distance-based logits: nearer indices more likely.
        # tau controls smoothness (larger tau => flatter logits)
        logits = -((a - idxs) / tau) ** 2

        # CRN Gumbel-Max: deterministic sample for this (layer, wire)
        g = _SMOOTH_GUMBELS[layer_idx, wire, :]
        g_idx = int(np.argmax(logits + g))

        gate = gate_set[g_idx]

        if gate == CNOT:
            # Preserve your existing CNOT target mapping behavior for fair ablations.
            target = map_to_choice(frac, len(actions))
            gate = X if target == wire else CNOT(wire, target)

        elif gate in [RX, RY, RZ]:
            angle = sign * remap_to_2pi(frac)
            gate = gate(angle)

        converted[wire] = gate
        gate_choices.append(g_idx)
        gate_params.append(frac)

    return converted, np.asarray(gate_choices, dtype=int), np.asarray(gate_params, dtype=float)


def prepare_circuit_smooth(circuit, solution, gate_set, n_layers, actions_to_gates_fn):
    repaired_solution = np.clip(solution, 0, len(gate_set) - 0.01)
    circuit.reset()
    circuit.set_matrix_circuit(repaired_solution.reshape(circuit.width, -1))
    assert circuit.get_matrix_circuit().shape == (circuit.width, n_layers)

    _reset_smooth_layer_idx()
    _ensure_smooth_gumbels(n_layers=n_layers, n_wires=circuit.width, n_gates=len(gate_set))

    for layer_idx, candidate_layer in enumerate(circuit.get_all_matrix_layers()):
        _set_smooth_layer_idx(layer_idx)

        new_layer, _, _ = actions_to_gates_fn(candidate_layer, gate_set)
        layer_id = len(circuit)
        sorted_gates = [new_layer.get(wire, I) for wire in range(circuit.width)]
        circuit.insert_gate_layer(layer_id, sorted_gates)

    return circuit



