import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from src.utils import QDParams
from src.quality_metrics import QualityMetric

from typing import Optional

def _infer_gate_set_size(metrics: list, params, gate_set: Optional[list] = None) -> int:
    # Check if params has gate_set as a list (common in EXPParams)
    gs = getattr(params, "gate_set", None)
    if isinstance(gs, (list, tuple)):
        return max(1, len(gs))

    if isinstance(gate_set, (list, tuple)):
        return max(1, len(gate_set))

    for attr in ("dual_n_gates", "gate_set_size", "n_gates"):
        v = getattr(params, attr, None)
        if v is not None:
            try:
                return max(1, int(v))
            except Exception:
                pass

    for m in metrics:
        for attr in ("gate_set", "gates", "gate_types"):
            if hasattr(m, attr):
                gs = getattr(m, attr)
                if isinstance(gs, (list, tuple)):
                    return max(1, len(gs))
        for attr in ("gate_set_len", "num_gates", "n_gates"):
            if hasattr(m, attr):
                try:
                    return max(1, int(getattr(m, attr)))
                except Exception:
                    pass

    print("WARNING [DualQD]: Could not infer gate set size from params or metrics. Defaulting to 3 (TINY). Diversity will be limited.")
    return 3


def _extract_structure_part(solution: np.ndarray, n_gates: int) -> np.ndarray:
    """Integer gate index per gene from a stored genotype."""
    s = np.floor(np.abs(np.asarray(solution, dtype=float))).astype(int)
    return np.clip(s, 0, max(0, n_gates - 1))


class DualChannelSchedulerWrapper:
    def __init__(
        self,
        *,
        param_scheduler: Scheduler,
        optimization_archive: GridArchive,
        result_archive: GridArchive,
        solution_dim: int,
        n_gates: int,
        params: QDParams,
    ):
        self._param_scheduler = param_scheduler
        self.optimization_archive = optimization_archive
        self.result_archive = result_archive

        self.solution_dim = int(solution_dim)
        self.n_gates = int(n_gates)

        self.batch_size = int(getattr(params, "batch_size", 1))
        self.num_emitters = int(getattr(params, "num_emitters", 1))

        # --- Structure evolution knobs (optional; safe defaults) ---
        self.struct_from_archive_prob = float(getattr(params, "dual_struct_from_archive_prob", 0.4))
        self.struct_mut_prob = float(getattr(params, "dual_struct_mut_prob", 0.50))
        self.struct_gene_mut_prob = float(getattr(params, "dual_struct_gene_mut_prob", 0.05))

        # Bias towards identity (index 0) during *random init* (optional).
        # Default to 1.0 (all identity/zeros) to match Base/Smooth QD behavior unless specified.
        self.p_identity_init = float(getattr(params, "dual_p_identity_init", 1.0))

        # Parameter channel mapping: ensure fractional part NEVER rounds up to 1.0
        # because your hard decoder uses round(action, 4) and string split.
        # So we cap at 1 - eps with eps >= 1e-4.
        self.param_eps = float(getattr(params, "dual_param_eps", 1e-3))
        if self.param_eps < 1e-4:
            self.param_eps = 1e-4

        # Keep a small pool of good structures sampled from elites
        self.struct_pool_max = int(getattr(params, "dual_struct_pool_max", 500))
        self._struct_pool: list[np.ndarray] = []

        # Per-emitter incumbent structure + best objective seen
        self._inc_structs: list[np.ndarray] = []
        self._inc_best_obj: np.ndarray = np.full(self.num_emitters, -np.inf, dtype=float)

        # Check for explicit start point (match Base/Smooth behavior)
        x0_init = getattr(params, "x0", None)

        for _ in range(self.num_emitters):
            if x0_init is not None:
                self._inc_structs.append(_extract_structure_part(x0_init, self.n_gates))
            else:
                self._inc_structs.append(self._random_structure())

        # Cached last ask()
        self._last_param_batch: Optional[np.ndarray] = None
        self._last_solution_batch: Optional[np.ndarray] = None
        self._last_structs_used: Optional[list[np.ndarray]] = None

    def _random_structure(self) -> np.ndarray:
        """Random integer gate indices, biased toward 0 for sparsity if desired."""
        if self.n_gates <= 1:
            return np.zeros(self.solution_dim, dtype=int)

        # Bias some genes to identity to start sparsely.
        s = np.random.randint(0, self.n_gates, size=self.solution_dim, dtype=int)
        if self.p_identity_init > 0:
            mask = np.random.rand(self.solution_dim) < self.p_identity_init
            s[mask] = 0
        return s

    def _sample_parent_structure(self, emitter_idx: int) -> np.ndarray:
        """
        Choose a parent structure:
          - sometimes from the pool of archive-derived structures
          - otherwise from the emitter's incumbent structure
        """
        if self._struct_pool and (np.random.rand() < self.struct_from_archive_prob):
            parent = self._struct_pool[np.random.randint(len(self._struct_pool))]
            return np.asarray(parent, dtype=int).copy()
        return np.asarray(self._inc_structs[emitter_idx], dtype=int).copy()

    def _mutate_structure(self, s: np.ndarray) -> np.ndarray:
        """Discrete mutation on gate indices."""
        if self.n_gates <= 1:
            return s

        if np.random.rand() >= self.struct_mut_prob:
            return s  # no mutation this iteration

        mask = np.random.rand(self.solution_dim) < self.struct_gene_mut_prob
        if not np.any(mask):
            mask[np.random.randint(self.solution_dim)] = True

        s = s.copy()
        s[mask] = np.random.randint(0, self.n_gates, size=int(np.sum(mask)))
        return s

    def _param_to_fraction(self, x: np.ndarray) -> np.ndarray:
        """
        Map unconstrained CMA parameters -> fractional part in [0, 1-eps].
        We use a sigmoid to keep it smooth, then clip to avoid rounding to 1.0000.
        """
        x = np.asarray(x, dtype=float)
        # stable sigmoid
        x = np.clip(x, -60.0, 60.0)
        frac = 1.0 / (1.0 + np.exp(-x))  # in (0,1)
        frac = np.clip(frac, 0.0, 1.0 - self.param_eps)
        return frac

    def ask(self) -> np.ndarray:
        # Ask internal CMA scheduler for parameter-channel samples
        param_batch = np.asarray(self._param_scheduler.ask(), dtype=float)
        if param_batch.ndim != 2 or param_batch.shape[1] != self.solution_dim:
            raise ValueError(
                f"DUAL ask(): expected param batch (B,{self.solution_dim}), got {param_batch.shape}"
            )

        total = param_batch.shape[0]
        expected = self.num_emitters * self.batch_size
        if total != expected:
            # Be robust: pyribs usually returns num_emitters*batch_size, but don't hard fail.
            self.num_emitters = max(1, total // max(1, self.batch_size))

        sol_batch = np.empty_like(param_batch, dtype=float)
        structs_used: list[np.ndarray] = []

        for e in range(self.num_emitters):
            sl = slice(e * self.batch_size, (e + 1) * self.batch_size)

            parent = self._sample_parent_structure(e)
            struct = self._mutate_structure(parent)
            structs_used.append(struct)

            frac = self._param_to_fraction(param_batch[sl])
            # Combine: integer part = structure, fractional part = params
            sol_batch[sl] = struct[None, :].astype(float) + frac

        self._last_param_batch = param_batch
        self._last_solution_batch = sol_batch
        self._last_structs_used = structs_used
        return sol_batch

    def tell(self, objective_batch, measure_batch) -> None:
        if self._last_solution_batch is None:
            raise RuntimeError("DUAL tell() called before ask().")

        obj = np.asarray(objective_batch, dtype=float).reshape(-1)
        meas = np.asarray(measure_batch, dtype=float)

        if obj.shape[0] != self._last_solution_batch.shape[0]:
            raise ValueError(
                f"DUAL tell(): objective length {obj.shape[0]} != last batch {self._last_solution_batch.shape[0]}"
            )
        if meas.shape[0] != self._last_solution_batch.shape[0]:
            raise ValueError(
                f"DUAL tell(): measures length {meas.shape[0]} != last batch {self._last_solution_batch.shape[0]}"
            )

        # 1) Update *full* QD archives with the real measures (structure affects measures!)
        self.optimization_archive.add(self._last_solution_batch, obj, meas)
        add_info = self.result_archive.add(self._last_solution_batch, obj, meas)

        # 2) Feed internal CMA scheduler (param channel only) with dummy measures (1D),
        #    so it behaves like pure objective-driven ES per emitter.
        dummy_meas = np.zeros((obj.shape[0], 1), dtype=float)
        self._param_scheduler.tell(obj, dummy_meas)

        # 3) Update structure pool from successful inserts (best-effort; robust to pyribs versions)
        inserted_mask = None
        try:
            if isinstance(add_info, dict) and "status" in add_info:
                inserted_mask = np.asarray(add_info["status"], dtype=int) > 0
            elif hasattr(add_info, "status"):
                inserted_mask = np.asarray(add_info.status, dtype=int) > 0
        except Exception:
            inserted_mask = None

        if inserted_mask is None:
            # Fallback: take a few top solutions by objective
            topk = min(5, obj.shape[0])
            idxs = np.argsort(obj)[-topk:]
        else:
            idxs = np.where(inserted_mask)[0]
            if idxs.size == 0:
                topk = min(5, obj.shape[0])
                idxs = np.argsort(obj)[-topk:]

        for idx in np.asarray(idxs, dtype=int).tolist():
            s = _extract_structure_part(self._last_solution_batch[idx], self.n_gates)
            self._struct_pool.append(s)

        if len(self._struct_pool) > self.struct_pool_max:
            self._struct_pool = self._struct_pool[-self.struct_pool_max :]

        # 4) Per-emitter incumbent update: if this iteration improved best objective for emitter,
        #    keep the mutated structure as its new incumbent.
        if self._last_structs_used is not None:
            for e in range(self.num_emitters):
                sl = slice(e * self.batch_size, (e + 1) * self.batch_size)
                best_local = float(np.max(obj[sl]))
                if best_local > float(self._inc_best_obj[e]):
                    self._inc_best_obj[e] = best_local
                    self._inc_structs[e] = np.asarray(self._last_structs_used[e], dtype=int).copy()


def dual_qd_optimizer(solution_dim: int, metrics: list[QualityMetric], params: QDParams, gate_set: list):
    # Full QD archives (same as basic)
    optimization_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[metric.range[1] for metric in metrics],
        ranges=[metric.range for metric in metrics],
        learning_rate=params.lr,
        threshold_min=0.0,
    )
    result_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[metric.range[1] for metric in metrics],
        ranges=[metric.range for metric in metrics],
    )

    # Internal "parameter-only" scheduler: 1D dummy archive so ES runs but ignores QD measures.
    param_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[1],
        ranges=[(0.0, 1.0)],
        learning_rate=1.0,
        threshold_min=-1e12,  # allow negative objectives if they ever occur
    )
    param_result_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[1],
        ranges=[(0.0, 1.0)],
    )

    param_ranker = getattr(params, "rankers", "obj")  # "obj" keeps it simple/stable
    x0_param = getattr(params, "x0", None)
    if x0_param is None:
        x0_param = np.zeros(solution_dim, dtype=float)

    emitters = [
        EvolutionStrategyEmitter(
            param_archive,
            x0=x0_param,
            sigma0=getattr(params, "s0", 0.5),
            ranker=param_ranker,
            selection_rule=params.selection_rule,
            restart_rule=params.restart_rule,
            batch_size=params.batch_size,
        )
        for _ in range(params.num_emitters)
    ]
    param_scheduler = Scheduler(param_archive, emitters, result_archive=param_result_archive)

    n_gates = _infer_gate_set_size(metrics, params, gate_set)
    print(f"INFO [DualQD]: Inferred gate set size = {n_gates}") # Confirm fix works
    scheduler = DualChannelSchedulerWrapper(
        param_scheduler=param_scheduler,
        optimization_archive=optimization_archive,
        result_archive=result_archive,
        solution_dim=solution_dim,
        n_gates=n_gates,
        params=params,
    )
    return scheduler, optimization_archive, result_archive


