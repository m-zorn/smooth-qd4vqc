from functools import partial 

from src.utils import OptimizerType
from src.utils import actions_to_gates as hard_actions_to_gates
from src.algorithms.base_qd import basic_qd_optimizer, prepare_circuit_base
from src.algorithms.smooth_qd import smooth_qd_optimizer, smooth_actions_to_gates, prepare_circuit_smooth, _set_smooth_crn_seed, deterministic_smooth_actions_to_gates
from src.algorithms.dual_qd import dual_qd_optimizer
from src.algorithms.smooth_dual_qd import smooth_dual_qd_optimizer
from src.quality_metrics import QualityMetric
from src.utils import QDParams


def make_qd_optimizer(optimizer_type: OptimizerType, solution_dim: int, metrics: list[QualityMetric], params: QDParams, gate_set: list):

    if optimizer_type == OptimizerType.BASE:
        actions_to_gates = hard_actions_to_gates
        prepare_circuit = partial(prepare_circuit_base, actions_to_gates_fn=actions_to_gates)
        scheduler, opt_arch, res_arch = basic_qd_optimizer(solution_dim, metrics, params, gate_set)
        return scheduler, opt_arch, res_arch, prepare_circuit, actions_to_gates

    elif optimizer_type == OptimizerType.DUAL:
        actions_to_gates = hard_actions_to_gates
        prepare_circuit = partial(prepare_circuit_base, actions_to_gates_fn=actions_to_gates)
        scheduler, opt_arch, res_arch = dual_qd_optimizer(solution_dim, metrics, params, gate_set)
        return scheduler, opt_arch, res_arch, prepare_circuit, actions_to_gates

    elif optimizer_type == OptimizerType.SMOOTH:
        smooth_det = bool(getattr(params, "smooth_deterministic", False))
        
        if smooth_det:
            _set_smooth_crn_seed(getattr(params, "smooth_crn_seed", None))
            actions_to_gates = deterministic_smooth_actions_to_gates
        else:
            actions_to_gates = smooth_actions_to_gates
        
        prepare_circuit = partial(prepare_circuit_smooth, actions_to_gates_fn=actions_to_gates)
        scheduler, opt_arch, res_arch = smooth_qd_optimizer(solution_dim, metrics, params)
        return scheduler, opt_arch, res_arch, prepare_circuit, actions_to_gates

    elif optimizer_type == OptimizerType.SMOOTH_DUAL:
        smooth_det = bool(getattr(params, "smooth_deterministic", False))
        
        if smooth_det:
            _set_smooth_crn_seed(getattr(params, "smooth_crn_seed", None))
            actions_to_gates = deterministic_smooth_actions_to_gates
        else:
            actions_to_gates = smooth_actions_to_gates

        prepare_circuit = partial(prepare_circuit_smooth, actions_to_gates_fn=actions_to_gates)
        scheduler, opt_arch, res_arch = smooth_dual_qd_optimizer(solution_dim, metrics, params, gate_set)
        return scheduler, opt_arch, res_arch, prepare_circuit, actions_to_gates

    else:
        raise NotImplementedError(f"Optimizer type {optimizer_type} not implemented yet.")