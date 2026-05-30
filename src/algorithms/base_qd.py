import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from src.quality_metrics import QualityMetric
from src.utils import QDParams
from src.circuit import I

def basic_qd_optimizer(solution_dim:int, metrics:list[QualityMetric], params: QDParams, gate_set: list):
    
    # pyribs optimization archive for annealed exploring
    optimization_archive = GridArchive(solution_dim=solution_dim,
                        dims=[metric.range[1] for metric in metrics],
                        ranges=[metric.range for metric in metrics],
                        learning_rate=params.lr,
                        threshold_min=0.0)
    # pyribs result archive for saving the elites
    result_archive = GridArchive(solution_dim=solution_dim,
                        dims=[metric.range[1] for metric in metrics],
                        ranges=[metric.range for metric in metrics]
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
        ) for _ in range(params.num_emitters)
    ]
    # distribution update algorithm
    scheduler = Scheduler(optimization_archive, emitters, result_archive=result_archive)
    return scheduler, optimization_archive, result_archive

def prepare_circuit_base(circuit, solution, gate_set, n_layers, actions_to_gates_fn):
    repaired_solution = np.clip(solution, 0, len(gate_set)-0.01)
    circuit.reset()
    circuit.set_matrix_circuit(repaired_solution.reshape(circuit.width, -1))
    assert circuit.get_matrix_circuit().shape == (circuit.width,n_layers)
    for candidate_layer in circuit.get_all_matrix_layers():
        new_layer, _ , _ = actions_to_gates_fn(candidate_layer, gate_set)
        layer_id = len(circuit)
        sorted_gates = [new_layer.get(wire, I) for wire in range(circuit.width)]
        circuit.insert_gate_layer(layer_id, sorted_gates)
    return circuit
