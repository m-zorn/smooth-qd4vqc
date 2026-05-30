import sys
import numpy as np
from tqdm import trange
import pandas as pd
from typing import List

from src.circuit import Optimization_Circuit, Gate, I
from src.utils import nx_graph_from, to_z_basis
from src.quality_metrics import QualityMetric, init_metric
from src.utils import OptimizerType, TaskType, QDParams

from src.algorithms import make_qd_optimizer

from src.convergence_metrics import compute_convergence_metrics

def extract_parameters(circuit: Optimization_Circuit):
    param_gates = []
    initial_params = []
    # We iterate sorted by layer to be deterministic
    for layer in sorted(circuit.gate_table.keys()):
        gates = circuit.gate_table[layer]
        for gate in gates:
            if hasattr(gate, 'angle'):
                param_gates.append(gate)
                initial_params.append(gate.angle)
    return param_gates, np.array(initial_params)

def update_parameters(param_gates, new_params):
    for gate, val in zip(param_gates, new_params):
        gate.angle = val

def qd_batch(circuit, solution_batch, gate_set, problem_input, objective_fn, metrics: List[QualityMetric], task_type:TaskType, n_layers: int, prepare_circuit_fn, optimizer_params:QDParams=None):
    objective_batch = []
    measures_batch = []

    # If generalization mode, nx_graph expects a list of graph objects (or edges)
    # We pre-process them into edge lists for speed if they are full Graph objects
    target_instances = problem_input if task_type == TaskType.GRAPH_GENERAL else [problem_input]

    # Pre-compute target signatures for RECOVER task to avoid re-running in loop
    target_sigs = None
    if task_type == TaskType.RECOVER:
        target_solution = target_instances[0]
        
        def get_gate_sig(g):
            t = type(g)
            if hasattr(g, 'angle'):
                return (t, g.angle)
            if hasattr(g, 'target'):
                return (t, g.target)
            return (t, None)
            
        circuit = prepare_circuit_fn(circuit, target_solution, gate_set, n_layers)
        target_sigs = []
        for layer in sorted(circuit.gate_table.keys()):
            for gate in circuit.gate_table[layer]:
                target_sigs.append(get_gate_sig(gate))

    for solution in solution_batch:    
        circuit = prepare_circuit_fn(circuit, solution, gate_set, n_layers)
                
        if task_type == TaskType.STATEPREP:
            sv = circuit.to_statevector_pennylane()()
            val = float(objective_fn(sv))
            objective_batch.append(val)
        
        elif task_type == TaskType.RECOVER:
            # Recover: Match gate types AND params of target structure
            candidate_sigs = []
            for layer in sorted(circuit.gate_table.keys()):
                for gate in circuit.gate_table[layer]:
                    candidate_sigs.append(get_gate_sig(gate))

            # Compare structure matches with continuous parameter scoring
            # Strictly matching rounded floats caused flat landscapes (0 gradients)
            total_score = 0.0
            n_gates = len(target_sigs)
            
            # Optimization_Circuit structure is fixed grid, so zip is aligned
            if n_gates > 0:
                for (t_type, t_val), (c_type, c_val) in zip(target_sigs, candidate_sigs):
                    if t_type != c_type:
                        continue # Mismatch type -> 0 points (Structure mismatch)

                    if t_val is None:
                        # Non-parameterized gate matched type (e.g. Identity, Hadamard)
                        total_score += 1.0
                    elif isinstance(t_val, (int, np.integer)):
                        # Discrete parameter (e.g. CNOT Target wire)
                        # Must match exactly (Discrete structure)
                        if t_val == c_val:
                            total_score += 1.0
                    else:
                        # Continuous parameter (Rotation Angle)
                        # Use smooth distance metric to provide gradients
                        diff = abs(t_val - c_val) % (2*np.pi)
                        diff = min(diff, 2*np.pi - diff)
                        # Linear decay over [0, pi] provides global gradient
                        # 1.0 match quality at 0 diff, 0.0 at pi diff
                        total_score += (1.0 - (diff / np.pi))

                score = total_score / n_gates
            else:
                score = 0.0
            
            objective_batch.append(score)

        elif task_type == TaskType.VQE:
            # 'objective' is treated as Hamiltonian itself (compute expectation value)
            # Since QD maximizes, we return -Energy.
            energy = float(circuit.eval(inputs=objective_fn))
            objective_batch.append(-energy)
        
        else:
            # CO (MaxCut etc)
            # In instance_set mode, we assume the circuit is independent of the graph instance
            # (parameterized state prep). We can run it once per solution.
            
            raw_samples = circuit.state()
            # Handle multiple shots (2D array) vs single shot (1D array)
            if isinstance(raw_samples, np.ndarray) and raw_samples.ndim == 2:
                samples = raw_samples
            else:
                samples = [raw_samples]
                
            all_sample_scores = []
            
            for sample in samples:
                sample_bitstring_z = to_z_basis(sample)
                
                scores = []
                for instance_graph in target_instances:
                    # instance_graph is either a networkx graph wrapper or just edges? 
                    # In qd(), nx_graph is created via nx_graph_from using edge_list.
                    # In instance_set mode, we should pass a list of such objects or edge lists.
                    # Let's assume target_instances is a list of objects with a .edges attribute or similar.
                    
                    # If instance_graph has .edges attribute (like NetworkX graph), use it.
                    # Use data='weight', default=1 to handle weighted/unweighted uniformly
                    if hasattr(instance_graph, "edges"):
                        # Calling edges(data=...) returns an iterator of (u, v, w)
                        edges = list(instance_graph.edges(data='weight', default=1.0))
                    else:  
                        edges = instance_graph # Fallback for list-of-edges input (mock objects)
                    
                    scores.append(float(objective_fn(sample_bitstring_z, edges)))
                
                # Mean score across all graphs (if generalization) for this sample
                all_sample_scores.append(np.mean(scores))
            
            # Aggregate: Mean over all samples (shots)
            objective_batch.append(np.mean(all_sample_scores))
        
        # Calculate measures using provided metrics
        measures = [metric.compute(solution, circuit) for metric in metrics]
        measures_batch.append(measures)

    return np.array(objective_batch), np.array(measures_batch)

def run_qd_experiment(
        problem_solution:float,
        problem_input,
        n_steps:int,
        n_layers:int,
        objective,
        gate_set:List[Gate],
        optimizer_type:OptimizerType,
        optimizer_params:QDParams,
        metrics:list[QualityMetric],
        task_type:TaskType,
        n_qubits:int,
        show_pbar:bool=True,
        shots:int|None=1,
    ):
    
    # experiment optimizer setup
    circuit = Optimization_Circuit(width=n_qubits, shots=shots)
    circuit_dims = n_qubits * n_layers
    metrics = [init_metric(metric, n_layers=n_layers, gate_set=gate_set, solution_dim=circuit_dims, n_wires=n_qubits) for metric in metrics]
    
    # Unpack the tuple of 5 items
    (
        scheduler, 
        optimization_archive, 
        result_archive, 
        prepare_circuit_fn, 
        actions_to_gates_fn
    ) = make_qd_optimizer(
        optimizer_type=optimizer_type, 
        solution_dim=circuit_dims, 
        metrics=metrics, 
        params=optimizer_params,
        gate_set=gate_set
    )

    # experiment data
    data = []
    solution_batches = []
    objective_batches = []
    measure_batches = []
    best_solution = -np.inf
    representative_solutions = []  # one genotype per step (e.g., best in that batch)
    

    # experiment loop
    pbar = trange(n_steps+1, file=sys.stdout, desc='Iterations') if show_pbar else range(n_steps+1)
    for itr in pbar:
        solution_batch = scheduler.ask()
        objective_batch, measure_batch = qd_batch(
            circuit, 
            solution_batch, 
            gate_set, 
            problem_input, 
            objective, 
            metrics=metrics,
            task_type=task_type,
            n_layers=n_layers,
            prepare_circuit_fn=prepare_circuit_fn,
            optimizer_params=optimizer_params
        )
        
        solution_batches.append(solution_batch)
        objective_batches.append(objective_batch)
        measure_batches.append(measure_batch)
        best_idx = int(np.argmax(objective_batch))
        representative_solutions.append(np.asarray(solution_batch[best_idx], dtype=float))
        
        scheduler.tell(objective_batch, measure_batch)
        current_best = max(objective_batch)
        if current_best > best_solution:
            best_solution = current_best
                        
        ar = best_solution/problem_solution if problem_solution != 0 else 0

        # Calculate averages for the current batch
        avg_batch_solution = np.mean(objective_batch)
        avg_batch_ar = avg_batch_solution / problem_solution if problem_solution != 0 else 0

        if show_pbar:
            pbar.set_description(f"[{itr}] AR:{ar:.3f} ({best_solution:.2f}/{problem_solution:.2f})")
        data.append([itr, best_solution, problem_solution, ar, np.round(result_archive.stats.coverage * 100 ,3), result_archive.stats.norm_qd_score, avg_batch_solution, avg_batch_ar])

    df = pd.DataFrame(data, columns=["step_id", "best_solution", "problem_solution", "approximation_ratio", "archive_coverage", "QD_score", "average_solution", "average_approximation_ratio"])
    
    # Pick a representative genotype: final best-so-far (best over all batches).
    all_objectives = np.concatenate([np.asarray(b, dtype=float) for b in objective_batches], axis=0)
    all_solutions = np.concatenate([np.asarray(b, dtype=float) for b in solution_batches], axis=0)
    best_global_idx = int(np.argmax(all_objectives))
    best_genotype = np.asarray(all_solutions[best_global_idx], dtype=float)

    metrics_results = compute_convergence_metrics(
        best_fit=df["best_solution"].to_numpy(),
        stability_signal=df["QD_score"].to_numpy(),  # non-monotone; good for drawdown/smoothness
        representative_solutions=representative_solutions,
        gate_set=gate_set,
        n_qubits=n_qubits,
        n_layers=n_layers,
        actions_to_gates_fn=actions_to_gates_fn,
        identity_gate=I,                       
    )

    for k, v in metrics_results.items():
        df[k] = v
    
    return df, optimization_archive, result_archive

def run_graph_experiment(
        graph_data: dict | List[dict],
        n_steps: int,
        n_layers: int,
        objective,
        gate_set: List[Gate],
        optimizer_type: OptimizerType,
        optimizer_params: QDParams,
        metrics: list[QualityMetric],
        task_type:TaskType,
        show_pbar: bool = True
    ):
    """
    Runs a QD experiment for one or multiple CO graph instance(s).
    """
    if task_type == TaskType.GRAPH_SINGLE:
        if not isinstance(graph_data, dict):
             raise ValueError("In 'GRAPH_SINGLE' mode, 'graph_data' must be a single graph dictionary.")
        
        problem_dim = graph_data.get("n_nodes")
        problem_solution = graph_data.get("solution").get("best_energy", 1.0)
        edge_list = graph_data.get("graph")
        problem_input = nx_graph_from(problem_dim, edge_list, optimal_value=problem_solution)
    
    elif task_type == TaskType.GRAPH_GENERAL:
        if not isinstance(graph_data, list):
             raise ValueError("In 'GRAPH_GENERAL' mode, 'graph_data' must be a list of graph dictionaries.")

        problem_dim = graph_data[0].get("n_nodes")
        try:
            # Calculate mean best energy for normalization
            problem_solution = np.mean([g.get("solution", {}).get("best_energy", 1.0) for g in graph_data])
        except:
            problem_solution = 1.0
        problem_input = [
            nx_graph_from(problem_dim, g["graph"], optimal_value=g.get("solution", {}).get("best_energy")) 
            for g in graph_data
        ]
    else:
        raise ValueError(f"Unsupported task_type {task_type} for graph experiment.")

    return run_qd_experiment(
        problem_solution=problem_solution,
        problem_input=problem_input,
        n_steps=n_steps,
        n_layers=n_layers,
        objective=objective,
        gate_set=gate_set,
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params,
        metrics=metrics,
        task_type=task_type,
        n_qubits=problem_dim,
        show_pbar=show_pbar,
        shots=1
    )

def run_sp_experiment(
        target_state: np.ndarray,
        n_qubits: int,
        n_steps: int,
        n_layers: int,
        objective, # partial function
        gate_set: List[Gate],
        optimizer_type: OptimizerType,
        optimizer_params: QDParams,
        metrics: list[QualityMetric],
        task_type: TaskType = TaskType.STATEPREP,
        show_pbar: bool = True
    ):
    """
    Runs a QD experiment for State Preparation (Fidelity maximization).
    """

    problem_solution = 1.0 #Max Fidelity
    
    dummy_input = {"n_nodes": n_qubits} 

    return run_qd_experiment(
        problem_solution=problem_solution,
        problem_input=dummy_input,
        n_steps=n_steps,
        n_layers=n_layers,
        objective=objective,
        gate_set=gate_set,
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params,
        metrics=metrics,
        task_type=task_type,
        n_qubits=n_qubits,
        show_pbar=show_pbar,
        shots=None
    )

def run_vqe_experiment(
        hamiltonian, # Pennylane Hamiltonian
        exact_energy: float,
        n_qubits: int,
        n_steps: int,
        n_layers: int,
        gate_set: List[Gate],
        optimizer_type: OptimizerType,
        optimizer_params: QDParams,
        metrics: list[QualityMetric],
        task_type: TaskType = TaskType.VQE,
        show_pbar: bool = True
    ):
    """
    Runs a QD experiment for VQE (Energy Minimization).
    """
    problem_solution = -exact_energy 

    dummy_input = {"n_nodes": n_qubits}

    return run_qd_experiment(
        problem_solution=problem_solution,
        problem_input=dummy_input,
        n_steps=n_steps,
        n_layers=n_layers,
        objective=hamiltonian,
        gate_set=gate_set,
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params,
        metrics=metrics,
        task_type=task_type,
        n_qubits=n_qubits,
        show_pbar=show_pbar,
        shots=None
    )

def run_rec_experiment(
        target_structure: np.ndarray,
        n_qubits: int,
        n_steps: int,
        n_layers: int,
        gate_set: List[Gate],
        optimizer_type: OptimizerType,
        optimizer_params: QDParams,
        metrics: list[QualityMetric],
        task_type: TaskType = TaskType.RECOVER,
        show_pbar: bool = True
    ):
    """
    Runs a QD experiment for Circuit Recovery (Structure Matching).
    """
    problem_solution = 1.0 # 100% match
    
    # problem_input is the target structure (solution vector)
    problem_input = target_structure

    return run_qd_experiment(
        problem_solution=problem_solution,
        problem_input=problem_input,
        n_steps=n_steps,
        n_layers=n_layers,
        objective=None, # Objective logic is hardcoded in qd_batch for RECOVER
        gate_set=gate_set,
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params,
        metrics=metrics,
        task_type=task_type,
        n_qubits=n_qubits,
        show_pbar=show_pbar,
        shots=None
    )
