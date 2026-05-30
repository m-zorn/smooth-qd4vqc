import pandas as pd
import time
import warnings
from src import DATA_PATH, TESTCR_PATH
from src.runnables.run_helpers import run_graph_experiment, run_sp_experiment, run_vqe_experiment, run_rec_experiment
from src.cost_hamiltonians import max_clique_hamiltonian, max_cut_hamiltonian, max_independed_set_hamiltonian, min_vertex_cover_hamiltonian, weighted_max_cut_hamiltonian
from src.utils import cd, seed_all, load_pickle, save_as_pickle
from src.utils import FULL
from src.state_prep_targets import get_target_state, state_prep_objective
from functools import partial
import numpy as np
from src.vqe_hamiltonians import get_vqe_hamiltonian, compute_ising_energy_exact
from src.utils import TargetType, OptimizerType, TaskType, get_sp_target_name
from src.quality_metrics import SparsityQuality, EntanglementStructureQuality
from src.utils import QDParams, EXPParams
from tqdm import tqdm
from dataclasses import replace 

def run_sp(exp_params: EXPParams, qubit_list: list[int], target_type:TargetType, optimizer_list:list[OptimizerType], save_archives:bool=False, run_id: int=None):
    ''' Minimal viable state preparation (sp) benchmark: GHZ(n), Dicke(n, n/2), random-state(n) for n in qubit list'''
        
    assert exp_params.n_steps > 0, "Number of steps must be positive"
    assert qubit_list, "Please provide a list of qubit counts for SP experiments."
    assert optimizer_list is not None, "Please provide an optimizer list for CO experiments."

    for n_qubits in qubit_list:
        target_name = get_sp_target_name(target_type, n_qubits)
        
        for optimizer_name, optimizer_type, optimizer_params, optimizer_metrics in optimizer_list:
            exp_name = f"{optimizer_name}_sp_{n_qubits}_{target_name}"
            
            with cd(TESTCR_PATH / exp_name, assert_exists=False, remove=False):
                start_time = time.time()
                runs_to_execute = [run_id] if run_id is not None else range(exp_params.n_runs)
                for current_run_id in runs_to_execute:
                    seed_all(current_run_id)
                    run_time = time.time()
                    target_state_vector = get_target_state(target_name, n_qubits)
                    objective_func = partial(state_prep_objective, target_state_vector=target_state_vector)
                    df, o_archive, r_archive = run_sp_experiment(
                        target_state=target_state_vector,
                        n_qubits=n_qubits,
                        n_steps=exp_params.n_steps,
                        n_layers=exp_params.n_layers,
                        objective=objective_func,
                        gate_set=exp_params.gate_set,
                        optimizer_type=optimizer_type,
                        optimizer_params=optimizer_params,
                        metrics=optimizer_metrics,
                        show_pbar=exp_params.run_parallel==False
                    )
                    df["name"] = f"{optimizer_name}"
                    df["problem"] = target_name
                    df["seed"] = current_run_id
                    df["n_qubits"] = n_qubits
                    df["run_time_sec"] = time.time() - run_time
                    df.to_csv(f"{current_run_id}.csv", index=False)
                    if save_archives:
                        save_as_pickle({
                            "objective_archive": o_archive,
                            "result_archive": r_archive
                        }, f"{current_run_id}_archives")

                print(f"*** Finished run for {exp_name} (took {(time.time() - start_time) / 60:.0f} minutes total)")

def run_co(exp_params: EXPParams, task_type: TaskType, vertices: list[int], problem_list:list[tuple], optimizer_list:list[OptimizerType], debug:bool=False, save_archives:bool=False):
    
    assert task_type in [TaskType.GRAPH_SINGLE, TaskType.GRAPH_GENERAL], "task_type must be 'GRAPH_SINGLE' or 'GRAPH_GENERAL'"
    assert exp_params.n_steps > 0, "Number of steps must be positive"
    assert vertices is not None, "Please provide a list of graph-vertice sizes for CO experiments."
    assert problem_list is not None, "Please provide a problem list for CO experiments."
    assert optimizer_list is not None, "Please provide an optimizer list for CO experiments."

    for vertex_count in vertices:
        for shorthand, dataset_name, hamiltonian_cost_objective in problem_list:
            for optimizer_name, optimizer_type, optimizer_params, optimizer_metrics in optimizer_list:
                exp_name = f"{optimizer_name}_{vertex_count}x50_{shorthand}_co"
                if debug:
                    all_graphs = load_pickle(f"{DATA_PATH}/erdos_reny_v{vertex_count}x50_{dataset_name}")
                    graphs = all_graphs[:5]
                    print(f'DEBUG MODE: Loading smaller graph set -> first {len(graphs)}/{len(all_graphs)} graphs')
                else:
                    graphs = load_pickle(f"{DATA_PATH}/erdos_reny_v{vertex_count}x50_{dataset_name}")
                                        
                with cd(TESTCR_PATH / exp_name, assert_exists=False, remove=True):
                    dfs = []
                    start_time = time.time()
                    for graph_id, graph in enumerate(graphs): 
                        seed_all(graph_id)
                        df, o_archive, r_archive = run_graph_experiment(
                            graph_data=graph,
                            n_steps=exp_params.n_steps,
                            n_layers=exp_params.n_layers,
                            objective=hamiltonian_cost_objective,
                            gate_set=exp_params.gate_set,
                            optimizer_type=optimizer_type,
                            optimizer_params=optimizer_params,
                            metrics=optimizer_metrics,
                            task_type=TaskType.GRAPH_SINGLE,
                            show_pbar=exp_params.run_parallel==False
                        )
                        df["name"] = f"{optimizer_name}"
                        df["problem"] = shorthand
                        df["eval_mode"] = "single"
                        df["graph"] = graph.get("name")
                        df["seed"] = graph_id
                        dfs.append(df)
                        pd.concat(dfs, ignore_index=True).to_csv(f"all.csv", index=False)
                        if save_archives:
                            save_as_pickle({
                                "objective_archive": o_archive,
                                "result_archive": r_archive
                            }, f"{exp_name}_archives_{graph_id}")

                    save_path = "all.csv"
                    pd.concat(dfs, ignore_index=True).to_csv(save_path, index=False)
                    print(f"*** Finished run for {exp_name} (took {(time.time() - start_time) / 60:.0f} minutes total)")

def run_vqe(exp_params: EXPParams, circuit_widths: list[int], optimizer_list:list[OptimizerType], save_archives:bool=False, run_id: int=None):    
    ''' Small Ising Chain as VQE Benchmark'''
    n_qubits_list = circuit_widths
    
    for n_qubits in n_qubits_list:
        target_name = "ising_chain"
        shorthand = f"vqe_{target_name}_{n_qubits}"

        # Construct Hamiltonian and compute exact ground state energy for reference
        hamiltonian = get_vqe_hamiltonian(target_name, n_qubits)
        exact_energy = compute_ising_energy_exact(n_qubits)
        #print(f"Target: {target_name} (n={n_qubits}) | Exact Energy: {exact_energy:.6f}")
            
        for optimizer_name, optimizer_type, optimizer_params, optimizer_metrics in optimizer_list:
            exp_name = f"{optimizer_name}_{shorthand}"
            
            with cd(TESTCR_PATH / exp_name, assert_exists=False, remove=False):
                start_time = time.time()
                runs_to_execute = [run_id] if run_id is not None else range(exp_params.n_runs)
                for current_run_id in runs_to_execute:
                    seed_all(current_run_id)
                    run_time = time.time()
                    df, o_archive, r_archive = run_vqe_experiment(
                        hamiltonian=hamiltonian,
                        exact_energy=exact_energy,
                        n_qubits=n_qubits,
                        n_steps=exp_params.n_steps,
                        n_layers=exp_params.n_layers,
                        gate_set=exp_params.gate_set,
                        optimizer_type=optimizer_type,
                        optimizer_params=optimizer_params,
                        metrics=optimizer_metrics,
                        show_pbar=exp_params.run_parallel==False
                    )
                    df["name"] = f"{optimizer_name}"
                    df["problem"] = target_name
                    df["seed"] = current_run_id
                    df["n_qubits"] = n_qubits
                    df["exact_energy"] = exact_energy
                    df["run_time_sec"] = time.time() - run_time
                    df.to_csv(f"{current_run_id}.csv", index=False)
                    if save_archives:
                        save_as_pickle({
                            "objective_archive": o_archive,
                            "result_archive": r_archive
                        }, f"{current_run_id}_archives")

                print(f"*** Finished run for {exp_name} (took {(time.time() - start_time) / 60:.0f} minutes total)")

def run_rec(exp_params: EXPParams, target: tuple[str, np.ndarray], optimizer_list:list[OptimizerType], n_qubits:int, save_archives:bool=False, run_id: int=None):
    ''' Circuit Recovery Benchmark '''
    
    target_name, target_structure = target
    shorthand = f"rec_{target_name}_{n_qubits}"

    for optimizer_name, optimizer_type, optimizer_params, optimizer_metrics in optimizer_list:
        exp_name = f"{optimizer_name}_{shorthand}"
        
        with cd(TESTCR_PATH / exp_name, assert_exists=False, remove=False):
            start_time = time.time()
            runs_to_execute = [run_id] if run_id is not None else range(exp_params.n_runs)
            for current_run_id in runs_to_execute:
                seed_all(current_run_id)
                run_time = time.time()
                df, o_archive, r_archive = run_rec_experiment(
                    target_structure=target_structure,
                    n_qubits=n_qubits,
                    n_steps=exp_params.n_steps,
                    n_layers=exp_params.n_layers,
                    gate_set=exp_params.gate_set,
                    optimizer_type=optimizer_type,
                    optimizer_params=optimizer_params,
                    metrics=optimizer_metrics,
                    show_pbar=exp_params.run_parallel==False
                )
                df["name"] = f"{optimizer_name}"
                df["problem"] = shorthand
                df["seed"] = current_run_id
                df["n_qubits"] = n_qubits
                df["run_time_sec"] = time.time() - run_time
                df.to_csv(f"{current_run_id}.csv", index=False)
                if save_archives:
                    save_as_pickle({
                        "objective_archive": o_archive,
                        "result_archive": r_archive
                    }, f"{current_run_id}_archives")

            print(f"*** Finished run for {exp_name} (took {(time.time() - start_time) / 60:.0f} minutes total)")

def configure_warnings():
    # Suppress CMA-ES numerical warnings
    warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*")
    warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*")
    warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*")

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    # Apply warnings to main process
    configure_warnings()
    
    DEBUG = False
    DEFAULT_PARAMS = EXPParams(
        n_steps=100,
        n_layers=10,
        n_runs=10,
        gate_set=FULL,
        run_parallel=False,
        parallel_workers=30
    )
    DEFAULT_METRICS = [
        SparsityQuality,
        EntanglementStructureQuality
    ]

    base_params = QDParams(
        lr=1.0,
        batch_size=15,
        num_emitters=10,
        s0=1.0,
        selection_rule="mu",
        rankers="2obj",
        restart_rule="no_improvement"
    )

    smooth_params = QDParams(
        lr=1.0,
        batch_size=15,
        num_emitters=10,
        s0=1.0,
        selection_rule="mu",
        rankers="2obj",
        restart_rule="no_improvement",
        tau0=0.3,
        tau_min=0.01,
        tau_decay=0.95,
        smooth_deterministic=False
    )

    smooth_det_params = QDParams(
        lr=1.0,
        batch_size=15,
        num_emitters=10,
        s0=1.0,
        selection_rule="mu",
        rankers="2obj",
        restart_rule="no_improvement",
        tau0=0.3,
        tau_min=0.01,
        tau_decay=0.95,
        smooth_deterministic=True
    )
    
    dual_params = QDParams(
        lr=1.0,
        batch_size=15,
        num_emitters=10,
        s0=0.02,
        selection_rule="mu",
        rankers="2obj",
        restart_rule="no_improvement",
        dual_struct_mut_prob=0.9,     
        dual_struct_gene_mut_prob=0.25,
        dual_struct_from_archive_prob=0.4, 
        dual_struct_pool_max=500,          
        dual_p_identity_init=1.0           
    )

    smooth_dual_params = QDParams(
        lr=1.0,
        batch_size=15,
        num_emitters=10,
        s0=0.05,
        selection_rule="mu",
        rankers="2obj",
        restart_rule="no_improvement",
        tau0=0.3,
        tau_min=0.01,
        tau_decay=0.95,
        smooth_deterministic=False,
        dual_struct_mut_prob=0.9,
        dual_struct_gene_mut_prob=0.25,
        dual_struct_from_archive_prob=0.4, 
        dual_struct_pool_max=500,          
        dual_p_identity_init=1.0 
    )

    smooth_dual_det_params = QDParams(
        lr=1.0,
        batch_size=15,
        num_emitters=10,
        s0=0.05,
        selection_rule="mu",
        rankers="2obj",
        restart_rule="no_improvement",
        tau0=0.3,
        tau_min=0.01,
        tau_decay=0.95,
        smooth_deterministic=True,
        dual_struct_mut_prob=0.9,
        dual_struct_gene_mut_prob=0.25,
        dual_struct_from_archive_prob=0.4, 
        dual_struct_pool_max=500,          
        dual_p_identity_init=1.0 
    )

    #optimizer_name, optimizer_type, optimizer_params, optimizer_metrics
    optimizer_list = [
        (f"base", OptimizerType.BASE, base_params, DEFAULT_METRICS),
        (f"smooth", OptimizerType.SMOOTH, smooth_params, DEFAULT_METRICS),
        (f"smooth-det", OptimizerType.SMOOTH, smooth_det_params, DEFAULT_METRICS),
        (f"dual", OptimizerType.DUAL, dual_params, DEFAULT_METRICS),
        (f"smooth-dual", OptimizerType.SMOOTH_DUAL, smooth_dual_params, DEFAULT_METRICS),
        (f"smooth-dual-det", OptimizerType.SMOOTH_DUAL, smooth_dual_det_params, DEFAULT_METRICS),
    ]

    # shorthand, dataset_name, hamiltonian_cost_objective
    co_problem_list = [
        ('maxCUT', 'maxcut', max_cut_hamiltonian),
        ('weighted_maxCUT', 'maxcut_weighted', weighted_max_cut_hamiltonian),
        #('maxIND', 'maxindependentset', max_independed_set_hamiltonian),
        #('maxCLI', 'maxclique', max_clique_hamiltonian),
        #('minVER', 'minvertexcover', min_vertex_cover_hamiltonian),
    ]
    co_problem_graphsize_list = [
        14,
    ]
    target_type_list = [
        #TargetType.GHZ,
        TargetType.DICKE, 
        TargetType.RANDOM
    ]
    target_qubit_list = [
        6,
    ]
    vqe_qubit_list = [
        6,
    ]
    ENTANGLING_STRUCTURE = np.array([
        1, 2, 3, 7.2, 0,
        1, 2, 3, 7.4, 0,
        1, 2, 3, 7.7, 0,
        1, 2, 3, 7.85, 0,
        1, 2, 3, 7.0, 0,
    ])
    ALTERNATING_STRUCTURE = np.array([
        4, 7.2, 1, 7.2, 2,
        4, 7.4, 1, 7.4, 2,
        4, 7.7, 1, 7.7, 2,
        4, 7.85, 1, 7.85, 2,
        4, 7.0, 1, 7.0, 2
    ])

    recovery_structures = {
        "entangling_structure": ENTANGLING_STRUCTURE,
        "alternating_structure": ALTERNATING_STRUCTURE,
    }

    if DEFAULT_PARAMS.run_parallel:
        tasks = []
                
        # # Queue SP Experiments
        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=10, n_steps=1000)
        for qubit in target_qubit_list:
            for target_type in target_type_list:
                for opt in optimizer_list:
                    for _run_id in range(DEFAULT_PARAMS.n_runs):
                        tasks.append((run_sp, (DEFAULT_PARAMS, [qubit], target_type, [opt], False, _run_id)))
        
        # Queue CO Experiments
        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=10, n_steps=100)
        for vertex_count in co_problem_graphsize_list:
            for prob in co_problem_list:
                for opt in optimizer_list:
                    tasks.append((run_co, (DEFAULT_PARAMS, TaskType.GRAPH_SINGLE, [vertex_count], [prob], [opt], DEBUG)))

        # Queue VQE Experiments
        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=6, n_steps=1000)
        for qubit in vqe_qubit_list:
            for opt in optimizer_list:
                tasks.append((run_vqe, (DEFAULT_PARAMS, [qubit], [opt])))
        
        # Queue REC Experiments
        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=5, n_steps=1500)
        for opt in optimizer_list:
            for structure_name, structure_array in recovery_structures.items():
                tasks.append((run_rec, (DEFAULT_PARAMS, (structure_name, structure_array), [opt], 5)))

        print(f"Running {len(tasks)} experiments with {DEFAULT_PARAMS.parallel_workers} workers...")
        with tqdm(total=len(tasks)) as pbar:
            with ProcessPoolExecutor(max_workers=DEFAULT_PARAMS.parallel_workers, initializer=configure_warnings) as executor:
                futures = [executor.submit(func, *args) for func, args in tasks]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Experimental run failed: {e}")
                    finally:
                        pbar.update(1)
    else:
        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=10, n_steps=1000)
        for target_type in [TargetType.RANDOM, TargetType.DICKE]: #TargetType.GHZ, TargetType.DICKE, 
            run_sp(DEFAULT_PARAMS, qubit_list=target_qubit_list, target_type=target_type, optimizer_list=optimizer_list)
        
        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=10, n_steps=100)
        run_co(DEFAULT_PARAMS, task_type=TaskType.GRAPH_SINGLE, vertices=co_problem_graphsize_list, problem_list=co_problem_list, optimizer_list=optimizer_list, debug=DEBUG)

        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=6, n_steps=1000)
        run_vqe(DEFAULT_PARAMS, circuit_widths=vqe_qubit_list, optimizer_list=optimizer_list, save_archives=True)

        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=5, n_steps=1500)
        for structure_name, structure_array in recovery_structures.items():
            run_rec(DEFAULT_PARAMS, target=(structure_name, structure_array), optimizer_list=optimizer_list, n_qubits=5, save_archives=True)