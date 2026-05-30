import time
import warnings
from src import ABLATIONS_PATH
from src.runnables.run_helpers import run_rec_experiment
from src.utils import cd, seed_all, save_as_pickle
from src.utils import FULL
import numpy as np
from src.utils import TargetType, OptimizerType
from src.quality_metrics import SparsityQuality, EntanglementStructureQuality
from src.utils import QDParams, EXPParams

from dataclasses import replace 

def format_ablation_value(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)

def build_ablation_optimizer_list(
    optimizer_list: list[OptimizerType],
    name_suffix: str,
    **overrides
) -> list[OptimizerType]:
    ablated_list = []
    for optimizer_name, optimizer_type, optimizer_params, optimizer_metrics in optimizer_list:
        updated_params = replace(optimizer_params, **overrides)
        ablated_list.append(
            (
                f"{optimizer_name}_{name_suffix}",
                optimizer_type,
                updated_params,
                optimizer_metrics,
            )
        )
    return ablated_list

def run_rec_ablation(
    exp_params: EXPParams,
    target: tuple[str, np.ndarray],
    optimizer_list: list[OptimizerType],
    n_qubits: int,
    study_name: str,
    extra_columns: dict,
    save_archives: bool = False,
    run_id: int = None,
):
    target_name, target_structure = target
    shorthand = f"rec_{target_name}_{n_qubits}"

    for optimizer_name, optimizer_type, optimizer_params, optimizer_metrics in optimizer_list:
        exp_name = f"{optimizer_name}_{shorthand}"

        with cd(ABLATIONS_PATH / study_name / exp_name, assert_exists=False, remove=False):
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
                for key, value in extra_columns.items():
                    df[key] = value
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
    RUN_ABLATIONS = True
    ABLATION_POPULATION_STUDY = "ablation_pop"
    ABLATION_TAU_STUDY = "ablation_tau"
    ABLATION_DUAL_STRUCT_STUDY = "ablation_dual_struct"

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

    optimizer_registry = {
        "base": ("base", OptimizerType.BASE, base_params, DEFAULT_METRICS),
        "smooth": ("smooth", OptimizerType.SMOOTH, smooth_params, DEFAULT_METRICS),
        "smooth-det": ("smooth-det", OptimizerType.SMOOTH, smooth_det_params, DEFAULT_METRICS),
        "dual": ("dual", OptimizerType.DUAL, dual_params, DEFAULT_METRICS),
        "smooth-dual": ("smooth-dual", OptimizerType.SMOOTH_DUAL, smooth_dual_params, DEFAULT_METRICS),
        "smooth-dual-det": ("smooth-dual-det", OptimizerType.SMOOTH_DUAL, smooth_dual_det_params, DEFAULT_METRICS),
    }

    def select_optimizers(names: list[str]) -> list[OptimizerType]:
        selected = []
        for name in names:
            if name not in optimizer_registry:
                raise ValueError(f"Unknown optimizer name '{name}'. Available: {list(optimizer_registry.keys())}")
            selected.append(optimizer_registry[name])
        return selected

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

    if RUN_ABLATIONS:
        ABLATION_RUNS = 5
        batch_size_values = [10, 15, 20]
        num_emitters_values = [5, 10, 20]
        tau0_values = [0.1, 0.3, 0.5]
        tau_decay_values = [0.9, 0.95, 0.99]
        dual_struct_mut_prob_values = [0.5, 0.7, 0.9]
        dual_struct_gene_mut_prob_values = [0.1, 0.5, 0.25]

        population_optimizer_names = ["base", "smooth-dual-det"]
        tau_optimizer_names = ["smooth", "smooth-det", "smooth-dual", "smooth-dual-det"]
        dual_struct_optimizer_names = ["dual", "smooth-dual", "smooth-dual-det"]

        population_optimizer_list = select_optimizers(population_optimizer_names)
        tau_optimizer_list = select_optimizers(tau_optimizer_names)
        dual_struct_optimizer_list = select_optimizers(dual_struct_optimizer_names)

        DEFAULT_PARAMS = replace(DEFAULT_PARAMS, n_layers=5, n_steps=1500, n_runs=ABLATION_RUNS)
        for structure_name, structure_array in recovery_structures.items():
            for batch_size in batch_size_values:
                for num_emitters in num_emitters_values:
                    name_suffix = f"bs{batch_size}_ne{num_emitters}"
                    ablated_optimizer_list = build_ablation_optimizer_list(
                        population_optimizer_list,
                        name_suffix=name_suffix,
                        batch_size=batch_size,
                        num_emitters=num_emitters,
                    )
                    run_rec_ablation(
                        DEFAULT_PARAMS,
                        target=(structure_name, structure_array),
                        optimizer_list=ablated_optimizer_list,
                        n_qubits=5,
                        study_name=ABLATION_POPULATION_STUDY,
                        extra_columns={
                            "batch_size": batch_size,
                            "num_emitters": num_emitters,
                        },
                        save_archives=False,
                    )

            for tau0 in tau0_values:
                for tau_decay in tau_decay_values:
                    tau0_label = format_ablation_value(tau0)
                    tau_decay_label = format_ablation_value(tau_decay)
                    name_suffix = f"tau{tau0_label}_td{tau_decay_label}"
                    ablated_optimizer_list = build_ablation_optimizer_list(
                        tau_optimizer_list,
                        name_suffix=name_suffix,
                        tau0=tau0,
                        tau_decay=tau_decay,
                    )
                    run_rec_ablation(
                        DEFAULT_PARAMS,
                        target=(structure_name, structure_array),
                        optimizer_list=ablated_optimizer_list,
                        n_qubits=5,
                        study_name=ABLATION_TAU_STUDY,
                        extra_columns={
                            "tau0": tau0,
                            "tau_decay": tau_decay,
                        },
                        save_archives=False,
                    )

            for dual_struct_mut_prob in dual_struct_mut_prob_values:
                for dual_struct_gene_mut_prob in dual_struct_gene_mut_prob_values:
                    mut_label = format_ablation_value(dual_struct_mut_prob)
                    gene_label = format_ablation_value(dual_struct_gene_mut_prob)
                    name_suffix = f"dmut{mut_label}_dgene{gene_label}"
                    ablated_optimizer_list = build_ablation_optimizer_list(
                        dual_struct_optimizer_list,
                        name_suffix=name_suffix,
                        dual_struct_mut_prob=dual_struct_mut_prob,
                        dual_struct_gene_mut_prob=dual_struct_gene_mut_prob,
                    )
                    run_rec_ablation(
                        DEFAULT_PARAMS,
                        target=(structure_name, structure_array),
                        optimizer_list=ablated_optimizer_list,
                        n_qubits=5,
                        study_name=ABLATION_DUAL_STRUCT_STUDY,
                        extra_columns={
                            "dual_struct_mut_prob": dual_struct_mut_prob,
                            "dual_struct_gene_mut_prob": dual_struct_gene_mut_prob,
                        },
                        save_archives=False,
                    )