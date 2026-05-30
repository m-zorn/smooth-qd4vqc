from typing import Tuple, List, Union
import os, shutil, pickle
from pathlib import Path
import itertools
import random
from enum import Enum
from dataclasses import dataclass
import numpy as np
import networkx as nx 

from src.circuit import Gate, RX, RZ, RY, H, I, S, T, CNOT, X, H

FULL = [I, RX, RY, RZ, H, S, T, CNOT]
CLIFFORDT = [I, CNOT, H, S, T]
ROTHCNOT = [I, RX, RY, RZ, H, CNOT]
COMMON = [I, RX, RY, RZ, S, CNOT]
SMALL = [I, RX, RZ, H, CNOT]
TINYH = [I, RX, H, CNOT]
TINY = [I, RX, CNOT]
IRX = [I, RX]

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

@dataclass
class EXPParams:
    n_steps: int
    n_layers:int
    n_runs:int
    gate_set:list
    run_parallel: bool
    parallel_workers: int

@dataclass
class QDParams:
    lr: float = 1.0
    batch_size: int = 2
    num_emitters: int = 20
    s0: float = 0.5
    x0: np.ndarray = None
    rankers: str = "imp"
    selection_rule: str = "mu"
    restart_rule: str = "basic"
    tau0: float = 0.75
    tau_min: float = 0.10
    tau_decay: float = 0.99
    smooth_deterministic: bool = False
    smooth_crn_seed: int = 42
    hybrid: bool = False
    hybrid_method: str = "COBYLA"
    dual_struct_mut_prob: float = 0.9
    dual_struct_gene_mut_prob: float = 0.1
    dual_struct_from_archive_prob: float = 0.4
    dual_struct_pool_max: int = 500
    dual_p_identity_init: float = 0.5

class OptimizerType(Enum):
    BASE = "base"
    SMOOTH = "smooth"
    DUAL = "dual"
    SMOOTH_DUAL = "smooth_dual"

class TargetType(Enum):
    GHZ = "ghz"
    DICKE = "dicke"
    RANDOM = "random"

class TaskType(Enum):
    GRAPH_SINGLE = "graph_single"
    GRAPH_GENERAL = "graph_general"
    STATEPREP= "stateprep"
    VQE = "vqe"
    RECOVER= "recover"

def solve_brute_force(n_vertices, edges, cost_function):
    max_val = -float('inf')
    for bits in itertools.product([-1, 1], repeat=n_vertices):
         val = cost_function(bits, edges)
         if val > max_val: max_val = val
    return max_val

def get_sp_target_name(target_type:TargetType, n_qubits:int) -> str:
    if target_type == TargetType.DICKE:
            k = n_qubits // 2
            target_name = f"dicke_{k}"
    elif target_type == TargetType.RANDOM:
        target_name = "random"
    elif target_type == TargetType.GHZ:
        target_name = "ghz"
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    return target_name

def nx_graph_from(n_vertices:int, edge_list:List[Tuple[int, int]], complement:bool=False, optimal_value:float=None) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    
    # Ensure edge_list is subscriptable (handles EdgeView, iterators, etc.)
    if not isinstance(edge_list, list):
        edge_list = list(edge_list)
        
    if len(edge_list) > 0 and len(edge_list[0]) == 3:
        G.add_weighted_edges_from(edge_list)
    else:
        G.add_edges_from(edge_list)    
    
    G = nx.complement(G) if complement else G
    
    if optimal_value is not None:
        G.graph["optimal_value"] = optimal_value
        
    return G

def to_z_basis(binary_solution:Union[str, List[int]]) -> List[int]:
    if isinstance(binary_solution, str):
        binary_solution = [int(i) for i in list(binary_solution)]
    return [-1 if c == 1 else 1 for c in binary_solution]

def save_as_pickle(obj, file_name):
    with open(f'{file_name}.pickle', 'wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    with open(f'{file_name}.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b

def seed_all(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def normalize(arr:np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / (arr.std() + 1e-10)

# from https://stackoverflow.com/a/13197763
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath:str, assert_exists:bool=True, remove:bool=False):
        self.newPath = Path(newPath)
        if assert_exists:
            assert not self.newPath.exists(), f'Path "{self.newPath}" already exists. Check your configuration!'
        elif remove and self.newPath.exists() and self.newPath.is_dir():
            print(f"\nRemoving old path-dir '{self.newPath}'.\n")
            shutil.rmtree(self.newPath)
        self.newPath.mkdir(exist_ok=True, parents=True)
        print(f"\nStarting new experiment in '{self.newPath}'!\n")

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def map_to_choice(value, n_choices):
    """ Map a floating-point number in the range [0, 1] to one of the n choices"""
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1")
    if n_choices <= 0:
        raise ValueError("Number of choices must be positive")
    # Determine the width of each segment based on the number of choices
    segment_width = 1.0 / n_choices
    # Determine which segment the value falls into (min function for edge case "value is 1")
    choice_index = min(int(value / segment_width), n_choices - 1)
    return choice_index

remap_to_2pi = lambda x: x * 2 * np.pi

def actions_to_gates(actions:List[float], gate_set:List[Gate]) -> dict[int, Gate]:
    ''' Returns actions converted to gates dep. on gate set as list of (wire, Gate) tuples '''
    converted = {}
    gate_choices = []
    gate_params = []
    for wire, action in enumerate(actions):
        # split number and decimel (- 2.14455 -> -2 & 14455), round to n decimal for fixed second part
        number, decimal = str(round(action, 4)).split(".")
        # divide decimel (second part) by length-1 s.t. it becomes float (1.4455)
        decimal = int(decimal) / (10**(len(decimal)))
        # take the sign of the input and only adjust second part (- 2.14455 -> 2 & -1.4455)
        number, decimal = abs(int(number)), decimal
        
        try:
            gate = gate_set[number]
        except IndexError:
            print(f"action: {action} | Number: {number}, decimal: {decimal}")
        if gate == CNOT:
            # if CNOT then decimal is the control target; For this we normalized to [0,1] (X.YYY -> 0.XYYY) and map to divided
            # segmentation range e.g.n=5: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0] => map_to_range(0.85,5)=4
            target = map_to_choice(abs(decimal), len(actions))
            if target == wire:
                gate = X
            else:
                gate = CNOT(wire, target)
        elif gate in [RX, RY, RZ]:
            # if is any other parameterized gate simple use decimal as the rotation angle
            # decimal = np.sign(action) * round(decimal%(2*np.pi), 3)
            decimal = np.sign(action) * remap_to_2pi(decimal)
            gate = gate(decimal)
        else:
            # any of the other unique gates (S,H,I) simply add on wire
            pass
        converted[wire] = gate
        gate_choices.append(number)
        gate_params.append(decimal)
    return converted, np.array(gate_choices), np.array(gate_params)