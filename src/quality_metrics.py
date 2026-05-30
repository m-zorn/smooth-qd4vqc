from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from src.circuit import CNOT

def init_metric(metrics_class, **kwargs):
    if metrics_class == SparsityQuality:
        return SparsityQuality(kwargs['solution_dim'])
    elif metrics_class == UniformityQuality:
        return UniformityQuality(kwargs['n_layers'], len(kwargs['gate_set']), kwargs['n_wires'])
    elif metrics_class == EntanglementStructureQuality:
        return EntanglementStructureQuality(kwargs['n_wires'])
    else:
        raise ValueError(f"Unknown metric class: {metrics_class}")

class QualityMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass

    @property
    @abstractmethod
    def range(self) -> Tuple[float, float]:
        """Value range (min, max) for the metric."""
        pass

    @abstractmethod
    def compute(self, solution: np.ndarray, circuit) -> float:
        """Computes the metric for a given solution and circuit."""
        pass

class SparsityQuality(QualityMetric):
    short = "SP"
    def __init__(self, solution_dim: int):
        self._solution_dim = solution_dim

    @property
    def name(self) -> str:
        return "sparsity"

    @property
    def range(self) -> Tuple[float, float]:
        return (0.0, float(self._solution_dim))

    def compute(self, solution: np.ndarray, circuit) -> float:
        # Count parameters with absolute value < 1
        return float(len(solution[np.abs(solution) < 1]))

class UniformityQuality(QualityMetric):
    short = "UN"
    def __init__(self, n_layers: int, gate_set_size: int, n_wires: int):
        self._n_layers = n_layers
        self._gate_set_size = gate_set_size
        self._n_wires = n_wires

    @property
    def name(self) -> str:
        return "uniformity"

    @property
    def range(self) -> Tuple[float, float]:
        # Max uniformity per layer is when every wire has a different gate, 
        # but bounded by the number of available distinct gates (excluding 0).
        max_score_per_layer = min(self._n_wires, self._gate_set_size - 1)
        return (0.0, float(self._n_layers * max_score_per_layer))

    def compute(self, solution: np.ndarray, circuit) -> float:
        # circuit.get_matrix_circuit() expected shape: (width, depth)
        matrix = circuit.get_matrix_circuit()
        
        # Round and cast to int to get gate indices
        matrix_int = matrix.round().astype(int)
        
        score = 0
        # Iterate over layers (columns in (width, depth) matrix, so rows in transpose)
        # matrix_int.T shape: (depth, width)
        for layer in matrix_int.T:
            # Count unique non-zero gates in this layer
            unique_gates = np.unique(layer[layer != 0])
            score += len(unique_gates)
            
        return float(score)

class EntanglementStructureQuality(QualityMetric):
    short = "ES"
    def __init__(self, n_wires: int):
        self._n_wires = n_wires

    @property
    def name(self) -> str:
        return "entanglement_structure"

    @property
    def range(self) -> Tuple[float, float]:
        # Maximum number of unique directed connections is n_wires * (n_wires - 1)
        return (0.0, float(self._n_wires * (self._n_wires - 1)))

    def compute(self, solution: np.ndarray, circuit) -> float:
        unique_connections = set()
        for layer_idx, gates in circuit.gate_table.items():
            for gate in gates:
                if isinstance(gate, CNOT):
                    unique_connections.add((gate.ctr, gate.target))
        return float(len(unique_connections))
