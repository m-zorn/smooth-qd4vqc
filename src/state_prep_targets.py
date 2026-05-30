import numpy as np
from itertools import combinations

# Fidelity definition: |<psi|target>|^2 for pure states
def fidelity(state_vector, target_state_vector):
    # Ensure numpy arrays
    state_vector = np.array(state_vector)
    target_state_vector = np.array(target_state_vector)
    
    # Compute overlap
    overlap = np.dot(np.conj(state_vector), target_state_vector)
    return np.abs(overlap)**2

def get_ghz_state(n_qubits):
    """
    Returns the statevector for the GHZ state: (|0...0> + |1...1>) / sqrt(2)
    """
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    state[dim-1] = 1.0
    return state / np.sqrt(2)

def get_dicke_state(n_qubits, k):
    """
    Returns the statevector for the Dicke state |D_{n,k}>:
    Equal superposition of all basis states with Hamming weight k.
    """
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    
    # Iterate over all combinations of k positions for 1s
    for positions in combinations(range(n_qubits), k):
        # Create basis state index
        index = 0
        for pos in positions:
            index |= (1 << (n_qubits - 1 - pos))
        state[index] = 1.0
        
    # Normalize
    norm = np.linalg.norm(state)
    return state / norm

def get_random_state(n_qubits: int, seed: int = None, sparsity: int | None = None):
    """
    Generates a reproducible random pure quantum state.
    If sparsity is None, returns a Haar-random state in the full Hilbert space.
    If sparsity = k, returns a random k-sparse state in the computational basis:
    a support of size k is chosen uniformly at random, amplitudes on that support
    are sampled from a complex Gaussian, and the state is normalized.
    """
    if seed is None:
        seed = 42 + n_qubits # Deterministic default based on size
        
    rng = np.random.default_rng(seed)
    
    # Generate random complex vector
    real_part = rng.normal(0, 1, 2**n_qubits)
    imag_part = rng.normal(0, 1, 2**n_qubits)
    state = real_part + 1j * imag_part
    
    # Create a mask to set all but 'sparsity' number of entries to zero
    if sparsity is not None and sparsity > 0:
        assert 0 < sparsity <= 2**n_qubits, "Sparsity cannot exceed the dimension of the state space"
        mask = np.zeros(2**n_qubits, dtype=bool)
        mask[rng.choice(2**n_qubits, sparsity, replace=False)] = True
        state = state * mask

    # Normalize
    state = state / np.linalg.norm(state)
    assert np.isclose(np.sum(np.abs(state)**2), 1.0), "State vector is not properly normalized"
    return state

def get_target_state(target_name: str, n_qubits: int, seed: int = None):
    if target_name == "ghz":
        return get_ghz_state(n_qubits)
    elif target_name.startswith("dicke"):
        # Extract k from name if present, e.g. "dicke_2"
        try:
            parts = target_name.split("_")
            if len(parts) > 1:
                k = int(parts[1])
            else:
                k = n_qubits // 2
        except:
             k = n_qubits // 2
        return get_dicke_state(n_qubits, k)
    elif target_name == "random":
        # Returns a fixed random target for this n_qubits
        return get_random_state(n_qubits, seed)
    else:
        raise ValueError(f"Unknown target state name: {target_name}")

def state_prep_objective(state_vector, target_state_vector):
    """
    Returns the fidelity.
    QD expects to maximize the objective, and fidelity is [0, 1].
    So we can return fidelity directly.
    """
    return fidelity(state_vector, target_state_vector)    