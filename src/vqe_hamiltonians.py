import pennylane as qml
import numpy as np

def get_ising_chain_hamiltonian(n_qubits, J=1.0, h=1.0):
    """
    Constructs the Transverse Field Ising Model Hamiltonian.
    H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)
    periodic boundary conditions: Z_{n-1} Z_0 term included.
    """
    coeffs = []
    obs = []
    
    # Interaction terms Z_i Z_{i+1}
    for i in range(n_qubits):
        coeffs.append(-J)
        obs.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits))
        
    # Transverse field terms X_i
    for i in range(n_qubits):
        coeffs.append(-h)
        obs.append(qml.PauliX(i))
        
    return qml.Hamiltonian(coeffs, obs)

def get_vqe_hamiltonian(name, n_qubits, **kwargs):
    """
    Factory method for VQE Hamiltonians.
    """
    if name == "ising_chain":
        return get_ising_chain_hamiltonian(n_qubits, **kwargs)
    else:
        raise ValueError(f"Unknown VQE Hamiltonian: {name}")

def compute_ising_energy_exact(n_qubits, J=1.0, h=1.0):
    """
    Computes the exact ground state energy of the Transverse Field Ising Model
    using numpy.linalg.eigh (diagonalization).
    Limit n_qubits <= 12 for speed.
    """
    if n_qubits > 12:
        print("Warning: n_qubits large for exact diagonalization.")
        
    # Construct matrix representation
    # Z = [[1, 0], [0, -1]], X = [[0, 1], [1, 0]]
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    id2 = np.eye(2)
    
    def tensor_product_list(op_list):
        res = op_list[0]
        for op in op_list[1:]:
            res = np.kron(res, op)
        return res

    H_mat = np.zeros((2**n_qubits, 2**n_qubits))
    
    # Interaction terms -J * Z_i Z_{i+1}
    for i in range(n_qubits):
        op_list = [id2] * n_qubits
        op_list[i] = sz
        op_list[(i + 1) % n_qubits] = sz
        H_mat += -J * tensor_product_list(op_list)
        
    # Transverse field terms -h * X_i
    for i in range(n_qubits):
        op_list = [id2] * n_qubits
        op_list[i] = sx
        H_mat += -h * tensor_product_list(op_list)
        
    eigenvalues = np.linalg.eigvalsh(H_mat)
    return np.min(eigenvalues)
