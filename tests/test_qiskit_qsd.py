import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.qiskit_qsd4 import (
    create_unitary,
    create_state_unitary,
    calculate_probabilities,
    simulate_circuit,
    create_quantum_circuit
)

# -----------------------------
# Test unitary creation
# -----------------------------
def test_create_unitary_identity():
    H = np.zeros((2, 2))
    U = create_unitary(H, t=1)
    assert np.allclose(U, np.identity(2))

def test_create_unitary_nonzero():
    H = np.array([[0, 1], [1, 0]])  # Pauli-X
    U = create_unitary(H, t=np.pi/2)
    expected = scipy.linalg.expm(-1j * H * np.pi/2)
    assert np.allclose(U, expected)

# -----------------------------
# Test state to unitary
# -----------------------------
def test_create_state_unitary_identity():
    v = np.array([1, 0])
    U = create_state_unitary(v)
    assert np.allclose(U @ np.array([1, 0]), v)

def test_create_state_unitary_nontrivial():
    v = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    U = create_state_unitary(v)
    result = U @ np.array([1, 0])
    assert np.allclose(result, v, atol=1e-12)

# -----------------------------
# Test probability calculation
# -----------------------------
def test_calculate_probabilities_basic():
    counts = {'00': 500, '01': 500}
    probs = calculate_probabilities(counts, N=2, nshots=1000)
    assert np.allclose(np.sum(probs), 1.0)

def test_calculate_probabilities_zero_counts():
    counts = {}
    probs = calculate_probabilities(counts, N=2, nshots=1000)
    assert np.allclose(np.sum(probs), 0.0)

# -----------------------------
# Test small circuit simulation
# -----------------------------
def test_simulate_circuit_simple():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    counts = simulate_circuit(qc, nshots=1000)
    assert '0' in counts and '1' in counts
    total_counts = sum(counts.values())
    assert total_counts == 1000

def test_create_quantum_circuit_shape():
    state = np.array([1, 0])
    U = np.identity(2)
    qc = create_quantum_circuit(N=1, state=state, U=U)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 1

# -----------------------------
# Integration test: mini QSD simulation
# -----------------------------
def test_mini_qsd_simulation():
    """Run a tiny 2-qubit Hamiltonian evolution to verify end-to-end pipeline."""
    N = 2
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])  # simple 2-qubit Hamiltonian
    state = np.array([1, 0, 0, 0])
    state /= np.linalg.norm(state)
    nshots = 100

    # Create state unitary
    U_state = create_state_unitary(state)
    
    # QSD decomposition
    from qiskit.synthesis.unitary.qsd import qs_decomposition
    qsd_state = qs_decomposition(U_state)
    
    # Time evolution
    t_steps = [0, 1]  # 2 steps
    probs = np.zeros((2**N, len(t_steps)))
    
    for k, t in enumerate(t_steps):
        U = create_unitary(H, t)
        # Compose QSD state with evolution
        from src.qiskit_qsd4 import create_quantum_circuit1
        qc = create_quantum_circuit1(N, qsd_state, U)
        counts = simulate_circuit(qc, nshots)
        probs[:, k] = calculate_probabilities(counts, N, nshots)
    
    # Check probabilities sum ~ 1 for each time step
    for col in range(probs.shape[1]):
        assert np.isclose(np.sum(probs[:, col]), 1.0, atol=1e-2)
