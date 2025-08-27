"""
Qiskit Quantum Shannon Decomposition (QSD) Simulation
Author: Anurag Dwivedi
Description: Simulates quantum circuits using QSD decomposition from a given Hamiltonian
             and initial quantum state, and computes measurement probabilities.
"""

import numpy as np
import scipy
from scipy.stats import unitary_group
from qiskit.synthesis.unitary.qsd import qs_decomposition
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit import qpy
import sys

# -----------------------------
# Quantum circuit creation
# -----------------------------
def create_quantum_circuit(N, state, U):
    """Create a quantum circuit initialized to `state` and apply the unitary `U` via QSD."""
    qsd = qs_decomposition(U)
    circuit = QuantumCircuit(N)
    circuit.initialize(state, range(N))
    circuit.compose(qsd)
    circuit.measure_all()
    return circuit

def create_quantum_circuit1(N, qsd_state, U):
    """Compose an existing QSD state with a new unitary U."""
    qsd = qs_decomposition(U)
    circuit = qsd_state.compose(qsd)
    circuit.measure_all()
    return circuit

# -----------------------------
# Unitary and state utilities
# -----------------------------
def create_unitary(H, t, conversion=1):
    """Return the time-evolution unitary U = exp(-i H t)."""
    return scipy.linalg.expm(-t * conversion * 1j * H)

def create_state_unitary(v):
    """Create a unitary that maps |0> to the state `v`."""
    dim = v.size
    if v[0] and not np.any(v[1:]):
        return np.identity(dim)
    e1 = np.zeros(dim)
    e1[0] = 1
    w = v / np.linalg.norm(v) - e1
    U = np.identity(dim) - 2 * (np.outer(w.T, w)) / np.dot(w, w.T)
    if np.linalg.norm(U @ e1 - v) > 1e-12:
        print('Warning: unitary creates init vector with error', np.linalg.norm(U @ e1 - v))
    return U

# -----------------------------
# Simulation and probabilities
# -----------------------------
def simulate_circuit(circuit, nshots):
    """Simulate the quantum circuit and return the counts."""
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=nshots)
    result = job.result()
    return result.get_counts(circuit)

def calculate_probabilities(counts, N, nshots):
    """Convert counts into probabilities for each basis state."""
    prob = np.zeros(2**N)
    for i in range(2**N):
        prob[i] = counts.get(f'{format(i, "0{}b".format(N))}', 0)
    return np.real(prob / nshots)

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Parse arguments
    N = int(sys.argv[1])
    IRead = int(sys.argv[2])
    t = float(sys.argv[3])
    total_time = float(sys.argv[4])
    H = np.loadtxt(sys.argv[5])
    state = np.loadtxt(sys.argv[6])
    nshots = int(sys.argv[7])

    # Normalize input state
    state /= np.linalg.norm(state)
    print("Norm of state:", np.linalg.norm(state))

    steps = int(total_time / t)
    tp = np.linspace(0, total_time, steps + 1)

    # Create initial QSD state
    U_state = create_state_unitary(state)
    qsd_state = qs_decomposition(U_state)

    # Run simulation over time
    probs = np.zeros((2**N, len(tp)))
    for k, time_step in enumerate(tp):
        U = create_unitary(H, time_step)
        circuit = create_quantum_circuit1(N, qsd_state, U)

        # Print gate counts
        print(f"\nGate Counts (t={time_step}):")
        print(circuit.count_ops())

        decomposed_circuit = circuit.decompose()
        print("\nDecomposed Gate Counts:")
        print(decomposed_circuit.count_ops())
        print("\nDecomposed Circuit:")
        print(decomposed_circuit)

        # Save QPY file
        with open(f'circuit_t{int(time_step):04d}.qpy', 'wb') as f:
            qpy.dump(decomposed_circuit, f)

        counts = simulate_circuit(circuit, nshots)
        probs[:, k] = calculate_probabilities(counts, N, nshots)

    # Save probabilities
    print("Total probability sum per state:", np.sum(probs, axis=1))
    np.savetxt('probability.txt', probs)
