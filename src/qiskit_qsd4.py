import numpy as np
import scipy
from scipy.stats import unitary_group
from qiskit.synthesis.unitary.qsd import qs_decomposition
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
#from qiskit.compiler import assemble
#from qiskit import transpile, assemble
from qiskit.visualization import plot_circuit_layout
from qiskit import qpy
import sys


def create_quantum_circuit(N, state, U):
    """Create a quantum circuit initialized to `state` and apply the unitary from `qsd`."""
    qsd = qs_decomposition(U)
    circuit = QuantumCircuit(N)
    circuit.initialize(state, range(N))
    circuit.compose(qsd)
    circuit.measure_all()
    return circuit

def create_quantum_circuit1(N, qsd_state, U):
    """Create a quantum circuit initialized to `state` and apply the unitary from `qsd`."""
    qsd = qs_decomposition(U)
    circuit = QuantumCircuit(N)
    circuit = qsd_state.compose(qsd)
    circuit.measure_all()
    return circuit

def create_unitary(H, t, conversion=1):
    U = scipy.linalg.expm(-t*conversion*1j*H)
    return U

def simulate_circuit(circuit, nshots):
    """Simulate the quantum circuit and return the measurement probabilities."""
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=nshots)
    result = job.result()
    return result.get_counts(circuit)

def calculate_probabilities(counts, N, nshots):
    """Calculate and return the measurement probabilities."""
    prob = np.zeros(2**N)
    for i in range(int(2**N)):
        prob[i] = counts.get('%s' % (format(i, '0%sb' % (N))), 0)
    probs = np.real(prob / nshots)
    return probs

def create_state_unitary(v):
    dim = v.size
    # Return identity if v is a multiple of e1
    if v[0] and not np.any(v[1:]):
        return np.identity(dim)
    e1 = np.zeros(dim)
    e1[0] = 1
    w = v/np.linalg.norm(v) - e1
    U = np.identity(dim) - 2*((np.outer(w.T, w))/(np.dot(w, w.T)))
    if (np.linalg.norm((U@ e1)-v) > 10**(-12)):
        print('unitary creates init vector with error ',np.linalg.norm((U@ e1)-v))
    return U


N = int(sys.argv[1])
T = 2**N
IRead = int(sys.argv[2])
t = float(sys.argv[3])
total_time = float(sys.argv[4])
H = np.loadtxt(sys.argv[5])
state = np.loadtxt(sys.argv[6])
nshots = int(sys.argv[7])
conv = 1
'''
if (IRead == 1):
	print('Reading in unitary from input file.')
	U = np.loadtxt(inp_file,dtype=complex)
elif (IRead == 2):
	print('Creating an arbitrary unitary.')
	U = unitary_group.rvs(T)
elif (IRead == 3):
	print('Reading in Hamiltonian matrix from input file.')
	H = np.loadtxt(inp_file)
	U = scipy.linalg.expm(-t*conv*1j*H)
else:
        print('No input U/H provided')
        sys.exit()

np.savetxt('U-%s-t%s.txt'%(N,'{:04d}'.format(int(t))),U)
print('U = ', U)
'''
# Normalize the state to ensure it's a valid quantum state
state /= np.linalg.norm(state)
# Verify normalization (optional)
print("Norm of state:", np.linalg.norm(state))
steps = int(total_time/t)
tp = np.linspace(0,total_time,steps+1)

U_state = create_state_unitary(state)
qsd_state = qs_decomposition(U_state)

# Create the quantum circuit
probs = np.zeros((2**N, len(tp)))
k = 0 
for i in (tp):
    U = create_unitary(H, i)
    #circuit = create_quantum_circuit(N, state, U)
    circuit = create_quantum_circuit1(N, qsd_state, U)
    # Count the number of gates of each type
    gate_counts = circuit.count_ops()
    print("\nGate Counts (%s):"%i)
    print(gate_counts)
    # Decompose the circuit to break down composite operations
    decomposed_circuit = circuit.decompose()
    decomposed_counts = decomposed_circuit.count_ops()
    print("\nDecomposed Gate Counts:")
    print(decomposed_counts)
    # Print the decomposed circuit for verification
    print("\nDecomposed Circuit:")
    print(decomposed_circuit)
    with open('circuit_t%s.qpy' % ('{:04d}'.format(int(i))), 'wb') as f:
        qpy.dump(decomposed_circuit, f)
    counts = simulate_circuit(circuit, nshots)
    probs[:,k] = calculate_probabilities(counts, N, nshots)
    k = k+1

print(np.sum(probs, axis=1))
np.savetxt('probability.txt', probs)

