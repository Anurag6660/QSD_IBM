# Qiskit Quantum Shannon Decomposition (QSD) Example

This document provides a complete example workflow for running the QSD simulation using the `qiskit_qsd4.py` code.

---

## 1️⃣ Setup Environment

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # For tcsh: source .venv/bin/activate.csh
pip install -r requirements.txt
```

Dependencies include:
- `qiskit` (tested with version 0.44.1)
- `qiskit-aer`
- `numpy`
- `scipy`

---

## 2️⃣ Repository Structure

```
qiskit-qsd/
│── src/                   # Python code
│   └── qiskit_qsd4.py
│── data/
│   └── Hamiltonian/       # Input Hamiltonians
│       └── left_Ham1.txt
│── results/               # Output files
│   ├── Debug_slurm1.script
│   ├── X_left.txt
│   └── left1/
│── tests/                 # Unit and integration tests
│── docs/                  # Documentation (this file)
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 3️⃣ Running Locally

Run the QSD simulation with the sample Hamiltonian:

```bash
python src/qiskit_qsd4.py N IRead t total_time data/Hamiltonian/left_Ham1.txt results/X_left.txt nshots
```

**Arguments:**
- `N`: Number of qubits  
- `IRead`: Input type (1: unitary file, 2: random unitary, 3: Hamiltonian)  
- `t`: Time step  
- `total_time`: Total simulation time  
- `nshots`: Number of shots per circuit  

Example:

```bash
python src/qiskit_qsd4.py 3 3 1 1000 data/Hamiltonian/left_Ham1.txt results/X_left.txt 1000
```

---

## 4️⃣ Output Files

- `probability.txt`: Probabilities of all basis states over time  
- `circuit_tXXXX.qpy`: Saved QPY circuits for each time step  
- Gate counts printed to the console for verification  

All outputs are written to the `results/` folder. Paths are relative, so the repo is portable.

---

## 5️⃣ Simulation Workflow

1. Normalize the input quantum state  
2. Create a unitary that maps |0> to the input state (`create_state_unitary`)  
3. Decompose the unitary using QSD (`qs_decomposition`)  
4. Construct a quantum circuit and apply time-evolution unitaries  
5. Simulate each circuit using `AerSimulator`  
6. Calculate measurement probabilities using `calculate_probabilities`  
7. Save circuits (`.qpy`) and probabilities (`probability.txt`)  

---

## 6️⃣ Reproducing Results

- Use the sample Hamiltonian in `data/Hamiltonian/` for testing  
- To run full simulations, replace with your own Hamiltonians  
- Avoid committing large output files; keep only small sample outputs for reproducibility  

---

## 7️⃣ Notes

- The repository avoids absolute paths to make it portable  
- Large files (>50–100 MB) should be stored externally or generated on demand  
- Users on different HPC systems may need to adjust SLURM options (`-A`, `--partition`, etc.)  

---

## 8️⃣ References

- [Qiskit Documentation](https://qiskit.org/documentation/)  
- Debadrita's QSD paper
