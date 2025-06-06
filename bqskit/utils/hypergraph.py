import sys
import os

# Ensure standard library modules are prioritized
sys.path.insert(0, os.path.dirname(os.__file__))


# Now import your modules
from bqskit.ir import Circuit
#from bqskit.ir.gates import Gate
from bqskit.ir.gate import Gate
import subprocess
from typing import Dict, List, Tuple


import subprocess
import random
"""
The code has two main functions:

1-circuit_to_hypergraph: Converts a BQSKIT quantum circuit into a hypergraph representation and writes it to a file 
in hMETIS format, which KaHyPar can process.

2-test_kahypar_partitioning: A helper function to test the partitioning process by running KaHyPar on the generated 
file and retrieving the results.

"""


#Converts a Circuit object into a hypergraph and saves it as a file(output_file).
def circuit_to_hypergraph(circuit: Circuit, 
                          output_file: str = "circuit.hgr",
                          # NEW: Add error rate parameters with defaults
                           gate_error_rates: Dict[str, float] = {'HGate': 1e-3, 'CNOTGate': 1e-2}
                          ) -> None:
    """
    Convert a BQSKIT Circuit to a hypergraph with quantum-aware features.
    Creates two hyperedge types: gate-level (multi-qubit) and temporal chains.
    """
    from bqskit.ir.gates import CNOTGate  # Import needed for gate type check
    
    # Step 1: Map gates to nodes with criticality weights
    gates = list(circuit.operations())
    num_gates = len(gates)
    num_qubits = circuit.num_qudits
    # MODIFIED: Node weights now fidelity-aware
    node_weights = [
        10 * (1/gate_error_rates.get(op.gate.__class__.__name__, 1e-2))  #Weight = 1/error_rate
        if isinstance(op.gate, CNOTGate) else 
        1 * (1/gate_error_rates.get(op.gate.__class__.__name__, 1e-3))   #HGate default 0.1% error
        for op in gates
    ]

    # Step 2: Build two hyperedge types
    hyperedges = []
    edge_weights = []
    gate_positions = {i: i for i in range(num_gates)}  # Temporal positions

    # Type 1: Gate-level hyperedges (multi-qubit interactions)
    for i, op in enumerate(gates):
        if len(op.location) > 1: # Check if gate acts on >1 qubit (e.g., CNOT)
            hyperedges.append([i]) # Create a hyperedge containing ONLY this gate
            ## MODIFIED: Edge weight now incorporates gate error rate
            error_rate = gate_error_rates.get(op.gate.__class__.__name__, 1e-2)
            edge_weights.append(100 * len(op.location) * (1/error_rate))  # NEW: Fidelity-driven weight
            # Weight = 100 * num_qubits_in_gate/error rate
            #Purpose: Explicitly model multi-qubit gates (e.g., CNOTs) as standalone hyperedges.
            #Why? Penalize cutting through multi-qubit gates, which disrupt entanglement.


    # Type 2: Temporal chains (qubit lifetimes)
    for q in range(num_qubits):
        qubit_ops = [i for i, op in enumerate(gates) if q in op.location] # All gates on qubit q
        if len(qubit_ops) > 1:
            hyperedges.append(qubit_ops) # Hyperedge = all gates on qubit q
            # Calculate temporal density weight
            temporal_density = 100 * (len(qubit_ops) // 2) * (1/gate_error_rates.get('HGate', 1e-3))  # NEW: Fidelity-driven weight
            #Purpose: Model qubit timelines but with weights adjusted for temporal locality.
            #Qubit 0 has 3 gates → temporal_density = 50 // (3-1) = 25.
            #Qubit 1 has 2 gates → temporal_density = 50 // (2-1) = 50.
            #Why? Favor grouping gates that are close in time on the same qubit.
            edge_weights.append(max(1, temporal_density))
            #

    # Step 3: Filter empty hyperedges (safeguard)
    valid_hyperedges = [he for he in hyperedges if he]
    valid_edge_weights = [edge_weights[i] for i in range(len(hyperedges)) if hyperedges[i]]

    # NEW: Weight normalization
    if valid_edge_weights:  # Prevent division by zero
        max_weight = max(valid_edge_weights)
        valid_edge_weights = [int(w * 1e6 / max_weight) for w in valid_edge_weights]
        
    # Step 4: Write to hMETIS format
    with open(output_file, 'w') as f:
        f.write(f"{len(valid_hyperedges)} {num_gates} 1\n")  # Header
        for he, ew in zip(valid_hyperedges, valid_edge_weights):
            he_str = ' '.join(str(g+1) for g in he)  # 1-based indexing
            f.write(f"{ew} {he_str}\n")
        for nw in node_weights:
            f.write(f"{nw}\n")


########################################


########################################


def test_kahypar_partitioning(input_file: str = "circuit.hgr", num_partitions: int = 2) -> List[int]:
    """
    Partition a hypergraph using the Mt-KaHyPar binary and return the partition labels.
    
    Args:
        input_file (str): Path to the hMETIS hypergraph file (default: "circuit.hgr").
        num_partitions (int): Number of partitions (default: 2).
    
    Returns:
        List[int]: Partition labels for each node (gate).
    """
    # Path to the Mt-KaHyPar binary
    mtkahypar_bin = "/mnt/c/bqskit/mt-kahypar/build/mt-kahypar/application/mtkahypar"
    #mtkahypar_bin = "mtkahypar" # Or "MtKaHyPar" depending on the executable name
    #make sure to put your own path to make sure things work well.
    
    # Expected partition file (adjust based on Mt-KaHyPar's default naming)
    partition_file = f"{input_file}.part{num_partitions}.epsilon{0.05}.seed42.KaHyPar"  # Matches --seed 42
    
    # Run Mt-KaHyPar with partition file output enabled
    cmd = [
        mtkahypar_bin,
        "-h", input_file,
        "-k", str(num_partitions),
        "-e", "0.05",# Allow 5% imbalance for better cuts
        "-o", "km1",
        #"--objective=cutnet", # Optimize for cut AND fidelity
        "-m", "direct",
        "--preset-type", "default",
        #"--imbalance", "5",  
        "--seed", "42",
        "-t", "16",
        "--write-partition-file=true",  # Enable partition file output
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # print(f"Command: {' '.join(cmd)}")
    #print(f"Stdout: {result.stdout}")
    #print(f"Stderr: {result.stderr}")
    
    if result.returncode != 0:
        raise RuntimeError(f"Mt-KaHyPar failed: {result.stderr}")
    
    #print("Partitioning done with Mt-KaHyPar")
    #print(result.stdout)

    # Check for partition file
    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file {partition_file} not found. Check Mt-KaHyPar output naming.")
    
    with open(partition_file, 'r') as f:
        partition_labels = [int(line.strip()) for line in f]
    
    print("Labels retrieved:")
    print(partition_labels)
    return partition_labels


if __name__ == "__main__":
    from bqskit.ir.gates import CNOTGate, HGate
    # test_circuit = Circuit(3)
    # test_circuit.append_gate(HGate(), [0])
    # test_circuit.append_gate(CNOTGate(), [0, 1])
    # test_circuit.append_gate(HGate(), [0])
    # test_circuit.append_gate(CNOTGate(), [1, 2])
    from bqskit.ir.gates import HGate, CNOTGate
    # Set random seed for reproducibility (optional, remove for different randomizations)
    # random.seed(42)

    # # Assuming a custom quantum circuit library with Circuit, HGate, and CNOTGate
    # # Create a circuit with 6 qubits (Q_0 to Q_5)
    # circuit_ind_1 = Circuit(6)

    #     # Q_0-Q_1 gates (original pair, now randomized for CNOTs)
    # circuit_ind_1.append_gate(HGate(), 0)          # H_1 on Q_0
    # # CNOT_2: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_2: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 0)          # H_3 on Q_0
    # # CNOT_4: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_4: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 0)          # H_5 on Q_0
    # # CNOT_6: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_6: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 1)          # H_7 on Q_1

    # # Q_2-Q_3 gates (original pair, now randomized for CNOTs)
    # circuit_ind_1.append_gate(HGate(), 2)          # H_8 on Q_2
    # # CNOT_9: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_9: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 2)          # H_10 on Q_2
    # # CNOT_11: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_11: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 2)          # H_12 on Q_2
    # # CNOT_13: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_13: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 3)          # H_14 on Q_3

    # # Q_4-Q_5 gates (original pair, now randomized for CNOTs)
    # circuit_ind_1.append_gate(HGate(), 4)          # H_15 on Q_4
    # # CNOT_16: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_16: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 4)          # H_17 on Q_4
    # # CNOT_18: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_18: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 4)          # H_19 on Q_4
    # # CNOT_20: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_20: control Q_{}, target Q_{}
    # circuit_ind_1.append_gate(HGate(), 4)          # H_21 on Q_4
    # # CNOT_22: Random control and target
    # control, target = random.sample(range(6), 2)
    # circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_22: control Q_{}, target Q_{}

    test_circuit = Circuit(5)

# Add 10 gates (Hadamard and CNOT)
    test_circuit.append_gate(HGate(), [0])
    test_circuit.append_gate(CNOTGate(), [0, 1])
    test_circuit.append_gate(HGate(), [1])
    test_circuit.append_gate(CNOTGate(), [1, 2])
    test_circuit.append_gate(HGate(), [2])
    test_circuit.append_gate(CNOTGate(), [2, 3])
    test_circuit.append_gate(HGate(), [3])
    test_circuit.append_gate(CNOTGate(), [3, 4])
    test_circuit.append_gate(HGate(), [4])
    test_circuit.append_gate(CNOTGate(), [4, 0])
    # Generate both hypergraphs
    #circuit_to_hypergraph_old(test_circuit, "old.hgr")  # Original function
    circuit_to_hypergraph(test_circuit, "circuit.hgr",gate_error_rates={
        'HGate': 1e-3,
        'CNOTGate': 5e-2,  # Higher penalty for noisy CNOTs
        'SwapGate': 3e-2
    })      # New function
    print("Hypergraph written to circuit.hgr")        
    with open("circuit.hgr") as f:
        print("\nNew Quantum-Aware Hypergraph:\n", f.read())
    
    partitions = test_kahypar_partitioning("circuit.hgr", 2)
    print("Partition Labels:", partitions)
    os.remove("circuit.hgr")
    os.remove("circuit.hgr.part2.epsilon0.05.seed42.KaHyPar")  # Clean up partition file

    # circuit_to_hypergraph(circuit, "circuit.hgr")
    # #print("Hypergraph written to circuit.hgr")
    # partitions = test_kahypar_partitioning("circuit.hgr", 2)
    # print("Partition Labels:", partitions)
    # os.remove("circuit.hgr")
    # os.remove("circuit.hgr.part2.epsilon0.03.seed42.KaHyPar")  # Clean up partition file

    """
    Labels retrieved: [1, 0, 1, 0]
    This means:

    Gate 0 (H@0): Partition 1

    Gate 1 (CNOT@0-1): Partition 0

    Gate 2 (H@0): Partition 1

    Gate 3 (CNOT@1-2): Partition 0

    How Partitioning Works
        Step 1: Hypergraph Input
        The hypergraph file (new.hgr) tells Mt-KaHyPar:

        Nodes (gates) and their importance (weights: CNOTs = 10, H = 1).

        Hyperedges and their criticality (weights: CNOT hyperedges = 200, temporal chains =50-25).

        Step 2: Mt-KaHyPar Partitioning
        Mt-KaHyPar assigns gates to partitions while:

        Minimizing cuts in high-weight hyperedges (e.g., avoid cutting CNOTs).

        Balancing partition sizes (with 5% allowed imbalance).

        Respecting node weights (prioritizing CNOTs).

        Step 3: Result Analysis
        CNOT@0-1 (Gate 1) and CNOT@1-2 (Gate 3) are in Partition 0.

        Both are grouped to preserve their high-weight hyperedges (200).

        H@0 (Gates 0 and 2) are split into Partition 1.

        Less critical (weight 1) and part of a lower-weight temporal chain (25).
    """