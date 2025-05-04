import sys
import os

# Ensure standard library modules are prioritized
sys.path.insert(0, os.path.dirname(os.__file__))


# Now import your modules
from bqskit.ir import Circuit
#from bqskit.ir.gates import Gate
from bqskit.ir.gate import Gate
import subprocess
from typing import List, Tuple


import subprocess

"""
The code has two main functions:

1-circuit_to_hypergraph: Converts a BQSKIT quantum circuit into a hypergraph representation and writes it to a file 
in hMETIS format, which KaHyPar can process.

2-test_kahypar_partitioning: A helper function to test the partitioning process by running KaHyPar on the generated 
file and retrieving the results.

"""


#Converts a Circuit object into a hypergraph and saves it as a file(output_file).
def circuit_to_hypergraph(circuit: Circuit, output_file: str = "circuit.hgr") -> None:
    """
    Convert a BQSKIT Circuit to a hypergraph and write it to an hMETIS-compatible file for KaHyPar.
    
    Args:
        circuit (Circuit): The input quantum circuit.
        output_file (str): Path to save the hypergraph file (default: "circuit.hgr").
    
    Returns:
        None: Writes the hypergraph to a file instead of returning it.
    """
    # Step 1: Map gates to nodes
    gates = list(circuit.operations())  # Get all gates in order
    num_gates = len(gates)
    node_weights = [1] * num_gates  # Simple weight; could adjust based on gate type
    """
    gates = list(circuit.operations()): Extracts all gates from the circuit in their execution order. Each gate is an 
    Operation object with attributes like gate (type) and location (qubits it acts on).
    num_gates = len(gates): Counts the gates, which become nodes in the hypergraph.
    node_weights = [1] * num_gates: Assigns a weight of 1 to each node (gate). This could be modified later 
    (e.g., higher weights for multi-qubit gates like CNOT).
    """


    # Step 2: Build hyperedges per qubit
    num_qubits = circuit.num_qudits
    hyperedges = [[] for _ in range(num_qubits)]  # One hyperedge per qubit
    #Initializes a list of empty lists, one per qubit. Each qubit will have a hyperedge connecting all gates acting on it.
    gate_positions = {}  # Map gate index to its temporal position, he order in which the gate appears in the circuit.

    for i, op in enumerate(gates):
        gate_positions[i] = i  # Temporal position is the index in the sequence
        for qubit in op.location:  # op.location is a tuple of qubit indices
            hyperedges[qubit].append(i)  # Add gate index to qubit's hyperedge
    """
    Circuit:
        q0: H --- CNOT ---
                    |
        q1: ------CNOT --- H

    Gates:

        Gate 0: H on qubit 0.

        Gate 1: CNOT on qubits 0 and 1.

        Gate 2: H on qubit 1.

    Hyperedges:

        Qubit 0: [0, 1] (gates 0 and 1 act on qubit 0).

        Qubit 1: [1, 2] (gates 1 and 2 act on qubit 1).

    Hyperedges:
        - Qubit 0: [0, 1] (H, CNOT)
        -Qubit 1: [1, 2] (CNOT, H)
    """

    """
    What it does: Assigns a weight to each hyperedge (qubit) based on how close together its gates are in the circuit's 
    sequence.
    Why: Higher weights for gates that are close in time encourage KaHyPar to keep them in the same partition, which can 
    improve optimization by preserving local gate interactions.

    Visual Example
    Circuit:
        q0: H --- CNOT ---
                    |
        q1: ------CNOT --- H

    Hyperedges:
        - q0: [0, 1] (positions [0, 1]) → avg_distance = 1 → weight = 50
        - q1: [1, 2] (positions [1, 2]) → avg_distance = 1 → weight = 50
    """
    # Step 3: Calculate temporal edge weights
    """
    Assign weights to each hyperedge (qubit) based on the temporal proximity of the gates acting on that qubit.

    Gates that are closer together in time (i.e., have smaller temporal distances) should have higher weights.


    """
    edge_weights = []
    for q in range(num_qubits):
        if len(hyperedges[q]) <= 1:
            edge_weights.append(1)  # Default weight for single-gate edges
        else:
            # Calculate average temporal distance between consecutive gates
            positions = sorted([gate_positions[g] for g in hyperedges[q]])
            avg_distance = sum(positions[j + 1] - positions[j] for j in range(len(positions) - 1)) / (len(positions) - 1)
            # Inverse weight: closer gates = higher weight
            weight = int(100 / (avg_distance + 1))  # Scale to reasonable integer range
            edge_weights.append(max(1, weight))  # Ensure non-zero weight

    # Step 4: Filter out empty hyperedges
    valid_hyperedges = [he for he in hyperedges if he]
    valid_edge_weights = [edge_weights[i] for i in range(len(hyperedges)) if hyperedges[i]]
    num_hyperedges = len(valid_hyperedges)

    # Step 5: Write to hMETIS file format for KaHyPar
    with open(output_file, 'w') as f:
        # Header: number of hyperedges, number of nodes, format (1 = weighted hyperedges)
        f.write(f"{num_hyperedges} {num_gates} 1\n")
        
        # Write hyperedges and their weights
        for he, ew in zip(valid_hyperedges, valid_edge_weights):
            # Convert gate indices to 1-based (hMETIS/KaHyPar convention)
            he_str = " ".join(str(g + 1) for g in he)
            f.write(f"{ew} {he_str}\n")
        
        # Write node weights
        for nw in node_weights:
            f.write(f"{nw}\n")


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
    
    # Expected partition file (adjust based on Mt-KaHyPar's default naming)
    partition_file = f"{input_file}.part{num_partitions}.epsilon{0.03}.seed42.KaHyPar"  # Matches --seed 42
    
    # Run Mt-KaHyPar with partition file output enabled
    cmd = [
        mtkahypar_bin,
        "-h", input_file,
        "-k", str(num_partitions),
        "-e", "0.03",
        "-o", "km1",
        "-m", "direct",
        "--preset-type", "default",
        "--seed", "42",
        "-t", "4",
        "--write-partition-file=true",  # Enable partition file output
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # print(f"Command: {' '.join(cmd)}")
    # print(f"Stdout: {result.stdout}")
    # print(f"Stderr: {result.stderr}")
    
    if result.returncode != 0:
        raise RuntimeError(f"Mt-KaHyPar failed: {result.stderr}")
    
    #print("Partitioning done with Mt-KaHyPar")
    #print(result.stdout)

    # Check for partition file
    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file {partition_file} not found. Check Mt-KaHyPar output naming.")
    
    with open(partition_file, 'r') as f:
        partition_labels = [int(line.strip()) for line in f]
    
    print("Labels retrieved")
    return partition_labels

# import pymetis
# def test_kahypar_partitioning_in_memory(circuit: Circuit, num_partitions: int) -> List[int]:
#     gates = list(circuit.operations())
#     hyperedges = [[] for _ in range(circuit.num_qudits)]
#     for i, op in enumerate(gates):
#         for qubit in op.location:
#             hyperedges[qubit].append(i)
#     adj = [[] for _ in range(len(gates))]
#     for he in hyperedges:
#         for i in range(len(he)):
#             for j in range(i + 1, len(he)):
#                 adj[he[i]].append(he[j])
#                 adj[he[j]].append(he[i])
#     _, part = pymetis.part_graph(num_partitions, adjacency=adj)
#     return part


if __name__ == "__main__":
    from bqskit.ir.gates import CNOTGate, HGate
    circuit = Circuit(2)
    circuit.append_gate(HGate(), [0])
    circuit.append_gate(CNOTGate(), [0, 1])
    circuit.append_gate(HGate(), [1])
    circuit_to_hypergraph(circuit, "circuit.hgr")
    #print("Hypergraph written to circuit.hgr")
    partitions = test_kahypar_partitioning("circuit.hgr", 2)
    #print("Partition Labels:", partitions)
    os.remove("circuit.hgr")
    os.remove("circuit.hgr.part2.epsilon0.03.seed42.KaHyPar")  # Clean up partition file