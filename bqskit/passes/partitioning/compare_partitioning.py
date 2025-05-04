import multiprocessing
multiprocessing.set_start_method('spawn')  # Add this before other imports

from bqskit.ir.circuit import Circuit
#from bqskit.ir.gates import GateSet
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.util.extend import ExtendBlockSizePass
from bqskit.passes.partitioning.hypergraphpartition import EnhancedHypergraphPartitionPass
from bqskit.runtime.manager import Manager  # Try this import
import os
import asyncio
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.compiler.basepass import BasePass
from bqskit.passes.util.log import LogPass  # Standard pass for logging
from bqskit.compiler.compile import build_partitioning_workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import GeneralGate 
from bqskit.passes.partitioning.gateset import CustomGateSet
from bqskit.compiler.task import CompilationTask
from bqskit.runtime.task import RuntimeTask  # Use task
from bqskit.runtime import default_manager_port
from bqskit.passes.util.log import LogPass
import logging
import time
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.compiler import Compiler

class DoNothingPass(BasePass):
    def run(self, circuit: Circuit, data: dict) -> None:
        print('data inside donothing: ', data)
        pass


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def partition_with_quick(circuit: Circuit, block_size: int = 3) -> list[Circuit]:
    """Partition using QuickPartitioner via Compiler."""
    workflow = [QuickPartitioner(block_size)]
    data = PassData(circuit)
    data._gate_set = CustomGateSet({HGate(), CNOTGate()})
    circuit_copy = circuit.copy()
    print("Initial circuit:", circuit_copy)
    print("Initial circuit ops:", circuit_copy.num_operations)
    
    compiler = Compiler(num_workers=1)
    try:
        print("Starting compilation...")
        compiled_circuit = compiler.compile(circuit_copy, workflow, data=data)
        print("Circuit after compile:", compiled_circuit)
        print("Ops after compile:", compiled_circuit.num_operations)
        
        partitions = []
        for op in compiled_circuit:
            if isinstance(op.gate, CircuitGate):
                subcircuit = op.gate._circuit.copy()
                partitions.append(subcircuit)
        
        if partitions:
            data['partitions'] = partitions
        else:
            logger.warning("No CircuitGates found; partitions may not have been created.")
        
        print("Data keys after compilation:", list(data.keys()))
        print("Partitions retrieved:", [f"Circuit with {p.num_operations} gates" for p in partitions])
        return data.get('partitions', [])
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        compiler.close()

async def partition_with_quick_manual(circuit: Circuit, block_size: int = 3) -> list[Circuit]:
    """Partition using QuickPartitioner manually."""
    workflow = [QuickPartitioner(block_size)]
    data = PassData(circuit)
    data._gate_set = CustomGateSet({HGate(), CNOTGate()})
    circuit_copy = circuit.copy()
    print("Manual - Initial circuit:", circuit_copy)
    print("Manual - Initial circuit ops:", circuit_copy.num_operations)
    
    for pass_obj in workflow:
        print(f"Running {pass_obj.__class__.__name__}...")
        await pass_obj.run(circuit_copy, data)
        print(f"After {pass_obj.__class__.__name__} - Circuit:", circuit_copy)
        print(f"After {pass_obj.__class__.__name__} - Circuit ops:", circuit_copy.num_operations)
        print(f"After {pass_obj.__class__.__name__} - Data keys:", list(data.keys()))
    
    partitions = []
    for op in circuit_copy:
        if isinstance(op.gate, CircuitGate):
            subcircuit = op.gate._circuit.copy()
            partitions.append(subcircuit)
    
    if partitions:
        data['partitions'] = partitions
    else:
        logger.warning("No CircuitGates found in manual pass.")
    
    print("Manual - Partitions retrieved:", [f"Circuit with {p.num_operations} gates" for p in partitions])
    return data.get('partitions', [])

# async def partition_with_hypergraph(circuit: Circuit, block_size: int = 3,num_workers: int = 8) -> list[Circuit]:
#     """Partition using HypergraphPartitionPass."""
#     hypergraph_pass = EnhancedHypergraphPartitionPass(block_size=block_size, num_workers=num_workers)
#     data = {}
#     await hypergraph_pass.run(circuit, data)
#     print("HypergraphPartitionPass data:", data)
#     return data.get('partitions', [])

# Updated partition_with_hypergraph
async def partition_with_hypergraph(circuit: Circuit, block_size: int = 3, num_workers: int = 8) -> list[Circuit]:
    hypergraph_pass = EnhancedHypergraphPartitionPass(block_size=block_size, num_workers=num_workers)
    data = {}
    await hypergraph_pass.run(circuit, data)
    print("Enhanced HypergraphPartitionPass data:", data)
    return data.get('partitions', [])

async def compare_partitions(circuit: Circuit, block_size: int = 3,num_workers: int = 8):
    import time
    print("\n=== Using Compiler ===")
    start = time.time()
    quick_subcircuits = partition_with_quick(circuit, block_size)  # Synchronous call
    quick_time = time.time() - start
    print("\n--- Original Partitioning (QuickPartitioner) ---") 
    print(f"Time: {time.time() - start:.3f} seconds")
    print(f"Number of partitions: {len(quick_subcircuits)}")
    print(f"Inferred cuts: {len(quick_subcircuits) - 1}")  # Cuts = Partitions - 1
    for i, subcircuit in enumerate(quick_subcircuits):
        print(f"Partition {i}: {subcircuit.num_operations} gates, Depth: {subcircuit.depth}")
        print(subcircuit) 
    # print(f"Number of partitions: {len(quick_subcircuits)}")
    # for i, subcircuit in enumerate(quick_subcircuits):
    #     print(f"Partition {i}: {subcircuit.num_operations} gates")
    #     print(f"Depth: {subcircuit.depth}")
    #     print(subcircuit)

    # print("\n=== Using Manual Passes ===")
    # quick_subcircuits_manual = await partition_with_quick_manual(circuit, block_size)
    # print("\n--- Manual Partitioning (QuickPartitioner) ---")
    # print(f"Number of partitions: {len(quick_subcircuits_manual)}")
    # for i, subcircuit in enumerate(quick_subcircuits_manual):
    #     print(f"Partition {i}: {subcircuit.num_operations} gates")
    #     print(f"Depth: {subcircuit.depth}")
    #     print(subcircuit)
    start = time.time()
    hypergraph_subcircuits =await partition_with_hypergraph(circuit.copy(), block_size,num_workers)
    hyper_time = time.time() - start
    print("\n--- Hypergraph Partitioning ---") 
    print(f"Time: {time.time() - start:.3f} seconds")
    print(f"Number of partitions: {len(hypergraph_subcircuits)}")
    print(f"Inferred cuts: {len(hypergraph_subcircuits) - 1}")  # Cuts = Partitions - 1
    for i, subcircuit in enumerate(hypergraph_subcircuits):
        print(f"Partition {i}: {subcircuit.num_operations} gates, Depth: {subcircuit.depth}")
        print(subcircuit) 
    def getQuickPartitions():
        return quick_subcircuits
    def getHyperPartitions():
        return hypergraph_subcircuits
        # Comparison
    print("\n=== Comparison ===")
    print(f"Quick Partitioning Time: {quick_time:.3f} seconds")
    print(f"Hypergraph Partitioning Time: {hyper_time:.3f} seconds")
    
    quick_gate_counts = [p.num_operations for p in quick_subcircuits]
    hyper_gate_counts = [p.num_operations for p in hypergraph_subcircuits]
    quick_depths = [p.depth for p in quick_subcircuits]
    hyper_depths = [p.depth for p in hypergraph_subcircuits]
    
    print(f"Quick Partitions: {len(quick_subcircuits)}, Gate Counts: min={min(quick_gate_counts) if quick_gate_counts else 'N/A'}, max={max(quick_gate_counts) if quick_gate_counts else 'N/A'}, Total={sum(quick_gate_counts)}")
    print(f"Hyper Partitions: {len(hypergraph_subcircuits)}, Gate Counts: min={min(hyper_gate_counts) if hyper_gate_counts else 'N/A'}, max={max(hyper_gate_counts) if hyper_gate_counts else 'N/A'}, Total={sum(hyper_gate_counts)}")
    print(f"Original Total Gates: {circuit.num_operations}")
    
    print(f"Quick Max Depth: {max(quick_depths) if quick_depths else 'N/A'}")
    print(f"Hyper Max Depth: {max(hyper_depths) if hyper_depths else 'N/A'}")

    print(f"Inferred cuts for Quick: {len(quick_subcircuits) - 1}")  # Cuts = Partitions - 1
    print(f"Inferred cuts for Hyper: {len(hypergraph_subcircuits) - 1}")  # Cuts = Partitions - 1



# Add to the end of compare_partitioning.py
async def main():
    from bqskit.ir.gates import HGate, CNOTGate
    
        # Test 1: Independent Circuits
    print("\n=== Test 1: Independent Circuits ===")
    # Circuit 1: Small Independent (9 gates, 6 qubits)
    circuit_ind_1 = Circuit(6)
    circuit_ind_1.append_gate(HGate(), 0)
    circuit_ind_1.append_gate(CNOTGate(), (0, 1))
    circuit_ind_1.append_gate(HGate(), 1)
    circuit_ind_1.append_gate(HGate(), 2)
    circuit_ind_1.append_gate(CNOTGate(), (2, 3))
    circuit_ind_1.append_gate(HGate(), 3)
    circuit_ind_1.append_gate(HGate(), 4)
    circuit_ind_1.append_gate(CNOTGate(), (4, 5))
    circuit_ind_1.append_gate(HGate(), 5)
    print(f"\nCircuit 1: Small Independent Circuit ({circuit_ind_1.num_operations} gates, {circuit_ind_1.num_qudits} qubits)")
    await compare_partitions(circuit_ind_1, block_size=3)

    # Circuit 2: Medium Independent (15 gates, 8 qubits)
    circuit_ind_2 = Circuit(8)
    for q in range(0, 8, 2):
        circuit_ind_2.append_gate(HGate(), q)
        circuit_ind_2.append_gate(CNOTGate(), (q, q+1))
        circuit_ind_2.append_gate(HGate(), q+1)
    circuit_ind_2.append_gate(HGate(), 0)
    circuit_ind_2.append_gate(HGate(), 2)
    circuit_ind_2.append_gate(HGate(), 4)
    print(f"\nCircuit 2: Medium Independent Circuit ({circuit_ind_2.num_operations} gates, {circuit_ind_2.num_qudits} qubits)")
    await compare_partitions(circuit_ind_2, block_size=3)

    # Test 2: Entangled Circuits
    print("\n=== Test 2: Entangled Circuits ===")
    # Circuit 1: Small Entangled (11 gates, 6 qubits)
    circuit_ent_1 = Circuit(6)
    circuit_ent_1.append_gate(HGate(), 0)
    circuit_ent_1.append_gate(CNOTGate(), (0, 1))
    circuit_ent_1.append_gate(HGate(), 1)
    circuit_ent_1.append_gate(CNOTGate(), (0, 2))
    circuit_ent_1.append_gate(HGate(), 2)
    circuit_ent_1.append_gate(CNOTGate(), (2, 3))
    circuit_ent_1.append_gate(HGate(), 3)
    circuit_ent_1.append_gate(CNOTGate(), (2, 4))
    circuit_ent_1.append_gate(HGate(), 4)
    circuit_ent_1.append_gate(CNOTGate(), (4, 5))
    circuit_ent_1.append_gate(HGate(), 5)
    print(f"\nCircuit 1: Small Entangled Circuit ({circuit_ent_1.num_operations} gates, {circuit_ent_1.num_qudits} qubits)")
    await compare_partitions(circuit_ent_1, block_size=3)

    # Circuit 2: Medium Entangled (17 gates, 8 qubits)
    circuit_ent_2 = Circuit(8)
    circuit_ent_2.append_gate(HGate(), 0)
    circuit_ent_2.append_gate(CNOTGate(), (0, 1))
    circuit_ent_2.append_gate(HGate(), 1)
    circuit_ent_2.append_gate(CNOTGate(), (1, 2))
    circuit_ent_2.append_gate(HGate(), 2)
    circuit_ent_2.append_gate(CNOTGate(), (2, 3))
    circuit_ent_2.append_gate(HGate(), 3)
    for q in range(4, 8, 2):
        circuit_ent_2.append_gate(HGate(), q)
        circuit_ent_2.append_gate(CNOTGate(), (q, q+1))
        circuit_ent_2.append_gate(HGate(), q+1)
    circuit_ent_2.append_gate(CNOTGate(), (3, 4))
    circuit_ent_2.append_gate(CNOTGate(), (4, 5))
    print(f"\nCircuit 2: Medium Entangled Circuit ({circuit_ent_2.num_operations} gates, {circuit_ent_2.num_qudits} qubits)")
    await compare_partitions(circuit_ent_2, block_size=3)

    # Test 3: Large Circuits
    print("\n=== Test 3: Large Circuits ===")
    # Circuit 1: Large Independent (19 gates, 10 qubits)
    circuit_large_1 = Circuit(10)
    for q in range(0, 10, 2):
        circuit_large_1.append_gate(HGate(), q)
        circuit_large_1.append_gate(CNOTGate(), (q, q+1))
        circuit_large_1.append_gate(HGate(), q+1)
    for q in range(0, 8, 2):
        circuit_large_1.append_gate(CNOTGate(), (q, q+2))
    print(f"\nCircuit 1: Large Independent Circuit ({circuit_large_1.num_operations} gates, {circuit_large_1.num_qudits} qubits)")
    await compare_partitions(circuit_large_1, block_size=3)

    # Circuit 2: Large with Cross-Partition Entanglement (25 gates, 12 qubits)
    circuit_large_2 = Circuit(12)
    for q in range(0, 12, 2):
        circuit_large_2.append_gate(HGate(), q)
        circuit_large_2.append_gate(CNOTGate(), (q, q+1))
        circuit_large_2.append_gate(HGate(), q+1)
    for q in range(0, 10, 2):
        circuit_large_2.append_gate(CNOTGate(), (q, q+2))
    circuit_large_2.append_gate(CNOTGate(), (2, 6))
    circuit_large_2.append_gate(CNOTGate(), (6, 10))
    print(f"\nCircuit 2: Large Circuit with Cross-Partition Entanglement ({circuit_large_2.num_operations} gates, {circuit_large_2.num_qudits} qubits)")
    await compare_partitions(circuit_large_2, block_size=3)

    # Test 4: Large Circuits with Big Sequences
    print("\n=== Test 4: Large Circuits with Big Sequences ===")
    # Circuit 1: Large with Big Sequences (22 gates, 6 qubits)
    circuit_seq_1 = Circuit(6)
    for _ in range(5):
        circuit_seq_1.append_gate(HGate(), 0)
        circuit_seq_1.append_gate(CNOTGate(), (0, 1))
    for _ in range(4):
        circuit_seq_1.append_gate(HGate(), 2)
        circuit_seq_1.append_gate(CNOTGate(), (2, 3))
    circuit_seq_1.append_gate(HGate(), 4)
    circuit_seq_1.append_gate(HGate(), 5)
    circuit_seq_1.append_gate(CNOTGate(), (1, 2))
    circuit_seq_1.append_gate(CNOTGate(), (3, 4))
    print(f"\nCircuit 1: Large Circuit with Big Sequences ({circuit_seq_1.num_operations} gates, {circuit_seq_1.num_qudits} qubits)")
    await compare_partitions(circuit_seq_1, block_size=6)

    # Circuit 2: Larger with Big Sequences (30 gates, 10 qubits)
    circuit_seq_2 = Circuit(10)
    for _ in range(6):
        circuit_seq_2.append_gate(HGate(), 0)
        circuit_seq_2.append_gate(CNOTGate(), (0, 1))
    for _ in range(5):
        circuit_seq_2.append_gate(HGate(), 2)
        circuit_seq_2.append_gate(CNOTGate(), (2, 3))
    for _ in range(4):
        circuit_seq_2.append_gate(HGate(), 4)
        circuit_seq_2.append_gate(CNOTGate(), (4, 5))
    circuit_seq_2.append_gate(CNOTGate(), (1, 2))
    circuit_seq_2.append_gate(CNOTGate(), (3, 4))
    circuit_seq_2.append_gate(CNOTGate(), (5, 6))
    print(f"\nCircuit 2: Larger Circuit with Big Sequences ({circuit_seq_2.num_operations} gates, {circuit_seq_2.num_qudits} qubits)")
    await compare_partitions(circuit_seq_2, block_size=6)

    # Test 5: Large Circuits with High Entanglement
    print("\n=== Test 5: Large Circuits with High Entanglement ===")
    # Circuit 1: Large Highly Entangled (48 gates, 20 qubits)
    circuit_high_ent_1 = Circuit(20)
    for _ in range(4):
        circuit_high_ent_1.append_gate(HGate(), 0)
        circuit_high_ent_1.append_gate(CNOTGate(), (0, 1))
        circuit_high_ent_1.append_gate(HGate(), 2)
        circuit_high_ent_1.append_gate(CNOTGate(), (2, 3))
    for _ in range(3):
        circuit_high_ent_1.append_gate(HGate(), 4)
        circuit_high_ent_1.append_gate(CNOTGate(), (4, 5))
        circuit_high_ent_1.append_gate(HGate(), 6)
        circuit_high_ent_1.append_gate(CNOTGate(), (6, 7))
    for _ in range(2):
        circuit_high_ent_1.append_gate(HGate(), 8)
        circuit_high_ent_1.append_gate(CNOTGate(), (8, 9))
        circuit_high_ent_1.append_gate(HGate(), 10)
        circuit_high_ent_1.append_gate(CNOTGate(), (10, 11))
    circuit_high_ent_1.append_gate(HGate(), 12)
    circuit_high_ent_1.append_gate(CNOTGate(), (12, 13))
    circuit_high_ent_1.append_gate(HGate(), 14)
    circuit_high_ent_1.append_gate(CNOTGate(), (14, 15))
    circuit_high_ent_1.append_gate(CNOTGate(), (1, 4))
    circuit_high_ent_1.append_gate(CNOTGate(), (5, 8))
    circuit_high_ent_1.append_gate(CNOTGate(), (9, 12))
    circuit_high_ent_1.append_gate(CNOTGate(), (13, 16))
    circuit_high_ent_1.append_gate(HGate(), 16)
    circuit_high_ent_1.append_gate(HGate(), 17)
    circuit_high_ent_1.append_gate(HGate(), 18)
    circuit_high_ent_1.append_gate(HGate(), 19)
    print(f"\nCircuit 1: Large Highly Entangled Circuit ({circuit_high_ent_1.num_operations} gates, {circuit_high_ent_1.num_qudits} qubits)")
    await compare_partitions(circuit_high_ent_1, block_size=6)

    # Circuit 2: Larger Highly Entangled (60 gates, 24 qubits)
    circuit_high_ent_2 = Circuit(24)
    for _ in range(5):
        circuit_high_ent_2.append_gate(HGate(), 0)
        circuit_high_ent_2.append_gate(CNOTGate(), (0, 1))
        circuit_high_ent_2.append_gate(HGate(), 2)
        circuit_high_ent_2.append_gate(CNOTGate(), (2, 3))
    for _ in range(4):
        circuit_high_ent_2.append_gate(HGate(), 4)
        circuit_high_ent_2.append_gate(CNOTGate(), (4, 5))
        circuit_high_ent_2.append_gate(HGate(), 6)
        circuit_high_ent_2.append_gate(CNOTGate(), (6, 7))
    for _ in range(3):
        circuit_high_ent_2.append_gate(HGate(), 8)
        circuit_high_ent_2.append_gate(CNOTGate(), (8, 9))
        circuit_high_ent_2.append_gate(HGate(), 10)
        circuit_high_ent_2.append_gate(CNOTGate(), (10, 11))
    for _ in range(2):
        circuit_high_ent_2.append_gate(HGate(), 12)
        circuit_high_ent_2.append_gate(CNOTGate(), (12, 13))
        circuit_high_ent_2.append_gate(HGate(), 14)
        circuit_high_ent_2.append_gate(CNOTGate(), (14, 15))
    circuit_high_ent_2.append_gate(HGate(), 16)
    circuit_high_ent_2.append_gate(CNOTGate(), (16, 17))
    circuit_high_ent_2.append_gate(HGate(), 18)
    circuit_high_ent_2.append_gate(CNOTGate(), (18, 19))
    circuit_high_ent_2.append_gate(CNOTGate(), (1, 4))
    circuit_high_ent_2.append_gate(CNOTGate(), (5, 8))
    circuit_high_ent_2.append_gate(CNOTGate(), (9, 12))
    circuit_high_ent_2.append_gate(CNOTGate(), (13, 16))
    circuit_high_ent_2.append_gate(CNOTGate(), (17, 20))
    circuit_high_ent_2.append_gate(HGate(), 20)
    circuit_high_ent_2.append_gate(HGate(), 21)
    circuit_high_ent_2.append_gate(HGate(), 22)
    circuit_high_ent_2.append_gate(HGate(), 23)
    print(f"\nCircuit 2: Larger Highly Entangled Circuit ({circuit_high_ent_2.num_operations} gates, {circuit_high_ent_2.num_qudits} qubits)")


    print(f"\nCircuit 2: Larger Highly Entangled Circuit ({circuit_high_ent_2.num_operations} gates, {circuit_high_ent_2.num_qudits} qubits)")
    await compare_partitions(circuit_high_ent_2, block_size=6)

if __name__ == "__main__":
    asyncio.run(main())