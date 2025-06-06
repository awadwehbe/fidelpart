from collections import defaultdict
import multiprocessing
import random
from typing import Any, Dict, List, Set, Tuple

from bqskit.ir.operation import Operation
multiprocessing.set_start_method('spawn')  # Add this before other imports

from bqskit.ir.circuit import Circuit
#from bqskit.ir.gates import GateSet
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.util.extend import ExtendBlockSizePass
from bqskit.passes.partitioning.hyp import EnhancedHypergraphPartitionPass
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
from bqskit.runtime import default_manager_port, result
from bqskit.passes.util.log import LogPass
import logging
import time
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.compiler import Compiler
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.gates import HGate, CNOTGate, CCXGate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time
from bqskit.ir.gates import HGate, CNOTGate, CCXGate
from collections import defaultdict
import logging



# --- Utility Functions ---

from typing import List, Set, Dict, Optional
from collections import defaultdict
import time
import logging
from dataclasses import dataclass
import networkx as nx  # For hardware graph representation

# Assume necessary imports for Circuit, gates, and partitioning passes
logger = logging.getLogger(__name__)

# Hardware Topology Representation
@dataclass
class HardwareQubit:
    id: int
    error_rate: float

class HardwareGraph:
    def __init__(self, topology: str = "linear", num_qubits: int = 5):
        self.graph = nx.Graph()
        self.topology = topology
        self.num_qubits = num_qubits
        self._build_topology()
        
    def _build_topology(self):
        """Initialize the hardware connectivity graph"""
        for q in range(self.num_qubits):
            self.graph.add_node(q, qubit=HardwareQubit(id=q, error_rate=1e-3))
        
        if self.topology == "linear":
            for q in range(self.num_qubits - 1):
                self.graph.add_edge(q, q+1, weight=1)
        elif self.topology == "grid":
            side = int(self.num_qubits**0.5)
            for i in range(side):
                for j in range(side):
                    q = i * side + j
                    if j < side - 1:
                        self.graph.add_edge(q, q+1, weight=1)  # Right
                    if i < side - 1:
                        self.graph.add_edge(q, q+side, weight=1)  # Down
    
    def shortest_path(self, start: int, end: int) -> List[int]:
        """Find the shortest path between two qubits"""
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return []

# def get_qubits(partition: Circuit) -> Set[int]:
#     """Extract all qubits involved in a partition."""
#     qubits = set()
#     for op in partition:
#         if hasattr(op, 'location'):  # Handle both Operation and Gate types
#             qubits.update(op.location)
#         elif hasattr(op, 'qubits'):
#             qubits.update(op.qubits)
#     return qubits


from itertools import combinations
from typing import List, Dict, Set, Optional

# --- MODIFIED compute_physical_swaps_between_partitions ---
def compute_physical_swaps_between_partitions(
    partitions: List[Circuit],
    qubit_maps: List[Dict[int, int]],
    global_cut_qubits_original_logical: Set[int],
    hardware: Optional[HardwareGraph] = None,
    verbose: bool = False
) -> List[int]: # Now returns a list of integers for per-partition comm SWAPs
    """
    Compute estimated SWAP gates needed, attributed to only one partition per edge.
    
    Args:
        partitions: List of circuit partitions.
        qubit_maps: List of mappings from original circuit physical qubit to
                    partition's local logical qubit.
        global_cut_qubits_original_logical: Set of original circuit logical qubit indices
                                            that are shared between partitions, used for teleportation tracking.
        hardware: Optional hardware graph for routing constraints.
        verbose: Whether to print debug info.
    
    Returns:
        List[int]: Where `result_list[i]` is the total estimated communication SWAPs attributed to partition `i`.
                   The sum of this list will directly equal the total unique communication SWAPs.
    """
    num_partitions = len(partitions)
    # Initialize with 0 to ensure integer arithmetic
    per_partition_comm_swaps = [0] * num_partitions 

    qubit_swap_count = {q: 0 for q in global_cut_qubits_original_logical}

    if verbose:
        print("\n--- Computing Physical SWAPs Between Partitions (Attributing to one partition per edge) ---")
        print(f"Initial `global_cut_qubits_original_logical` (for teleportation tracking): {global_cut_qubits_original_logical}")

    cut_info = get_physical_cut_qubits(partitions, qubit_maps)
    pairwise_physical_cut_qubits = cut_info['pairwise_cut_qubits']

    if verbose:
        print(f"\n`get_physical_cut_qubits` returned:")
        print(f"  Global Physical Cut Qubits: {cut_info['global_cut_qubits']}")
        print(f"  Pairwise Physical Cut Qubits: {pairwise_physical_cut_qubits}")

    for (p1_idx, p2_idx), shared_original_phys_qubits in pairwise_physical_cut_qubits.items():
        swaps_for_current_edge = 0
        if verbose:
            print(f"\n--- Analyzing Partition Pair ({p1_idx}, {p2_idx}) ---")
            print(f"  Shared Original Physical Qubits: {shared_original_phys_qubits}")

        for original_phys_q in shared_original_phys_qubits:
            local_logical_q_in_p1 = qubit_maps[p1_idx].get(original_phys_q)
            local_logical_q_in_p2 = qubit_maps[p2_idx].get(original_phys_q)

            if verbose:
                print(f"  Considering Original Physical Qubit {original_phys_q}:")
                print(f"    In Partition {p1_idx}, it maps to Local Logical Qubit: {local_logical_q_in_p1}")
                print(f"    In Partition {p2_idx}, it maps to Local Logical Qubit: {local_logical_q_in_p2}")

            if local_logical_q_in_p1 is None or local_logical_q_in_p2 is None:
                if verbose:
                    print(f"    Warning: Original Physical Qubit {original_phys_q} not consistently mapped in qubit_map for partitions {p1_idx} or {p2_idx}. Skipping this qubit for SWAP calculation.")
                continue

            if local_logical_q_in_p1 == local_logical_q_in_p2:
                if verbose:
                    print(f"    Physical Qubit {original_phys_q} maps to the **same Logical Qubit ({local_logical_q_in_p1})** in Partition {p1_idx} and {p2_idx}. **No SWAP needed for alignment.**")
                continue

            if verbose:
                print(f"    Physical Qubit {original_phys_q} maps to **Logical Qubit {local_logical_q_in_p1} in P{p1_idx}** and **Logical Qubit {local_logical_q_in_p2} in P{p2_idx}**. This indicates a need for SWAP to align the logical states.")

            if local_logical_q_in_p1 in qubit_swap_count:
                qubit_swap_count[local_logical_q_in_p1] += 1
                if verbose:
                    print(f"    Tracking for original circuit logical qubit {local_logical_q_in_p1}: Current SWAP count for this qubit is {qubit_swap_count[local_logical_q_in_p1]}.")
                
                if qubit_swap_count[local_logical_q_in_p1] > 3 and random.random() < 0.6:
                    if verbose:
                        print(f"    Teleportation condition met for original circuit logical qubit {local_logical_q_in_p1} (on original physical qubit {original_phys_q}) between P{p1_idx} and P{p2_idx}. Skipping SWAP count for this instance.")
                    continue
            elif verbose:
                print(f"    Note: Logical qubit {local_logical_q_in_p1} (derived from original physical qubit {original_phys_q}) is not in `global_cut_qubits_original_logical`, so teleportation tracking is skipped for this instance.")

            path_swaps = 1 
            
            if hardware:
                if verbose:
                    print(f"    Hardware graph provided. Assuming {path_swaps} SWAP for logical re-assignment on shared Original Physical Qubit {original_phys_q}.")
            else:
                if verbose:
                    print(f"    No hardware graph provided. Assuming {path_swaps} SWAP for logical re-assignment on shared Original Physical Qubit {original_phys_q}.")
            
            swaps_for_current_edge += path_swaps
            if verbose:
                print(f"    **Original Physical Qubit {original_phys_q} contributes {path_swaps} SWAP(s) to the total for this edge.**")

        if verbose:
            print(f"\n  Total SWAPs for Partition Pair ({p1_idx}, {p2_idx}): {swaps_for_current_edge}")
        
        # AGGREGATE SWAPS FOR P1_IDX ONLY:
        # We assign the full cost of the edge to the first partition in the pair.
        if 0 <= p1_idx < num_partitions:
            per_partition_comm_swaps[p1_idx] += swaps_for_current_edge
        # We do NOT add to p2_idx here, to ensure the sum of per_partition_comm_swaps
        # equals the total sum of unique edge costs.

    if verbose:
        print("\n--- SWAP Computation Complete ---")
        print(f"Final `per_partition_comm_swaps` list (each element is total comm SWAPs attributed to that partition): {per_partition_comm_swaps}")
        print(f"Final `qubit_swap_count` (for teleportation, tracking original logical qubits): {qubit_swap_count}")

    return per_partition_comm_swaps
    
    

from collections import defaultdict
from typing import List, Set, Dict, Any

def get_physical_qubits(partition: Circuit, qubit_map: Dict[int, int]) -> Set[int]:
    """
    Extract all physical qubit indices used in a partition.
    
    Args:
        partition: The circuit partition with logical qubit indices
        qubit_map: Mapping from physical to logical qubit indices
        
    Returns:
        Set of physical qubit indices used in this partition
    """
    # Inverse the qubit map to go from logical to physical
    inverse_map = {logical: physical for physical, logical in qubit_map.items()}
    
    physical_qubits = set()
    for op in partition:
        qubits = op.location if hasattr(op, 'location') else op.qubits
        # Convert each logical qubit to its physical qubit
        for q in qubits:
            if q in inverse_map:
                physical_qubits.add(inverse_map[q])
            else:
                print(f"Warning: Logical qubit {q} not found in qubit_map for partition")
    
    return physical_qubits

def get_physical_cut_qubits(
    partitions: List[Circuit], 
    qubit_maps: List[Dict[int, int]]
) -> Dict[str, Any]:
    """
    Identify physical cut qubits between partitions, accounting for qubit mappings.
    
    Args:
        partitions: List of circuit partitions with logical qubit indices
        qubit_maps: List of mappings from physical to logical qubit indices for each partition
        
    Returns:
        Dictionary with global cut qubits and pairwise cut qubits (all in physical indices)
    """
    if len(partitions) != len(qubit_maps):
        raise ValueError(f"Number of partitions ({len(partitions)}) must match number of qubit maps ({len(qubit_maps)})")
    
    # Track which partitions each physical qubit appears in
    qubit_partitions = defaultdict(set)
    
    # For each partition, get its physical qubits and track which partition they appear in
    for i, (partition, qubit_map) in enumerate(zip(partitions, qubit_maps)):
        physical_qubits = get_physical_qubits(partition, qubit_map)
        for q in physical_qubits:
            qubit_partitions[q].add(i)
    
    # Global cut qubits (physical qubits in >1 partition)
    global_cut = {q for q, parts in qubit_partitions.items() if len(parts) > 1}
    
    # Pairwise cut qubits between partitions
    pairwise_cut = {}
    num_partitions = len(partitions)
    
    for i in range(num_partitions):
        for j in range(i + 1, num_partitions):
            # Find physical qubits shared between partition i and j
            shared_qubits = set()
            for q, parts in qubit_partitions.items():
                if i in parts and j in parts:
                    shared_qubits.add(q)
            if shared_qubits:
                pairwise_cut[(i,j)] = shared_qubits
    
    return {
        'global_cut_qubits': global_cut,
        'pairwise_cut_qubits': pairwise_cut
    }

def analyze_partition(
    partition: Circuit,
    swap_count: int,
    include_swaps: bool = False
) -> Dict:
    h_count = cnot_count = actual_swaps = 0
    for op in partition:
        gate = op.gate if hasattr(op, 'gate') else op
        if isinstance(gate, HGate):
            h_count += 1
        elif isinstance(gate, CNOTGate):
            cnot_count += 1
        elif isinstance(gate, CCXGate):  # Toffoli (CCX) decomposition
            cnot_count += 6  # Accurate CCX decomposition requires 6 CNOTs
        elif include_swaps and isinstance(gate, SwapGate):
            actual_swaps += 1
    return {
        'h_count': h_count,
        'cnot_count': cnot_count,
        'swap_count': actual_swaps + swap_count
    }

def compute_fidelity(
    partition_info: Dict,
    epsilon_h: float = 1e-3,
    epsilon_cnot: float = 1e-2
) -> Dict:
    """
    Compute fidelity using:
    - 1 SWAP = 3 CNOT gates
    - No separate ε_swap - SWAPs modeled as CNOTs
    """
    effective_cnots = partition_info['cnot_count'] + 3 * partition_info['swap_count']
    fidelity = (1 - epsilon_h) ** partition_info['h_count'] * (1 - epsilon_cnot) ** effective_cnots
    return {'fidelity': fidelity, 'error_rate': 1 - fidelity}


async def compare_partitions_physical(
    circuit: Circuit,
    block_size: int = 5,
    num_workers: int = 8,
    hardware_topology: str = "linear"
) -> Dict:
    hardware = HardwareGraph(topology=hardware_topology, num_qubits=circuit.num_qudits)
    print("\n=== Partitioning Circuits ===")
    
    start = time.time()
    # Get partitions with GLOBAL indices from QuickPartitioner
    quick_subs_global = partition_with_quick(circuit, block_size, hardware) 
    
    # Derive LOCAL circuits and their {global: local} maps
    quick_subs_local, quick_maps_list = derive_and_remap_for_quick(quick_subs_global, hardware) # Corrected unpacking
    
    # >>> Add this debug print to confirm <<<
    print(f"\n>>> DEBUG: AFTER derive_and_remap: quick_subs_local = {len(quick_subs_local)}, quick_maps_list = {len(quick_maps_list)}\n")

    quick_results = _calculate_physical_partition_metrics(quick_subs_local, quick_maps_list, hardware, "Quick") # Use correct variables
    quick_results['start_time'] = start
    
    # ... (Hypergraph part) ...
    hyper_subs_local, hyper_maps_list = await partition_with_hypergraph(circuit.copy(), block_size, num_workers)
    
    
    print(f"\n>>> DEBUG: AFTER partition_with_hypergraph: hyper_subs_local = {len(hyper_subs_local)}, hyper_maps_list = {len(hyper_maps_list)}\n")

    hyper_results = _calculate_physical_partition_metrics(hyper_subs_local, hyper_maps_list, hardware, "Hypergraph")
    hyper_results['start_time'] = start
    
    # Pass the correct local circuits to validation if needed
    _validate_gate_counts(circuit, quick_subs_local, hyper_subs_local) 
    return _format_comparison_results(circuit, quick_results, hyper_results)

def _validate_gate_counts(
    original: Circuit,
    quick_partitions: List[Circuit],
    hyper_partitions: List[Circuit]
) -> None:
    """Validate no gates were lost during partitioning."""
    
    def count_gates(circuit):
        return {
            'H': sum(1 for op in circuit if isinstance(op.gate, HGate)),
            'CNOT': sum(1 for op in circuit if isinstance(op.gate, CNOTGate)),
            'CCX': sum(1 for op in circuit if isinstance(op.gate, CCXGate)),
            'SWAP': sum(1 for op in circuit if isinstance(op.gate, SwapGate))
        }

    # Ensure all partitions use the same number of qudits
    max_qudits = max(p.num_qudits for p in quick_partitions + hyper_partitions)
    from bqskit.ir.circuit import Circuit as BQSKITCircuit

    def resize_circuit(circuit: BQSKITCircuit, new_size: int) -> BQSKITCircuit:
        """
        Resize a BQSKit Circuit to a new number of qudits.
        
        If new_size > current size: Adds empty qudits (operations stay the same).
        If new_size < current size: Truncates qubit indices beyond new_size.
        """
        if new_size < 1:
            raise ValueError("new_size must be at least 1")
        
        new_circuit = BQSKITCircuit(new_size)

        for op in circuit:
            # Only add operations whose qubits are within the new size
            valid_qubits = [q for q in op.location if q < new_size]
            if len(valid_qubits) == len(op.location):
                # All qubits are valid, append directly
                new_circuit.append(op)
            elif valid_qubits:
                # Some qubits are out of bounds — try to create a compatible operation
                try:
                    new_op = Operation(op.gate, valid_qubits)
                    new_circuit.append(new_op)
                except Exception as e:
                    print(f"Skipping operation due to resizing error: {e}")
        
        return new_circuit
    # Resize partitions to uniform qudit count
    resized_quick = [resize_circuit(p, max_qudits) for p in quick_partitions]
    resized_hyper = [resize_circuit(p, max_qudits) for p in hyper_partitions]

    # Now safely combine
    quick_circuit_sum = sum(resized_quick, Circuit(max_qudits))
    hyper_circuit_sum = sum(resized_hyper, Circuit(max_qudits))

    original_counts = count_gates(original)
    quick_counts = count_gates(quick_circuit_sum)
    hyper_counts = count_gates(hyper_circuit_sum)

    for name, counts in [('Quick', quick_counts), ('Hypergraph', hyper_counts)]:
        for gate, count in counts.items():
            if count != original_counts.get(gate, 0):
                logger.warning(f"{name} partitioning mismatch for {gate}: {count} vs {original_counts.get(gate, 0)}")

import random # Needed for the teleportation heuristic
import time
from typing import List, Dict, Set, Optional

# Assuming Circuit and HardwareGraph are defined elsewhere or imported
# from .circuit import Circuit
# from .hardware_graph import HardwareGraph

def _calculate_physical_partition_metrics(
    partitions: List[Circuit], 
    qubit_maps: List[Dict[int, int]],
    hardware: Optional[HardwareGraph] = None, 
    method_name: str = "Method"
) -> Dict:
    """
    Calculate partition metrics and print detailed gate information with qubit usage summary.
    
    Args:
        partitions: List of circuit partitions
        qubit_maps: List of mappings from original circuit qubits to partition qubits {original_q: trimmed_q}
        hardware: Optional hardware graph for routing constraints
        method_name: Name of the partitioning method for logging
        
    Returns:
        Dictionary of partition metrics
    """
    import time
    # Get cut qubits analysis based on original circuit qubits
    cut_data = get_physical_cut_qubits(partitions, qubit_maps)
    global_cut_qubits = cut_data['global_cut_qubits']
    
    print(f"\n--- {method_name} Partition Analysis ---")
    print(f"Global cut qubits (original circuit indices): {global_cut_qubits}")
    print("\nPairwise cut qubits (original circuit indices):")
    for (p1, p2), qubits in cut_data['pairwise_cut_qubits'].items():
        print(f"Partitions {p1} ↔ {p2}: {qubits}")
    
    # Compute SWAPs based on original circuit qubit dependencies
    swap_counts = compute_physical_swaps_between_partitions(
        partitions, 
        qubit_maps,
        global_cut_qubits, 
        hardware
    )
    total_swaps = sum(swap_counts)
    """
    compute_physical_swaps_between_partitions now returns a list where each element is the sum of communication SWAPs for that specific partition. To get the true global total of unique communication SWAPs, you sum this list and divide by 2.
    
    """
    print(f"Total SWAP gates needed: {total_swaps}")
    
    # Print hardware context
    if hardware:
        all_original_qubits = set(hardware.graph.nodes)
        print(f"Hardware Topology: {hardware.topology}")
        print(f"Total Qubits in Hardware: {hardware.num_qubits} (Indices: {sorted(all_original_qubits)})")
    else:
        all_original_qubits = set()
        for partition, qubit_map in zip(partitions, qubit_maps):
            all_original_qubits.update(qubit_map.keys())
        print(f"No hardware graph provided; Total Qubits in Hardware: {len(all_original_qubits)} (Indices: {sorted(all_original_qubits)})")
    
    metrics = []
    total_fidelity = 1.0
    block_size = 5  # Match derive_qubit_maps_for_quick
    
    for i, (partition, qubit_map) in enumerate(zip(partitions, qubit_maps)):
        # Create inverse map for partition to original circuit conversion
        inverse_map = {trimmed_q: original_q for original_q, trimmed_q in qubit_map.items()}
        
        # Collect gate details and partition qubits
        gate_details = []
        partition_qubits = set()
        original_qubits = set(qubit_map.keys())  # Original circuit qubits used in this partition
        for op in partition:
            gate_type = op.gate.__class__.__name__ if hasattr(op, 'gate') else op.__class__.__name__
            qubits = op.location if hasattr(op, 'location') else op.qubits
            # Cap qubit indices at block_size-1 to match qubit_map
            capped_qubits = tuple(min(q, block_size - 1) for q in qubits)
            partition_qubits.update(capped_qubits)
            # Map partition qubits to original circuit qubits
            original_circuit_qubits = tuple(inverse_map.get(q, q) for q in capped_qubits)
            gate_str = f"{gate_type}@(Partition Qubits: {', '.join(map(str, capped_qubits))}; Original Circuit Qubits: {', '.join(map(str, original_circuit_qubits))})"
            gate_details.append(gate_str)
        
        # Distinguish active and assigned qubits
        active_original_qubits = set(inverse_map.get(q, q) for q in partition_qubits)
        assigned_original_qubits = set(qubit_map.keys())
        unassigned_original_qubits = all_original_qubits - assigned_original_qubits if all_original_qubits else set()
        assigned_but_unused = assigned_original_qubits - active_original_qubits
        
        # Include both inherent SWAPs and estimated communication SWAPs
        swap_count = swap_counts[i] if i < len(swap_counts) else 0
        info = analyze_partition(partition, swap_count)
        fidelity_data = compute_fidelity(info)
        total_fidelity *= fidelity_data['fidelity']
        
        # Print partition details
        print(f"\nPartition {i}:")
        print(f"- Original Circuit Qubits Used: {len(original_qubits)} (Indices: {sorted(original_qubits)}) "
              f"(Qubits from the original circuit used in this partition)")
        print(f"- Partition Qubits: {len(partition_qubits)} (Indices: {sorted(partition_qubits)}) "
              f"(Qubit indices used in this partition's gate operations)")
        if assigned_but_unused:
            print(f"- Assigned but Unused Original Circuit Qubits: {sorted(assigned_but_unused)}")
        if unassigned_original_qubits:
            print(f"- Unassigned Original Circuit Qubits: {sorted(unassigned_original_qubits)}")
        print(f"- Qubit Map (Original Circuit Qubits(global logical qubits) → Partition Qubits(linear chain physical mapping)): {qubit_map} "
              f"(Maps qubits from the original circuit to qubit indices in this partition)")
        # print(f"- Inverse Map (Partition Qubits → Original Circuit Qubits): {inverse_map} "
        #       f"(Maps partition qubit indices to qubits from the original circuit)")
        print(f"- Gates:")
        for idx, gate in enumerate(gate_details, 1):
            print(f"  {idx}. {gate}")
        print(f"- Number of Gates: {partition.num_operations}")
        print(f"- Depth: {partition.depth}")
        print(f"- H gates: {info['h_count']}")
        print(f"- CNOT gates: {info['cnot_count']}")
        swap_count = swap_counts[idx] if idx < len(swap_counts) else 0
        print(f"- SWAP gates: {info['swap_count']}")
        print(f"- Fidelity: {fidelity_data['fidelity']:.6f}")
        print(f"- Error rate: {fidelity_data['error_rate']:.6f}")
        
        metrics.append({
            'partition': i,
            'num_gates': partition.num_operations,
            'depth': partition.depth,
            'h_count': info['h_count'],
            'cnot_count': info['cnot_count'],
            'swap_count': info['swap_count'],
            **fidelity_data
        })
    
    return {
        'partitions': partitions,
        'metrics': metrics,
        'total_fidelity': total_fidelity,
        'total_swaps': total_swaps,
        'cut_qubits': global_cut_qubits,
        'pairwise_cut_qubits': cut_data['pairwise_cut_qubits'],
        'time': time.time(),
        'start_time': time.time()
    }

def _format_comparison_results(
    original: Circuit,
    quick: Dict,
    hyper: Dict
) -> Dict:
    """Format and print comparison results."""
    print("\n=== Final Comparison ===")
    quick_time = quick['time'] - quick.get('start_time', quick['time'])
    hyper_time = hyper['time'] - hyper.get('start_time', hyper['time'])
    print(f"\nTime (Quick vs Hyper): {quick_time:.3f}s vs {hyper_time:.3f}s")
    
    # Depth comparison
    quick_depths = [p.depth for p in quick['partitions']]
    hyper_depths = [p.depth for p in hyper['partitions']]
    print(f"Max Depth: Quick={max(quick_depths)}, Hyper={max(hyper_depths)}")
    
    # Fidelity comparison
    print("\nFidelity Results:")
    print(f"Quick: {quick['total_fidelity']:.6f} (Error: {1 - quick['total_fidelity']:.6f})")
    print(f"Hyper: {hyper['total_fidelity']:.6f} (Error: {1 - hyper['total_fidelity']:.6f})")
    
    # SWAP overhead
    print(f"\nSWAP Gates: Quick={quick['total_swaps']}, Hyper={hyper['total_swaps']}")
    print(f"Cut Qubits: Quick={len(quick['cut_qubits'])}, Hyper={len(hyper['cut_qubits'])}")
    
    return {
        'quick': quick,
        'hyper': hyper,
        'original_gates': original.num_operations
    }

# Your existing partition_with_* functions remain unchanged
# [Keep all your existing partition_with_quick, partition_with_hypergraph functions here]
# [They should work as-is with the new analysis functions]
from bqskit.ir.circuit import Circuit, CircuitGate, Operation
from bqskit.compiler import Compiler, PassData
from bqskit.passes import QuickPartitioner

from typing import List, Optional, Dict
import logging

# Assuming HardwareGraph is defined elsewhere
# Configure logging if needed
logger = logging.getLogger(__name__)

def partition_with_quick(
    circuit: Circuit, 
    block_size: int = 5, 
    hardware: Optional[HardwareGraph] = None
) -> List[Circuit]:
    """
    Partitions a circuit using QuickPartitioner and returns a list of 
    sub-circuits where each gate uses its original GLOBAL qubit indices.

    Args:
        circuit: The original circuit to partition.
        block_size: The target block size for QuickPartitioner.
        hardware: Optional hardware graph.

    Returns:
        A list of Circuit objects, each representing a partition, with
        gates operating on global qubit indices.
    """
    from bqskit.passes.mapping.routing import PAMRoutingPass # Keep if needed, though not used here
    print("Circuit structure:")
    for i, op in enumerate(circuit.operations()):
        print(f"Gate {i:2}: {op.gate.name} on {op.location}")
    workflow = [QuickPartitioner(block_size)]
    data = PassData(circuit)
    # Define a basic gate set BQSKit can understand
    data._gate_set = CustomGateSet({HGate(), CNOTGate()}) 
    if hardware:
        data['hardware_graph'] = hardware.graph

    circuit_copy = circuit.copy()
    compiler = Compiler(num_workers=8)
    partitions = []

    try:
        print("Starting QuickPartitioner compilation...")
        # Compile the circuit, which results in a circuit of CircuitGates
        compiled_circuit = compiler.compile(circuit_copy, workflow, data=data)
        print("QuickPartitioner compilation finished.")

        # Iterate through the compiled circuit, expecting CircuitGates
        for op in compiled_circuit:
            if isinstance(op.gate, CircuitGate):
                subcircuit = op.gate._circuit
                global_map_list = op.location  # This is the key: maps local to global

                # Create a new circuit for this partition. 
                # Use the original circuit's width to hold global indices.
                new_partition_circuit = Circuit(circuit.num_qudits) 
                
                # Rebuild the subcircuit, translating local indices to global
                for gate_op in subcircuit:
                    local_indices = gate_op.location
                    
                    # Translate local indices back to global indices
                    try:
                        global_indices = tuple(global_map_list[local_q] for local_q in local_indices)
                    except IndexError:
                        logger.error(f"IndexError during local-to-global mapping!")
                        logger.error(f"  CircuitGate location: {global_map_list}")
                        logger.error(f"  Sub-operation local location: {local_indices}")
                        continue # Skip this gate or handle error appropriately

                    # Add the gate with its GLOBAL indices
                    new_partition_circuit.append(Operation(gate_op.gate, global_indices, gate_op.params))
                    
                partitions.append(new_partition_circuit)
            
            elif isinstance(op, Operation):
                 # This handles any gates *not* put into a CircuitGate.
                 # Decide how to handle these: often, they might be ignored,
                 # or added to a separate 'remainder' partition, or indicate
                 # that partitioning didn't fully segment the circuit.
                 # For now, we'll create a single-gate partition for it.
                 logger.warning(f"Found a non-CircuitGate operation: {op}. Creating single-op partition.")
                 single_op_partition = Circuit(circuit.num_qudits)
                 single_op_partition.append(op) # It already has global indices
                 partitions.append(single_op_partition)

            else:
                logger.warning(f"Unexpected item in compiled circuit: {type(op)}. Skipping.")

        print(f"QuickPartitioner resulted in {len(partitions)} partitions.")
        print(f"partition_with_quick is returning {len(partitions)} partitions.")
        for i, p in enumerate(partitions):
            print(f"  Partition {i}: {p.num_qudits} qubits, {p.num_operations} gates.")
            active_q = set()
            for op in p:
                active_q.update(op.location)
            print(f"    Global Qubits Found: {sorted(list(active_q))}")
        return partitions

    except Exception as e:
        print(f"Error during QuickPartitioner processing: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        compiler.close()

async def partition_with_quick_manual(circuit: Circuit, block_size: int = 3) -> List[Circuit]:
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

async def partition_with_hypergraph(
    circuit: Circuit, 
    block_size: int = 5, 
    num_workers: int = 8
) -> Tuple[List[Circuit], List[Dict[int, int]]]:
    """Ensure we always return Circuit objects"""
    hypergraph_pass = EnhancedHypergraphPartitionPass(block_size=block_size, num_workers=num_workers)
    data = {}
    await hypergraph_pass.run(circuit, data)
    
    # Convert any Operations to Circuits
    partitions = []
    for p in data.get('partitions', []):
        if isinstance(p, Operation):
            new_circuit = Circuit(circuit.num_qudits)
            new_circuit.append(p)
            partitions.append(new_circuit)
        else:
            partitions.append(p)
    
    return partitions, data.get('qubit_maps', [])

from typing import List, Optional
import random

from typing import List, Dict, Set, Optional
from bqskit.ir.circuit import Circuit 
# Assuming HardwareGraph is defined and imported
# Assuming logging is configured

def derive_and_remap_for_quick(
    partitions_global: List[Circuit],
    hardware: Optional[HardwareGraph] = None
) -> Tuple[List[Circuit], List[Dict[int, int]]]:

    print(f"\n--- Inside derive_and_remap_for_quick ---") # 1. Check entry
    print(f"Received {len(partitions_global)} partitions.")

    qubit_maps = []
    partitions_local = []
    # ... (available_qubits setup) ...

    # Check if partitions_global is actually a list
    if not isinstance(partitions_global, list):
        print("ERROR: Input is not a list!")
        return [], []

    for i, partition in enumerate(partitions_global):
        print(f"\n  Processing partition index {i}...") # 2. Check each iteration

        active_global_qubits = set()
        try:
            for op in partition:
                active_global_qubits.update(op.location)
            print(f"    Found active global: {sorted(list(active_global_qubits))}") # 3. Check active Qs
        except Exception as e:
            print(f"    ERROR finding active qubits: {e}")
            active_global_qubits = set() # Assume empty on error

        if not active_global_qubits:
            print("    Partition seems empty, adding empty map/circuit.") # 4. Check empty handling
            qubit_maps.append({})
            partitions_local.append(Circuit(0))
            continue

        # ... (Sort global qubits) ...
        sorted_global_qubits = sorted(list(active_global_qubits))

        # ... (Create physical_qubit_map) ...
        physical_qubit_map = {
            global_q: local_q
            for local_q, global_q in enumerate(sorted_global_qubits)
        }
        print(f"    Created map (size {len(physical_qubit_map)}): {physical_qubit_map}") # 5. Check map creation

        # ... (Build the trimmed/local circuit) ...
        trimmed = Circuit(len(active_global_qubits))
        try:
            for op in partition:
                local_indices = [physical_qubit_map[global_q] for global_q in op.location]
                trimmed.append(Operation(op.gate, local_indices, op.params))
            print(f"    Created local circuit (size {trimmed.num_qudits}, {trimmed.num_operations} gates).") # 6. Check circuit rebuild
        except Exception as e:
            print(f"    ERROR rebuilding local circuit: {e}")
            trimmed = Circuit(0) # Add empty on error

        qubit_maps.append(physical_qubit_map)
        partitions_local.append(trimmed)
        print(f"    Appended. Map list size: {len(qubit_maps)}, Partition list size: {len(partitions_local)}") # 7. Check append

    print(f"--- Exiting derive_and_remap_for_quick ---") # 8. Check exit
    print(f"Returning {len(partitions_local)} circuits and {len(qubit_maps)} maps.")

    return partitions_local, qubit_maps

# Your invocation would then look like:
# quick_subs_global = partition_with_quick(circuit, block_size, hardware)
# quick_subs_local, quick_maps = derive_and_remap_for_quick(quick_subs_global, hardware)
# quick_results = _calculate_physical_partition_metrics(quick_subs_local, quick_maps, hardware, "Quick")
    

# Add to the end of compare_partitioning.py
async def main():
    from bqskit.ir.gates import HGate, CNOTGate
    # # Set random seed for reproducibility (optional, remove for different randomizations)
    random.seed(42)

    # Assuming a custom quantum circuit library with Circuit, HGate, and CNOTGate
    # Create a circuit with 6 qubits (Q_0 to Q_5)
    circuit_ind_1 = Circuit(6)

        # Q_0-Q_1 gates (original pair, now randomized for CNOTs)
    circuit_ind_1.append_gate(HGate(), 0)          # H_1 on Q_0
    # CNOT_2: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_2: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 0)          # H_3 on Q_0
    # CNOT_4: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_4: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 0)          # H_5 on Q_0
    # CNOT_6: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_6: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 1)          # H_7 on Q_1

    # Q_2-Q_3 gates (original pair, now randomized for CNOTs)
    circuit_ind_1.append_gate(HGate(), 2)          # H_8 on Q_2
    # CNOT_9: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_9: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 2)          # H_10 on Q_2
    # CNOT_11: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_11: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 2)          # H_12 on Q_2
    # CNOT_13: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_13: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 3)          # H_14 on Q_3

    # Q_4-Q_5 gates (original pair, now randomized for CNOTs)
    circuit_ind_1.append_gate(HGate(), 4)          # H_15 on Q_4
    # CNOT_16: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_16: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 4)          # H_17 on Q_4
    # CNOT_18: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_18: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 4)          # H_19 on Q_4
    # CNOT_20: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_20: control Q_{}, target Q_{}
    circuit_ind_1.append_gate(HGate(), 4)          # H_21 on Q_4
    # CNOT_22: Random control and target
    control, target = random.sample(range(6), 2)
    circuit_ind_1.append_gate(CNOTGate(), (control, target))  # CNOT_22: control Q_{}, target Q_{}
    #await compare_partitions_physical(circuit_ind_1, block_size=4)
    



    # Create a 10-qubit quantum circuit

    # Create a new 10-qubit circuit
    circuit_M = Circuit(10)

    # Add Hadamard gates and CNOTs in a highly entangled pattern
    # Qubit 0
    circuit_M.append_gate(HGate(), 0)
    circuit_M.append_gate(CNOTGate(), [0, 1])
    circuit_M.append_gate(CNOTGate(), [0, 2])
    circuit_M.append_gate(CNOTGate(), [0, 3])
    circuit_M.append_gate(CNOTGate(), [0, 4])
    circuit_M.append_gate(CNOTGate(), [0, 5])
    circuit_M.append_gate(CNOTGate(), [0, 6])
    circuit_M.append_gate(CNOTGate(), [0, 7])
    circuit_M.append_gate(CNOTGate(), [0, 8])
    circuit_M.append_gate(CNOTGate(), [0, 9])

    # Qubit 1
    circuit_M.append_gate(HGate(), 1)
    circuit_M.append_gate(CNOTGate(), [1, 2])
    circuit_M.append_gate(CNOTGate(), [1, 3])
    circuit_M.append_gate(CNOTGate(), [1, 4])
    circuit_M.append_gate(CNOTGate(), [1, 5])
    circuit_M.append_gate(CNOTGate(), [1, 6])
    circuit_M.append_gate(CNOTGate(), [1, 7])
    circuit_M.append_gate(CNOTGate(), [1, 8])
    circuit_M.append_gate(CNOTGate(), [1, 9])

    # Qubit 2
    circuit_M.append_gate(HGate(), 2)
    circuit_M.append_gate(CNOTGate(), [2, 3])
    circuit_M.append_gate(CNOTGate(), [2, 4])
    circuit_M.append_gate(CNOTGate(), [2, 5])
    circuit_M.append_gate(CNOTGate(), [2, 6])
    circuit_M.append_gate(CNOTGate(), [2, 7])
    circuit_M.append_gate(CNOTGate(), [2, 8])
    circuit_M.append_gate(CNOTGate(), [2, 9])

    # Qubit 3
    circuit_M.append_gate(HGate(), 3)
    circuit_M.append_gate(CNOTGate(), [3, 4])
    circuit_M.append_gate(CNOTGate(), [3, 5])
    circuit_M.append_gate(CNOTGate(), [3, 6])
    circuit_M.append_gate(CNOTGate(), [3, 7])
    circuit_M.append_gate(CNOTGate(), [3, 8])
    circuit_M.append_gate(CNOTGate(), [3, 9])

    # Qubit 4
    circuit_M.append_gate(HGate(), 4)
    circuit_M.append_gate(CNOTGate(), [4, 5])
    circuit_M.append_gate(CNOTGate(), [4, 6])
    circuit_M.append_gate(CNOTGate(), [4, 7])
    circuit_M.append_gate(CNOTGate(), [4, 8])
    circuit_M.append_gate(CNOTGate(), [4, 9])

    # Qubit 5
    circuit_M.append_gate(HGate(), 5)
    circuit_M.append_gate(CNOTGate(), [5, 6])
    circuit_M.append_gate(CNOTGate(), [5, 7])
    circuit_M.append_gate(CNOTGate(), [5, 8])
    circuit_M.append_gate(CNOTGate(), [5, 9])

    # Qubit 6
    circuit_M.append_gate(HGate(), 6)
    circuit_M.append_gate(CNOTGate(), [6, 7])
    circuit_M.append_gate(CNOTGate(), [6, 8])
    circuit_M.append_gate(CNOTGate(), [6, 9])

    # Qubit 7
    circuit_M.append_gate(HGate(), 7)
    circuit_M.append_gate(CNOTGate(), [7, 8])
    circuit_M.append_gate(CNOTGate(), [7, 9])

    # Qubit 8
    circuit_M.append_gate(HGate(), 8)
    circuit_M.append_gate(CNOTGate(), [8, 9])

    # Qubit 9
    circuit_M.append_gate(HGate(), 9)

    # Save the circuit to a .bqk file
    
    
    #await compare_partitions_physical(circuit=circuit_M,block_size=6)
    
    
    # Print the circuit
    from bqskit.ir.gates import  CCXGate

    # Initialize a 24-qubit quantum circuit
    circuit_high_ent_2 = Circuit(24)

    # Layer 1: Initial entanglement using H and CNOT gates
    for i in range(0, 24, 2):
        circuit_high_ent_2.append_gate(HGate(), i)  # Apply Hadamard gate to even qubits
        circuit_high_ent_2.append_gate(CNOTGate(), (i, i + 1))  # Apply CNOT gate between consecutive qubits

    # Layer 2: Cross-qubit entanglement
    for i in range(0, 16, 4):  # Connect qubits in groups of 4
        circuit_high_ent_2.append_gate(CNOTGate(), (i, i + 4))  # CNOT between q[i] and q[i+4]
        circuit_high_ent_2.append_gate(CNOTGate(), (i + 1, i + 5))  # CNOT between q[i+1] and q[i+5]

    # Layer 3: Multi-qubit entanglement using Toffoli (CCX) gates
    for i in range(0, 20, 6):  # Apply Toffoli gates to create deeper entanglement
        circuit_high_ent_2.append_gate(CCXGate(), (i, i + 1, i + 2))  # Controlled-Controlled-NOT gate

    # Layer 4: Randomized cross-connections
    cross_connections = [(0, 10), (2, 14), (4, 18), (6, 20), (8, 22)]  # Example pairs
    for control, target in cross_connections:
        circuit_high_ent_2.append_gate(CNOTGate(), (control, target))

    # Layer 5: Global entanglement
    for i in range(0, 23):  # Pairwise CNOTs to entangle all qubits globally
        circuit_high_ent_2.append_gate(CNOTGate(), (i, i + 1))

    # Layer 6: Final Hadamard gates for superposition
    for i in range(24):
        circuit_high_ent_2.append_gate(HGate(), i)

    print(f"\nCircuit 2: Larger Highly Entangled Circuit ({circuit_high_ent_2.num_operations} gates, {circuit_high_ent_2.num_qudits} qubits)")


    print(f"\nCircuit 2: Larger Highly Entangled Circuit ({circuit_high_ent_2.num_operations} gates, {circuit_high_ent_2.num_qudits} qubits)")
    await compare_partitions_physical(circuit_high_ent_2, block_size=8)

    # Print the circuit (optional)
    print(circuit_high_ent_2)

if __name__ == "__main__":
    asyncio.run(main())