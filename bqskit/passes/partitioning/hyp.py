# Hardware-Aware Partitioning
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import logging
import math
import random
from bqskit.ir.circuit import Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation
from bqskit.utils.hypergraph import circuit_to_hypergraph, test_kahypar_partitioning
import os
import asyncio
from bqskit.compiler import Compiler

from bqskit.ir import Circuit
from bqskit.compiler.basepass import BasePass
from typing import Any, List, Dict, Set, Tuple
import networkx as nx

class EnhancedHypergraphPartitionPass(BasePass):
    def __init__(self, block_size: int = 3, num_workers: int = 8,hardware_type: str = 'grid',  # 'grid', 'linear', or 'infer'
        grid_dim: Tuple[int, int] = (8, 8),treshold: int = 2  # For grid hardware
        ):
        self.block_size = block_size
        self.num_workers = num_workers
        self.hardware_type = hardware_type
        self.grid_dim = grid_dim
        self.hw_graph = None
        self.dag = None
        self.qubit_maps = None
        self.teleportation_enabled = True  # Add control flag
        self.operation_metadata = {}  # Initialize operation metadata dictionary


    async def run(self, circuit: Circuit, data: dict) -> None:
        # Initialize hardware graph
        self._initialize_hardware_graph(circuit.num_qudits)
        # Step 1: Hypergraph Partitioning (Algorithm 1's improved version)
        print("\n[Step 1/4] Partitioning circuit...")
        partitions, qubit_maps = await self.partition_circuit(circuit)

        # Step 2: Track Original Qubit Indices
        print("[Step 2/4] Tracking original qubit indices...")
        data['partitions'] = partitions
        data['qubit_maps'] = qubit_maps  # Maps: {partition_idx: {trimmed_q: original_q}}

        # NEW: Cut Qubit Minimization Phase
        partitions, qubit_maps = self._merge_partitions(
        partitions, qubit_maps, threshold=2)

        # Step 3: SWAP Minimization
        # print("[Step 3/4] Inserting SWAP gates between partitions...")
        # data['partitions'] = await self.insert_swaps_between_partitions(
        #     data['partitions'], data['qubit_maps']
        # )
        
        # Step 4: Critical Path Parallelism
        print("[Step 4/4] Building dependency graph...")
        data['dependency_graph'] =await self.build_dependency_graph(data['partitions'], data['qubit_maps'])

        print("\nPartitioning complete!")
        print("="*40 + "\n")

    
    def _initialize_hardware_graph(self, num_qudits: int) -> None:
        """Initialize the hardware graph as a fully connected graph."""
        
        self.hw_graph = PhaseAwareHardwareGraph(num_qudits)
        for i in range(num_qudits):
            for j in range(i + 1, num_qudits):
                self.hw_graph.add_edge(i, j)
        logging.info(f"Initialized fully connected hardware graph with {num_qudits} qubits")


    def get_path(self, src: int, dst: int) -> List[int]:
        """Find a SWAP path from src to dst in the hardware graph."""
        if src == dst:
            return [src]
        if not self.hw_graph.has_edge(src, dst):
            logging.warning(f"No direct edge between {src} and {dst}")
            # Find shortest path (simplified for fully connected graph)
            if src < self.hw_graph.num_qudits and dst < self.hw_graph.num_qudits:
                return [src, dst]
            return []
        return [src, dst]

    async def partition_circuit(self, circuit: Circuit) -> tuple[List[Circuit], List[Dict]]:
        # Use Algorithm 1's improved hypergraph construction
        circuit_to_hypergraph(circuit, "temp.hgr",gate_error_rates={
        'HGate': 1e-3,
        'CNOTGate': 5e-2,  # Higher penalty for noisy CNOTs
        'SwapGate': 3e-2
    })
        print("Hypergraph written to circuit.hgr")        
        with open("temp.hgr") as f:
            print("\nNew Quantum-Aware Hypergraph:\n", f.read())
        partition_labels = test_kahypar_partitioning("temp.hgr", num_partitions = max(2, min(circuit.num_operations // self.block_size, int(circuit.num_qudits**0.5))))
        
        # Create partitions with qubit mapping tracking
        partitions, qubit_maps = self._create_trimmed_partitions(circuit, partition_labels)
        return partitions, qubit_maps



    def _create_trimmed_partitions(
    self, 
    circuit: Circuit, 
    labels: List[int]
) -> Tuple[List[Circuit], List[Dict[int, int]]]:
        """Create subcircuits with global-to-physical qubit mapping.
        
        Args:
            circuit: Original circuit with global logical qubits (q0, q1, ...).
            labels: Partition assignments for each gate.
            
        Returns:
            List of partitioned subcircuits and their {global qubit → physical qubit} maps.
        """
        partitions = []
        qubit_maps = []  # Now stores {global qubit → physical qubit}
        
        all_ops = list(circuit.operations())
        if len(labels) != len(all_ops):
            raise ValueError("Labels/operations mismatch.")

        # Iterate over partitions
        for part_id in sorted(set(labels)):
            # Step 1: Collect gates for this partition
            subcircuit = Circuit(circuit.num_qudits)
            for op, label in zip(all_ops, labels):
                if label == part_id:
                    subcircuit.append(op)
            
            # Step 2: Identify active global qubits in this partition
            active_global_qubits = set()  # Track global qubits (q0, q1, ...)
            for op in subcircuit:
                active_global_qubits.update(op.location)  # op.location = global qubits
            
            if not active_global_qubits:
                logging.warning(f"Partition {part_id} has no active qubits. Skipping.")
                continue
            
            # Step 3: Assign physical qubits to global qubits
            # Here we map global qubits to contiguous physical qubits (0, 1, ...)
           
            physical_qubit_map = {
                global_q: physical_q 
                for physical_q, global_q in enumerate(sorted(active_global_qubits))
            }
            # Example: If active_global_qubits = {q0, q4, q6}, then:
            # physical_qubit_map = {0: 0, 4: 1, 6: 2} (global → physical)
            
            # Step 4: Build the trimmed circuit with physical qubits
            trimmed = Circuit(len(active_global_qubits))
            for op in subcircuit:
                try:
                    # Remap global qubits → physical qubits
                    physical_qubits = [physical_qubit_map[global_q] for global_q in op.location]
                    trimmed.append(Operation(op.gate, physical_qubits, op.params))
                except KeyError as e:
                    logging.warning(f"Skipping operation {op}: global qubit {e} not mapped.")
            
            partitions.append(trimmed)
            qubit_maps.append(physical_qubit_map)  # Store {global → physical}
        
        return partitions, qubit_maps

    
    def _infer_hardware_graph(
    self,
    partitions: List[Circuit],
    qubit_maps: List[Dict[int, int]]
) -> nx.Graph:
        """Infer hardware connectivity from circuit structure with physical qubit mapping"""
        hw_graph = nx.Graph()
        
        # 1. Add ALL possible physical nodes from predefined mapping
        hw_graph.add_nodes_from(self.qubit_mapping.values())  # Changed from all_physical
        
        # 2. Infer connectivity from operations
        for i, partition in enumerate(partitions):
            for op in partition.operations():  # Simplified operation access
                if len(op.location) > 1:
                    try:
                        # Convert through BOTH mappings: logical→intermediate→physical
                        phys_qubits = [
                            self.qubit_mapping[qubit_maps[i][q]]  # Double mapping
                            for q in op.location
                        ]
                        # Connect all involved physical qubits
                        for q1, q2 in combinations(phys_qubits, 2):
                            hw_graph.add_edge(q1, q2)
                    except KeyError as e:
                        print(f"Warning: Missing qubit mapping in operation {op}: {e}")
                        continue
        
        # 3. Ensure minimum connectivity - now respects hardware topology
        if not hw_graph.edges():
            if self.hardware_type == 'linear':
                nx.add_path(hw_graph, sorted(self.qubit_mapping.values()))
            else:  # Grid or other
                # Connect nearest neighbors based on physical positions
                for node1, node2 in combinations(self.qubit_mapping.values(), 2):
                    if self._is_adjacent(node1, node2):  # New helper method
                        hw_graph.add_edge(node1, node2)
    
        return hw_graph

    def _is_adjacent(self, pos1, pos2) -> bool:
        """Check if two positions are adjacent in current topology"""
        if self.hardware_type == 'grid':
            return (abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])) == 1
        return abs(pos1 - pos2) == 1  # Linear case


    async def insert_swaps_between_partitions(
        self,
        partitions: List[Circuit],
        qubit_maps: List[Dict[int, int]]
    ) -> List[Circuit]:
        """Insert SWAP gates or teleportation channels to align cut qubits across partitions, preserving critical path."""
        # Functionality: Defines an asynchronous method that takes a list of Circuit objects (partitions) and their corresponding qubit mappings,
        #                returning a modified list of Circuits with SWAP gates or teleportation channels inserted.
        # Purpose: Serves as the main entry point to handle cut qubits using CPPQR, ensuring alignment while preserving circuit depth.
        # Effect: Prepares the partitions for hardware execution by resolving qubit mismatches with minimal SWAPs and zero depth increase.
        self.qubit_maps = qubit_maps
        self.partitions = partitions  # Assign partitions to instance variable
        
        if self.hw_graph is None:
            max_qubit = max((q for qmap in qubit_maps for q in qmap.keys()), default=-1)
            self._initialize_hardware_graph(max_qubit + 1)
        # Functionality: Checks if the hardware graph (self.hw_graph) is uninitialized; if so, finds the maximum original qubit index
        #                across all qubit_maps and initializes the hardware graph with that size plus one.
        # Purpose: Ensures a phase-aware hardware graph exists to represent the connectivity, sized appropriately.
        
        if self.hardware_type == 'infer':
            self.hw_graph = PhaseAwareHardwareGraph(max((q for qmap in qubit_maps for q in qmap.keys()), default=0) + 1)
        # Functionality: If the hardware type is 'infer', creates a new PhaseAwareHardwareGraph instance with a size based on the maximum qubit index
        #                plus one.
        # Purpose: Initializes a phase-aware graph for depth-preserving routing if no prior hardware model exists.
        
        # Functionality: Verifies that self.hw_graph is an instance of PhaseAwareHardwareGraph to ensure compatibility with inter-partition teleportation.
        # Purpose: Guarantees the hardware graph supports phase-aware pathfinding and partition-aware connectivity.
        # Effect: Raises an error or logs a warning if the graph type is incompatible.
        if not isinstance(self.hw_graph, PhaseAwareHardwareGraph):
            logging.error("Hardware graph must be an instance of PhaseAwareHardwareGraph")
            raise ValueError("Incompatible hardware graph type")

        # Functionality: Maps each trimmed qubit index to its corresponding partition based on qubit_maps.
        # Purpose: Enables the hardware graph to track partition assignments for cross-partition pathfinding.
        # Effect: Populates self.hw_graph.partition_map with qubit-to-partition mappings.
        for part_id, qmap in enumerate(qubit_maps):
            for orig_q, trim_q in qmap.items():
                self.hw_graph.add_qubit_to_partition(trim_q, part_id)

        # Functionality: Establishes hardware-supported links between partitions based on the dependency graph or a predefined topology.
        # Purpose: Models inter-partition connectivity required for teleporting qubits across partitions.
        # Effect: Updates self.hw_graph.inter_partition_links with bidirectional links between all partition pairs (simplified topology).
        for src_part in range(len(partitions)):
            for dst_part in range(src_part + 1, len(partitions)):
                self.hw_graph.add_partition_link(src_part, dst_part)

        self.dag = await self.build_dependency_graph(partitions, qubit_maps)
        # Functionality: Asynchronously calls a method to build a dependency graph (DAG) for the partitions using the qubit maps.
        # Purpose: Constructs a directed acyclic graph to represent dependencies, aiding in identifying cut qubits and their phases.
        
        cut_qubits = self._find_cut_qubits(partitions, qubit_maps)
        # Purpose: Identifies the set of cut qubits (e.g., qubits with gates in different partitions), which are targets for alignment.
        
        print("\nPartition Qubit Information:")
        for i, (part, qmap) in enumerate(zip(partitions, qubit_maps)):
            active_qubits_trimmed = {q for op in part.operations() for q in op.location}
            active_qubits_original = {orig_q for trim_q in active_qubits_trimmed 
                                    for orig_q, t in qmap.items() if t == trim_q}
            gate_count = sum(1 for _ in part.operations())
            print(f"Partition {i}:")
            print(f"  Mapped qubits (original): {set(qmap.keys())}")
            print(f"  Active qubits (trimmed): {active_qubits_trimmed}")
            print(f"  Active qubits (original): {active_qubits_original}")
            print(f"  Gate count: {gate_count}")
        
        # Analyze critical path across all partitions
        tracker = PhaseTracker()
        critical_phases = {}
        for part in partitions:
            critical_phases.update(tracker.analyze_critical_path(part))
        self.hw_graph.max_phase = tracker.current_phase
        # Functionality: Analyzes the critical path for each partition to assign phases to gates.
        # Purpose: Ensures SWAPs are inserted in non-critical phases to preserve depth.
        # Effect: Updates self.hw_graph.max_phase with the highest phase encountered.
        
        for q in sorted(cut_qubits):
            print(f"\nInserting SWAPs for qubit {q}:")
            pairs = await self._get_partition_pairs(q)
            print('partition pairs:', pairs, ' share qubit:', q)
            # Functionality: Asynchronously calls a method to determine pairs of partitions that share the cut qubit q.
            # Purpose: Identifies which partitions need alignment for this qubit based on the dependency graph.
            # Effect: Returns a list of (src_part, dst_part) tuples to guide the insertion process.
            
            for src_part, dst_part in pairs:
                try:
                    if q not in qubit_maps[src_part] or q not in qubit_maps[dst_part]:
                        logging.warning(f"Qubit {q} missing in partition {src_part} or {dst_part}")
                        continue
                    
                    src_trim = qubit_maps[src_part][q]
                    dst_trim = qubit_maps[dst_part][q]
                    
                    print(f"Partition {src_part} → Partition {dst_part}")
                    print(f"Path: [{src_trim}, {dst_trim}]")
                    
                    if src_trim == dst_trim:
                        print(f"  No SWAP needed for qubit {q}: same trimmed index {src_trim}")
                        continue
                    
                    # Resize partition if needed
                    max_index = max(src_trim, dst_trim)
                    if max_index >= partitions[src_part].num_qudits:
                        new_size = max_index + 1
                        logging.info(f"Resizing partition {src_part} from {partitions[src_part].num_qudits} to {new_size} qubits")
                        new_circuit = Circuit(new_size)
                        for op in partitions[src_part].operations():
                            new_circuit.append(op)
                        partitions[src_part] = new_circuit
                    # Functionality: Checks if the maximum trimmed index exceeds the partition’s current size; if so, creates a new circuit
                    #                with an increased size and copies existing operations.
                    # Purpose: Ensures the partition’s circuit can accommodate the new qubit index.
                    # Effect: Expands the partition if needed, preserving existing gates.
                    
                    # Extend hardware graph if needed
                    if max_index >= self.hw_graph.num_qudits:
                        logging.info(f"Extending hardware graph to {max_index + 1} qubits")
                        self.hw_graph.extend(max_index + 1)
                    
                    # Debug method types before calling
                    print(f"Debug: create_teleportation_channel type = {type(self.create_quantum_teleportation_channel)}")
                    print(f"Debug: insert_swaps_with_phase_management type = {type(self.insert_swaps_with_phase_management)}")

                    # Decide between teleportation and SWAP insertion
                    rand_val = random.random()
                    if rand_val < 0.1:  # TELEPORTATION_THRESHOLD = 0.1
                        print(f"Choosing teleportation for qubit {q} (random = {rand_val})")
                        print(f"Before calling create_teleportation_channel: type = {type(self.create_quantum_teleportation_channel)}")
                        await self.create_quantum_teleportation_channel(src_part, dst_part, src_trim, q)
                        print(f"After teleportation call for qubit {q}")
                    else:
                        path = self.hw_graph.get_phase_path(
                            src_trim, dst_trim,
                            critical_phases.get(q, 0)
                        )
                        if not path:
                            logging.warning(f"No valid phase-aware path found for qubit {q} from {src_trim} to {dst_trim}")
                            continue
                        print(f"Before SWAP insertion call for qubit {q}")
                        await self.insert_swaps_with_phase_management(
                        partitions[src_part], path, critical_phases
                        )
                        print(f"After SWAP insertion call for qubit {q}")
                        
                        # Functionality: Decides whether to use teleportation or phase-aware SWAP insertion based on a random threshold.
                        # Purpose: Minimizes SWAP gates with teleportation (30% chance) or inserts SWAPs in non-critical phases.
                        # Effect: Updates the partition with either teleportation channels or SWAP gates, preserving depth.
                        await self._update_qubit_maps_after_swaps(qubit_maps, src_part,dst_part, path, shared_qubit=q)
                except Exception as e:
                    logging.error(f"Error processing qubit {q}: {str(e)}")
                    continue
        
        return partitions



    async def build_dependency_graph(self, partitions: List[Circuit], qubit_maps: List[Dict[int, int]]) -> nx.DiGraph:
        #from . import print_dependencies
        dag = nx.DiGraph()
        dag.add_nodes_from(range(len(partitions)))
        
        for i, p1 in enumerate(partitions):
            for j, p2 in enumerate(partitions):
                if i >= j:  # Avoid duplicate checks and self-loops
                    continue
                if await self._partitions_share_qubits(p1, p2, partitions, qubit_maps):
                    # Default: Lower index → Higher index
                    dag.add_edge(i, j)
        await self.print_dependencies(dag, partitions, qubit_maps)
        return dag
    
    async def _partitions_share_qubits(
    self, 
    p1: Circuit, 
    p2: Circuit, 
    partitions: List[Circuit], 
    qubit_maps: List[Dict[int, int]]
) -> bool:
        """
        Check if two partitions share physical qubits by:
        1. Finding their indices in the partitions list
        2. Comparing their physical qubits (original indices) from qubit_maps
        
        Args:
            p1: First partition circuit
            p2: Second partition circuit
            partitions: List of all partition circuits
            qubit_maps: Parallel list of qubit mappings
        
        Returns:
            bool: True if they share any physical qubits
        """
        # Find indices of p1 and p2 in partitions list
        try:
            p1_idx = partitions.index(p1)
            p2_idx = partitions.index(p2)
        except ValueError:
            return False  # One of the partitions wasn't found
        
        # Get original (physical) qubits for each partition
        p1_physical = set(qubit_maps[p1_idx].keys())  # Original qubit indices
        p2_physical = set(qubit_maps[p2_idx].keys())
        
        # Check overlap
        return not p1_physical.isdisjoint(p2_physical)

    async def print_dependencies(self, dag: nx.DiGraph, partitions: List[Circuit], qubit_maps: List[Dict[int, int]]):
        print("\nDependency Graph:")
        print("-----------------")
        for src, dst in dag.edges():
            shared_qubits = set(qubit_maps[src].keys()) & set(qubit_maps[dst].keys())
            print(f"Partition {src:2} → Partition {dst:2} | Shared qubits: {shared_qubits}")
        print(f"\nTotal dependencies: {len(dag.edges())}")
        print("Legend: Partition X → Partition Y means X must execute before Y\n")

    def _find_cut_qubits(self, partitions: List[Circuit], qubit_maps: List[Dict[int, int]]) -> Set[int]:
        """
            Find all physical qubits shared between partitions (cut qubits).
            
            Args:
                partitions: List of partitioned circuits.
                qubit_maps: List of dictionaries mapping logical to physical qubits for each partition.
                
            Returns:
                Set of physical qubit indices shared by at least two partitions.
                
            Notes:
                - Time Complexity: O(N² * K) where N=number of partitions, K=qubits per partition.
                - Handles arbitrary numbers of partitions and qubits.
                - Skips empty partitions automatically.
        """
        cut_qubits = set()
        # Validate inputs
        if len(partitions) != len(qubit_maps):
            raise ValueError("Partitions and qubit_maps must have equal length")
        
        # Compare all partition pairs
        for i in range(len(partitions)):#for i in range(2):  # Compare partitions 0 and 1
            if not qubit_maps[i]:  # Skip empty mappings
                continue
            for j in range(i + 1, len(partitions)):# for j in range(i+1, 2):
                # Get physical qubits for each partition
                p1_qubits = set(qubit_maps[i].keys()) # Physical qubits in partition i,  p1_qubits = {0, 1}
                p2_qubits = set(qubit_maps[j].keys()) # Physical qubits in partition j,  p1_qubits = {1, 2}
                
                # Add shared qubits to cut set
                cut_qubits.update(p1_qubits & p2_qubits) # Compute the intersection (shared qubits) and add to cut_qubits.
                # cut_qubits.update({0,1} & {1,2})  # Adds {1}
                # Output: cut_qubits = {1}
        
        return cut_qubits

    async def _get_partition_pairs(self, qubit: int) -> List[Tuple[int, int]]:
        # Goal: Find dependent partitions sharing qubit 1.
        """
        Get ordered partition pairs (src, dst) that share the given qubit,
        following the dependency graph topological order.

        Notes:
        - Only returns pairs where src must execute before dst (per DAG).
        - Handles cases where qubit is shared across any number of partitions.
        - Validates DAG structure before processing.

        Execution:
            using_partitions = [0, 1]  # Both partitions use qubit 1
            pairs = []
            for src in [0, 1]:
                for dst in [0, 1]:
                    if src != dst and dag.has_edge(src, dst):  # Only 0→1 exists
                        pairs.append((0, 1))
            Output: pairs = [(0, 1)]
        

        """
        # Get all partitions using this qubit
        using_partitions = [
            i for i, qmap in enumerate(self.qubit_maps) 
            if qubit in qmap # Check if qubit exists in partition's mapping
        ]
        
        # Order pairs based on dependency graph (src before dst)
        pairs = []
        for src in using_partitions:
            for dst in using_partitions:
                # Add pair only if dependency exists and src != dst
                if src != dst and self.dag.has_edge(src, dst):
                    pairs.append((src, dst))
        
        return pairs
    
    async def insert_swaps_with_phase_management(
        self,
        partition: Circuit,
        path: List[int],
        critical_phases: Dict[int, int]
    ) -> None:
        """Insert SWAP gates only in non-critical phases."""
        print(f"Inserting SWAPs along path {path}")
        if not path or len(path) < 2:
            logging.warning(f"Invalid SWAP path: {path}")
            return

        max_phase = max(critical_phases.values(), default=0)
        
        for i in range(len(path) - 1):
            q1, q2 = path[i], path[i + 1]
            available_phase = self.find_available_phase(partition, q1, q2, critical_phases, max_phase)
            from bqskit.ir.gates import SwapGate
            from bqskit.ir.operation import Operation as BQSKOperation
            swap_op = BQSKOperation(SwapGate(), (q1, q2))
            
            self.operation_metadata[id(swap_op)] = {'phase': available_phase}
            
            partition.append(swap_op)
            print(f"  + Added SWAP({q1}, {q2}) at phase {available_phase}")
            self.hw_graph.phase_map[q1] = available_phase
            self.hw_graph.phase_map[q2] = available_phase
            #await self._update_qubit_maps_after_swaps(self.qubit_maps, src_part, dst_part, path, shared_qubit)

    def find_available_phase(self, partition: Circuit, q1: int, q2: int, critical_phases: Dict[int, int], max_phase: int) -> int:
        """Find the earliest phase where q1 and q2 are not on the critical path."""
        phase = 0
        while True:
            conflict = False
            for op in partition.operations():
                op_phase = self.operation_metadata.get(id(op), {}).get('phase', 0)
                if (q1 in op.location or q2 in op.location) and op_phase == phase:
                    conflict = True
                    break
                for qubit in [q1, q2]:
                    critical_phase = critical_phases.get(qubit, 0)
                    if critical_phase == phase:
                        conflict = True
                        break
            if not conflict:
                return phase
            phase += 1

    
    async def create_quantum_teleportation_channel(self, src_part: int, dst_part: int, src_trim: int, original_qubit: int) -> None:
        """Create a teleportation channel between partitions with hardware-supported entanglement using PhaseAwareHardwareGraph."""
        print(f"Teleporting qubit {original_qubit} from P{src_part} to P{dst_part}")
        if self.partitions is None:
            logging.error("self.partitions is not initialized")
            return
        if self.hw_graph is None or not isinstance(self.hw_graph, PhaseAwareHardwareGraph):
            logging.error("Hardware graph is not initialized or not a PhaseAwareHardwareGraph")
            return

        # Step 1: Allocate ancilla qubits
        ancilla_1 = self.partitions[src_part].num_qudits  # For local Bell pair in src_part
        ancilla_2 = self.partitions[src_part].num_qudits + 1  # Second qubit of Bell pair in src_part
        ancilla_dst = self.partitions[dst_part].num_qudits  # Target ancilla in dst_part

        # Map ancilla qubits to partitions
        self.hw_graph.add_qubit_to_partition(ancilla_1, src_part)
        self.hw_graph.add_qubit_to_partition(ancilla_2, src_part)
        self.hw_graph.add_qubit_to_partition(ancilla_dst, dst_part)

        # Extend partitions to accommodate ancilla qubits
        new_size_src = self.partitions[src_part].num_qudits + 2
        new_size_dst = ancilla_dst + 1
        logging.info(f"Extending partition {src_part} to {new_size_src} qubits for ancilla")
        logging.info(f"Extending partition {dst_part} to {new_size_dst} qubits for ancilla")
        
        new_circuit_src = Circuit(new_size_src)
        for op in self.partitions[src_part].operations():
            new_circuit_src.append(op)
        self.partitions[src_part] = new_circuit_src

        new_circuit_dst = Circuit(new_size_dst)
        for op in self.partitions[dst_part].operations():
            new_circuit_dst.append(op)
        self.partitions[dst_part] = new_circuit_dst

        # Use BQSKOperation to create operations
        from bqskit.ir.operation import Operation as BQSKOperation
        from bqskit.ir.gates import HGate, CNOTGate, XGate, ZGate, SwapGate
        from bqskit.ir import Operation  # To create custom measurement operation
        from bqskit.ir.gates.measure import MeasurementPlaceholder

        # Custom measurement operation class to wrap MeasurementPlaceholder
        class CustomMeasurementOperation(Operation):
            def __init__(self, classical_regs, measurements, location):
                super().__init__(MeasurementPlaceholder(classical_regs, measurements), location)

        # Step 2: Create a local Bell pair in src_part between ancilla_1 and ancilla_2
        h_op = BQSKOperation(HGate(), (ancilla_1,))
        self.partitions[src_part].append(h_op)
        self.operation_metadata[id(h_op)] = {'bell_prep': True}

        cnot_op = BQSKOperation(CNOTGate(), (ancilla_1, ancilla_2))
        self.partitions[src_part].append(cnot_op)
        self.operation_metadata[id(cnot_op)] = {'bell_prep': True}

        # Step 3: Distribute ancilla_2 to ancilla_dst via hardware path
        path = self.hw_graph.get_phase_path(ancilla_2, ancilla_dst, 0)  # Use phase-aware path
        if not path or len(path) < 2:
            logging.error(f"No valid hardware path found between q{ancilla_2} (P{src_part}) and q{ancilla_dst} (P{dst_part})")
            return

        current_qubit = ancilla_2
        classical_bit_index = 0  # Track classical bit indices
        classical_regs = [('c_reg', len(path) * 2)]  # Single classical register for all measurements

        for next_qubit in path[1:-1]:  # Intermediate qubits in the path
            if next_qubit >= self.partitions[src_part].num_qudits:
                new_size_src = next_qubit + 1
                logging.info(f"Extending partition {src_part} to {new_size_src} qubits for path")
                new_circuit_src = Circuit(new_size_src)
                for op in self.partitions[src_part].operations():
                    new_circuit_src.append(op)
                self.partitions[src_part] = new_circuit_src
            ancilla_link = self.partitions[src_part].num_qudits
            new_size_src = self.partitions[src_part].num_qudits + 1
            logging.info(f"Extending partition {src_part} to {new_size_src} qubits for link ancilla")
            new_circuit_src = Circuit(new_size_src)
            for op in self.partitions[src_part].operations():
                new_circuit_src.append(op)
            self.partitions[src_part] = new_circuit_src

            h_op_link = BQSKOperation(HGate(), (ancilla_link,))
            self.partitions[src_part].append(h_op_link)
            self.operation_metadata[id(h_op_link)] = {'bell_prep': True, 'link': True}

            cnot_op_link = BQSKOperation(CNOTGate(), (ancilla_link, next_qubit))
            self.partitions[src_part].append(cnot_op_link)
            self.operation_metadata[id(cnot_op_link)] = {'bell_prep': True, 'link': True}

            cnot_op_dist = BQSKOperation(CNOTGate(), (current_qubit, ancilla_link))
            self.partitions[src_part].append(cnot_op_dist)
            self.operation_metadata[id(cnot_op_dist)] = {'teleport': True, 'target': current_qubit}

            h_op_dist = BQSKOperation(HGate(), (current_qubit,))
            self.partitions[src_part].append(h_op_dist)
            self.operation_metadata[id(h_op_dist)] = {'teleport': True, 'target': current_qubit}

            # Add measurement using CustomMeasurementOperation
            measurements = {current_qubit: ('c_reg', classical_bit_index), ancilla_link: ('c_reg', classical_bit_index + 1)}
            measure_op = CustomMeasurementOperation(classical_regs, measurements, (current_qubit, ancilla_link))
            self.partitions[src_part].append(measure_op)
            self.operation_metadata[id(measure_op)] = {'teleport': True, 'target': current_qubit}

            x_op_dist = BQSKOperation(XGate(), (next_qubit,))
            self.partitions[src_part].append(x_op_dist)
            self.operation_metadata[id(x_op_dist)] = {'teleport_correction': 'X', 'depends_on': classical_bit_index}
            z_op_dist = BQSKOperation(ZGate(), (next_qubit,))
            self.partitions[src_part].append(z_op_dist)
            self.operation_metadata[id(z_op_dist)] = {'teleport_correction': 'Z', 'depends_on': classical_bit_index + 1}

            current_qubit = next_qubit
            classical_bit_index += 2

        # Final teleportation to ancilla_dst (avoid direct cross-partition CNOT)
        if current_qubit != ancilla_dst:
            ancilla_link_final = self.partitions[src_part].num_qudits
            ancilla_bridge = self.partitions[dst_part].num_qudits  # Bridge qubit in dst_part
            new_size_src = self.partitions[src_part].num_qudits + 1
            new_size_dst = self.partitions[dst_part].num_qudits + 1
            logging.info(f"Extending partition {src_part} to {new_size_src} qubits for final link ancilla")
            logging.info(f"Extending partition {dst_part} to {new_size_dst} qubits for bridge ancilla")
            new_circuit_src = Circuit(new_size_src)
            for op in self.partitions[src_part].operations():
                new_circuit_src.append(op)
            self.partitions[src_part] = new_circuit_src
            new_circuit_dst = Circuit(new_size_dst)
            for op in self.partitions[dst_part].operations():
                new_circuit_dst.append(op)
            self.partitions[dst_part] = new_circuit_dst

            # Create a Bell pair in src_part using ancilla_link_final
            h_op_final = BQSKOperation(HGate(), (ancilla_link_final,))
            self.partitions[src_part].append(h_op_final)
            self.operation_metadata[id(h_op_final)] = {'bell_prep': True, 'link': True}

            # Create a Bell pair in dst_part using ancilla_bridge and ancilla_dst
            h_op_bridge = BQSKOperation(HGate(), (ancilla_bridge,))
            self.partitions[dst_part].append(h_op_bridge)
            self.operation_metadata[id(h_op_bridge)] = {'bell_prep': True, 'link': True}
            cnot_op_bridge = BQSKOperation(CNOTGate(), (ancilla_bridge, ancilla_dst))
            self.partitions[dst_part].append(cnot_op_bridge)
            self.operation_metadata[id(cnot_op_bridge)] = {'bell_prep': True, 'link': True}

            # Teleport current_qubit to ancilla_bridge
            cnot_op_dist_final = BQSKOperation(CNOTGate(), (current_qubit, ancilla_link_final))
            self.partitions[src_part].append(cnot_op_dist_final)
            self.operation_metadata[id(cnot_op_dist_final)] = {'teleport': True, 'target': current_qubit}

            h_op_dist_final = BQSKOperation(HGate(), (current_qubit,))
            self.partitions[src_part].append(h_op_dist_final)
            self.operation_metadata[id(h_op_dist_final)] = {'teleport': True, 'target': current_qubit}

            # Add measurement using CustomMeasurementOperation
            measurements = {current_qubit: ('c_reg', classical_bit_index), ancilla_link_final: ('c_reg', classical_bit_index + 1)}
            measure_op_final = CustomMeasurementOperation(classical_regs, measurements, (current_qubit, ancilla_link_final))
            self.partitions[src_part].append(measure_op_final)
            self.operation_metadata[id(measure_op_final)] = {'teleport': True, 'target': current_qubit}

            # Apply corrections in dst_part
            x_op_dist_final = BQSKOperation(XGate(), (ancilla_bridge,))
            self.partitions[dst_part].append(x_op_dist_final)
            self.operation_metadata[id(x_op_dist_final)] = {'teleport_correction': 'X', 'depends_on': classical_bit_index}
            z_op_dist_final = BQSKOperation(ZGate(), (ancilla_bridge,))
            self.partitions[dst_part].append(z_op_dist_final)
            self.operation_metadata[id(z_op_dist_final)] = {'teleport_correction': 'Z', 'depends_on': classical_bit_index + 1}

            classical_bit_index += 2

            # The state is now on ancilla_bridge in dst_part; SWAP to ancilla_dst if needed
            if ancilla_bridge != ancilla_dst:
                swap_op = BQSKOperation(SwapGate(), (ancilla_bridge, ancilla_dst))
                self.partitions[dst_part].append(swap_op)
                self.operation_metadata[id(swap_op)] = {'swap': True, 'link': True}

        # Step 4: Use the distributed Bell pair (ancilla_1, ancilla_dst) to teleport original_qubit
        cnot_op_main = BQSKOperation(CNOTGate(), (src_trim, ancilla_1))
        self.partitions[src_part].append(cnot_op_main)
        self.operation_metadata[id(cnot_op_main)] = {'teleport': True, 'original_qubit': original_qubit}

        h_op_main = BQSKOperation(HGate(), (src_trim,))
        self.partitions[src_part].append(h_op_main)
        self.operation_metadata[id(h_op_main)] = {'teleport': True, 'original_qubit': original_qubit}

        # Add measurement using CustomMeasurementOperation
        measurements = {src_trim: ('c_reg', classical_bit_index), ancilla_1: ('c_reg', classical_bit_index + 1)}
        measure_op_main = CustomMeasurementOperation(classical_regs, measurements, (src_trim, ancilla_1))
        self.partitions[src_part].append(measure_op_main)
        self.operation_metadata[id(measure_op_main)] = {'teleport': True, 'original_qubit': original_qubit}

        x_op_main = BQSKOperation(XGate(), (ancilla_dst,))
        self.partitions[dst_part].append(x_op_main)
        self.operation_metadata[id(x_op_main)] = {'teleport_correction': 'X', 'depends_on': classical_bit_index}
        z_op_main = BQSKOperation(ZGate(), (ancilla_dst,))
        self.partitions[dst_part].append(z_op_main)
        self.operation_metadata[id(z_op_main)] = {'teleport_correction': 'Z', 'depends_on': classical_bit_index + 1}

        # Step 5: Update qubit_maps to reflect teleportation
        # The state of original_qubit has moved to ancilla_dst in dst_part
        if hasattr(self, 'qubit_maps') and self.qubit_maps is not None:
            # Remove the original qubit mapping from src_part
            if original_qubit in self.qubit_maps[src_part]:
                del self.qubit_maps[src_part][original_qubit]
            # Add the new mapping in dst_part
            self.qubit_maps[dst_part][original_qubit] = ancilla_dst
            logging.debug(f"Updated qubit_maps: Moved qubit {original_qubit} to index {ancilla_dst} in P{dst_part}")

        logging.info(f"Created teleport channel for qubit {original_qubit} between P{src_part} and P{dst_part}")
    

    async def _update_qubit_maps_after_swaps(self, qubit_maps: List[Dict[int, int]], src_part: int, dst_part: int, path: List[int], shared_qubit: int) -> None:
        """Update qubit mappings after SWAP operations between partitions."""
        if len(path) < 2:
            return
        src_idx, dst_idx = path[0], path[-1]
        orig_q = next((orig_q for orig_q, trim_q in qubit_maps[src_part].items() if trim_q == src_idx), None)
        if orig_q is None or orig_q != shared_qubit:
            logging.debug(f"qubit_maps[{src_part}] = {qubit_maps[src_part]}")
            logging.warning(f"Cannot update map: qubit for trimmed index {src_idx} not found or does not match shared qubit {shared_qubit}")
            return
        if orig_q in qubit_maps[src_part]:
            del qubit_maps[src_part][orig_q]
        qubit_maps[dst_part][orig_q] = dst_idx
        logging.debug(f"Updated mapping: Moved qubit {orig_q} from P{src_part} (index {src_idx}) to P{dst_part} (index {dst_idx})")

   
    def _combine_partitions(
    self, 
    p1: Circuit, 
    p2: Circuit, 
    map1: Dict[int, int], 
    map2: Dict[int, int]
) -> Tuple[Circuit, Dict[int, int]]:
        """Merge two partitions into a single circuit with a unified qubit map."""

        # Initialize an empty set to collect all qubits involved in the merge
        all_qubits = set()
        # Loop through each operation in p1 and add its qubit indices to the set
        for op in p1:
            all_qubits.update(op.location)
        # Loop through each operation in p2 and add its qubit indices to the set
        for op in p2:
            all_qubits.update(op.location)
        
        # Add all original qubits from map1 (keys are original qubits)
        all_qubits.update(map1.keys())
        # Add all original qubits from map2 (keys are original qubits)        
        all_qubits.update(map2.keys())
        
        if not all_qubits:
            logging.warning("No qubits found in merged partitions.")
            return Circuit(1), {}
        
        # Create a new circuit with the total number of unique qubits
        combined = Circuit(len(all_qubits))
        # Create a mapping from original qubits to new indices (0 to len(all_qubits)-1)
        physical_to_logical = {p: i for i, p in enumerate(sorted(all_qubits))}
        # Create a new qubit map using the same mapping for all qubits
        new_map = {q: physical_to_logical[q] for q in all_qubits}
        
        # Set to track unique operations
        seen_operations = set()
        # Remap and add all operations from p1 to the combined circuit
        for op in p1:
            try:
                # Map each qubit in the operation's location to the new indices
                new_location = tuple(new_map[q] for q in op.location)
                op_key = (op.gate, new_location)  # Unique key for deduplication
                # Add the operation with remapped qubits to the combined circuit
                combined.append(Operation(op.gate, new_location, op.params))
                seen_operations.add(op_key)
            except KeyError as e:
                # If a qubit isn't in the map, log a warning and skip the operation
                logging.warning(f"Skipping operation {op} in p1: qubit {e} not mapped")
                continue
        
        # Remap and add all operations from p2 to the combined circuit
        for op in p2:
            try:
                # Map each qubit in the operation's location to the new indices
                new_location = tuple(new_map[q] for q in op.location)
                op_key = (op.gate, new_location)  # Unique key for deduplication
                # Add the operation with remapped qubits to the combined circuit
                combined.append(Operation(op.gate, new_location, op.params))
                seen_operations.add(op_key)
            except KeyError as e:
                # If a qubit isn't in the map, log a warning and skip the operation
                logging.warning(f"Skipping operation {op} in p2: qubit {e} not mapped")
                continue
        
        logging.info(f"Merged partitions into circuit with {len(combined)} gates and {len(all_qubits)} qubits")
        return combined, new_map


    #####################################################################################
    # Merge section include: merge function,combine function and shared qubits function #
    #####################################################################################

    def _merge_partitions(self, partitions: List[Circuit], qubit_maps: List[Dict[int, int]], threshold: int = 4) -> Tuple[List[Circuit], List[Dict[int, int]]]:
        """Merge partitions with sufficient shared qubits to minimize cut qubits."""

        # Initialize a flag to track if any merges occurred in this iteration
        merged = True
        # Set to track all merged partition pairs across iterations
        merged_pairs = set()  # (min(i, j), max(i, j)) to ensure uniqueness

        while merged:
            merged = False
            new_partitions = [] # empty list for new partitions after merging
            new_qubit_maps = [] # empty for new qubit maps after merging

            # Initialize a set to track indices of partitions that have been processed
            used = set() # Tracks which partitions have been merged or kept
            
            # Loop over each partition index to find candidates for merging
            for i in range(len(partitions)): #for example if we have 4 partitions so i start form 0 to 3 
                if i in used:
                    continue

                # Get the circuit and qubit map for partition i
                p1, map1 = partitions[i], qubit_maps[i] 

                # Initialize variables to track the best merge candidate
                shared_max = 0 # Maximum number of shared qubits found
                merge_idx = -1 # Index of the best candidate partition to merge with
                
                # Verify partition has valid mappings
                # Verify that p1's operations only use qubits that are in map1
                part_qubits = {q for op in p1 for q in op.location}  # Collect all qubits used in p1's operations
                #{expression for item in iterable for subitem in subiterable}
                """
                Example with Partition 0:
                Assume p1 (Partition 0) has gates [HGate(0,), CNOTGate(0, 1)] from your earlier example.
                First operation: op = HGate(0,), op.location = (0,).
                Inner loop: q takes value 0.
                Set starts with {0}.
                Second operation: op = CNOTGate(0, 1), op.location = (0, 1).
                Inner loop: q takes values 0, then 1.
                Set updates to {0, 1} (0 is already there, 1 is added).
                Result: part_qubits = {0, 1}
                """

                # Map the trimmed qubits back to original qubits using map1
                part_qubits_mapped = {orig_q for trim_q in part_qubits 
                                    for orig_q, t in map1.items() if t == trim_q}
                #{expression for item in iterable for subitem in subiterable if condition}.
                """
                Example with Partition 0:
                part_qubits = {0, 1}
                map1 = {0: 0, 1: 1} (from earlier example, where original qubits 0 and 1 map to trimmed 0 and 1).
                First trim_q = 0:
                map1.items() gives [(0, 0), (1, 1)].
                Check t == 0: Matches when (orig_q, t) = (0, 0), so add orig_q = 0.
                Result so far: {0}.
                Second trim_q = 1:
                map1.items() again gives [(0, 0), (1, 1)].
                Check t == 1: Matches when (orig_q, t) = (1, 1), so add orig_q = 1.
                Result updates to {0, 1}.
                Result: part_qubits_mapped = {0, 1}
                """

                # Check if there are any unmapped qubits (shouldn't happen, but safety check)
                if part_qubits_mapped - set(map1.keys()):
                    logging.warning(f"Partition {i} has operations on unmapped qubits: {part_qubits_mapped - set(map1.keys())}")
                    new_partitions.append(p1)
                    new_qubit_maps.append(map1)
                    used.add(i)
                    continue
                
                # Find best merge candidate, Look for another partition to merge with
                for j in range(i + 1, len(partitions)):
                    if j in used:
                        continue
                    
                    # Check if this pair was already merged
                    pair = tuple(sorted([i, j]))
                    if pair in merged_pairs:
                        continue

                    # Get the circuit and qubit map for partition j
                    p2, map2 = partitions[j], qubit_maps[j]
                    # Find the number of shared original qubits between p1 and p2
                    shared = self._shared_qubits(p1, p2, map1, map2)
                    # Check if the number of shared qubits meets the threshold
                    if len(shared) >= threshold:
                    # If this pair has more shared qubits than the current maximum, update the candidate
                        if len(shared) > shared_max:
                            shared_max = len(shared)
                            merge_idx = j
                
                if merge_idx != -1:
                    # Get the circuit and map of the merge candidate
                    p2, map2 = partitions[merge_idx], qubit_maps[merge_idx]
                    try:
                        # Print state before merging
                        print(f"Before merging partitions {i} and {merge_idx}:")
                        print(f"Partition {i} gates: {[str(op) for op in p1]}")
                        print(f"Partition {i} map: {map1}")
                        print(f"Partition {merge_idx} gates: {[str(op) for op in p2]}")
                        print(f"Partition {merge_idx} map: {map2}")
                        # Merge the two partitions using _combine_partitions
                        combined, new_map = self._combine_partitions(p1, p2, map1, map2)

                        # Print state after merging
                        print(f"After merging partitions {i} and {merge_idx}:")
                        print(f"Merged partition gates: {[str(op) for op in combined]}")
                        print(f"Merged partition map: {new_map}")
                        print()
                        # Add the merged circuit to the new partitions list
                        new_partitions.append(combined)
                        # Add the new qubit map to the new maps list
                        new_qubit_maps.append(new_map)
                        # Mark both partitions as used
                        used.add(i)
                        used.add(merge_idx)
                        # Indicate that a merge occurred, so the loop will continue
                        merged = True
                        merged_pairs.add(tuple(sorted([i, merge_idx])))  # Record the merge
                        logging.info(f"Merged partitions {i} and {merge_idx} with {shared_max} shared qubits")
                    except Exception as e:
                        # If merging fails, log a warning and keep the first partition as-is
                        logging.warning(f"Merge failed for partitions {i} and {merge_idx}: {str(e)}")
                        new_partitions.append(p1)
                        new_qubit_maps.append(map1)
                        used.add(i)
                else:
                    new_partitions.append(p1)
                    new_qubit_maps.append(map1)
                    used.add(i)
            
            partitions, qubit_maps = new_partitions, new_qubit_maps
        
        logging.info(f"Final merge: {len(partitions)} partitions with {len(self._find_cut_qubits(partitions, qubit_maps))} cut qubits")
        return partitions, qubit_maps


    def _shared_qubits(
        self,
        p1: Circuit,
        p2: Circuit,
        map1: Dict[int, int],
        map2: Dict[int, int]
    ) -> Set[int]:
        """Compute the set of shared original qubits between two partitions."""
        # Get original qubits from operations (via trimmed indices)
        # Collect all qubits used in p1's operations
        p1_qubits_trimmed = {q for op in p1 for q in op.location}
        # Collect all qubits used in p2's operations
        p2_qubits_trimmed = {q for op in p2 for q in op.location}
        
        # Map trimmed indices to original qubits
        # Map p1's trimmed qubits back to original qubits using map1
        p1_qubits_original = {orig_q for trim_q in p1_qubits_trimmed 
                            for orig_q, t in map1.items() if t == trim_q}
        # Map p2's trimmed qubits back to original qubits using map2
        p2_qubits_original = {orig_q for trim_q in p2_qubits_trimmed 
                            for orig_q, t in map2.items() if t == trim_q}
        
        # Include all qubits in maps to ensure merge eligibility
        p1_qubits_original.update(map1.keys())
        p2_qubits_original.update(map2.keys())
        
        # Return intersection
        shared = p1_qubits_original & p2_qubits_original
        return shared
###############################################################
#                                                             #
#PhaseAwareHardwareGraph that support phase-aware pathfinding.#
#                                                             #
###############################################################

class PhaseAwareHardwareGraph:
    def __init__(self, num_qudits: int):
        """Initialize a 2D grid hardware graph with phase tracking for num_qudits qubits."""
        self.num_qudits = num_qudits
        self.grid_width = math.ceil(math.sqrt(num_qudits))
        self.grid_height = math.ceil(num_qudits / self.grid_width)
        self.edges = set()
        self.phase_map = defaultdict(int)
        self.max_phase = 0
        self.partition_map = {}  # Map qubit to partition: {qubit: partition_id}
        self.inter_partition_links = {}  # {partition_id: set of linked partitions}
        self._build_grid()

    def _build_grid(self) -> None:
        """Build edges for a 2D grid topology."""
        for qubit in range(self.num_qudits):
            x, y = qubit % self.grid_width, qubit // self.grid_width
            if x + 1 < self.grid_width:  # Right
                neighbor = y * self.grid_width + (x + 1)
                if neighbor < self.num_qudits:
                    self.add_edge(qubit, neighbor)
            if y + 1 < self.grid_height:  # Down
                neighbor = (y + 1) * self.grid_width + x
                if neighbor < self.num_qudits:
                    self.add_edge(qubit, neighbor)

    def add_edge(self, q1: int, q2: int) -> None:
        """Add a bidirectional edge between q1 and q2."""
        if q1 >= self.num_qudits or q2 >= self.num_qudits:
            new_size = max(q1, q2) + 1
            self.extend(new_size)
        self.edges.add((min(q1, q2), max(q1, q2)))
        logging.debug(f"Added edge ({q1}, {q2})")

    def add_qubit_to_partition(self, qubit: int, partition_id: int) -> None:
        """Assign a qubit to a partition."""
        self.partition_map[qubit] = partition_id

    def add_partition_link(self, part1: int, part2: int) -> None:
        """Add a hardware-supported link between partitions."""
        if part1 not in self.inter_partition_links:
            self.inter_partition_links[part1] = set()
        if part2 not in self.inter_partition_links:
            self.inter_partition_links[part2] = set()
        self.inter_partition_links[part1].add(part2)
        self.inter_partition_links[part2].add(part1)
        logging.debug(f"Added link between partitions {part1} and {part2}")

    def get_neighbors(self, qubit: int) -> List[int]:
        """Return list of adjacent qubits."""
        neighbors = []
        for q1, q2 in self.edges:
            if q1 == qubit:
                neighbors.append(q2)
            elif q2 == qubit:
                neighbors.append(q1)
        return neighbors

    def get_phase_path(self, src: int, dst: int, critical_phase: int) -> List[int]:
        """Find path that doesn’t conflict with critical phase operations, supporting inter-partition links."""
        if src == dst:
            return [src]

        src_part = self.partition_map.get(src)
        dst_part = self.partition_map.get(dst)
        if src_part is None or dst_part is None:
            logging.error(f"Qubit {src} or {dst} not mapped to a partition")
            return []

        if src_part != dst_part:
            # Check for inter-partition link
            if src_part in self.inter_partition_links and dst_part in self.inter_partition_links[src_part]:
                # Simulate a direct path through the link
                # In practice, this would involve hardware-specific routing
                path = [src, dst]
                # Check phase conflicts along the path
                for qubit in path:
                    if self.phase_map[qubit] > critical_phase:
                        continue  # Try next phase
                return path
            logging.error(f"No link between partitions {src_part} and {dst_part}")
            return []

        # Same partition: Use existing phase-aware pathfinding
        for t in range(critical_phase + 1, self.max_phase + 1):
            path = self._find_non_conflicting_path(src, dst, t)
            if path:
                return path
        return self.get_path(src, dst)

    def _find_non_conflicting_path(self, src: int, dst: int, target_phase: int) -> List[int]:
        """Find a path avoiding phase conflicts."""
        queue = deque([(src, [src])])
        visited = {src}
        while queue:
            current, path = queue.popleft()
            if current == dst:
                return path
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and self.phase_map[neighbor] <= target_phase:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def get_path(self, src: int, dst: int) -> List[int]:
        """Find shortest path using BFS."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        adj = defaultdict(list)
        for q1, q2 in self.edges:
            adj[q1].append(q2)
            adj[q2].append(q1)
        while queue:
            current, path = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == dst:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return []

    def extend(self, new_size: int) -> None:
        """Extend the graph to support new_size qubits."""
        if new_size <= self.num_qudits:
            return
        old_width = self.grid_width
        self.num_qudits = new_size
        self.grid_width = math.ceil(math.sqrt(new_size))
        self.grid_height = math.ceil(new_size / self.grid_width)
        self.edges.clear()
        self._build_grid()
        logging.info(f"Extended 2D grid hardware graph from {old_width}x{math.ceil(self.num_qudits/old_width)} to {self.grid_width}x{self.grid_height} ({new_size} qubits)")

#############################
#
#
#############################
class PhaseTracker:
    def __init__(self):
        self.current_phase = 0
        self.phase_chains = defaultdict(list)

    def analyze_critical_path(self, circuit: Circuit) -> Dict[int, int]:
        """Map each gate to its critical phase."""
        # Functionality: Analyzes the circuit in reverse order to assign phases based on gate dependencies.
        # Purpose: Identifies the critical path by determining the latest phase each qubit is involved in.
        # Effect: Returns a dictionary mapping qubits to their maximum phase.
        phase_map = {}
        # Convert iterator to list to enable reverse iteration
        operations_list = list(circuit.operations())
        for op in reversed(operations_list):
            deps = [phase_map.get(q, 0) for q in op.location]
            phase = max(deps, default=0) + 1
            for q in op.location:
                phase_map[q] = phase
            self.phase_chains[phase].append(op)
        self.current_phase = max(phase_map.values(), default=0)
        return phase_map

class MeasureGate:
    def __init__(self, classical_bit: int = None):
        """Represent a quantum measurement operation."""
        # Functionality: Defines a gate that measures a qubit, collapsing its state.
        # Purpose: Used in teleportation to measure the source qubit and its entangled partner, storing the result.
        # Effect: Produces a classical outcome for use in correction at the destination.
        self.name = "measure"
        self.classical_bit = classical_bit  # Index of classical bit to store the measurement result
        # Note: Assumes the circuit has a classical register; if not, this needs to be handled externally.

class ClassicalCorrectGate:
    def __init__(self, measurement_bits: Tuple[int, int] = None):
        """Represent a classical correction operation based on measurement outcomes."""
        # Functionality: Defines a gate that applies corrections based on classical measurement data.
        # Purpose: Used in teleportation to adjust the destination qubit state using classical feedback.
        # Effect: Applies conditional X and Z gates to the destination qubit based on measurement outcomes.
        self.name = "classical_correct"
        self.measurement_bits = measurement_bits  # Tuple of (bit0, bit1) for X and Z corrections
        # Note: bit0 controls X gate (if 1, apply X); bit1 controls Z gate (if 1, apply Z).

class CustomOperation:
    def __init__(self, gate, location: Tuple[int, ...], metadata=None, params=None):
        self.gate = gate
        self.location = location
        self.metadata = metadata or {}
        self.params = params or []

async def main():
    from bqskit.ir.gates import HGate, CNOTGate
    from bqskit.ir.operation import Operation
    from bqskit.ir.circuit import Circuit

    from bqskit.ir.gates import HGate, CNOTGate
    # Set random seed for reproducibility (optional, remove for different randomizations)
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
    #################################################
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

    # Create and run partition pass
    hypergraph_pass = EnhancedHypergraphPartitionPass(block_size=6, num_workers=8)
    data = {}
    print("\nInitial Circuit:")
    print("-" * 40)
    print(f"Total qubits: {test_circuit.num_qudits}")
    print(f"Total gates: {len(test_circuit)}")
    print("Circuit structure:")
    for i, op in enumerate(test_circuit.operations()):
        print(f"Gate {i:2}: {op.gate.name} on {op.location}")

    await hypergraph_pass.run(test_circuit,data)

    print("\nFinal Results:")
    print("-" * 40)
    print(f"Created {len(data['partitions'])} partitions:")
    for idx, (part, qmap) in enumerate(zip(data['partitions'], data['qubit_maps'])):
        print(f"\nPartition {idx}:")
        print(f"Qubit Map: {qmap}")
        print(f"Contains {len(part)} gates:")
        for op in part.operations():
            print(f"  - {op.gate.name} on {op.location}")
    
    print("\nDependency Graph Visualization:")
    print("-" * 40)
    for src, dst in data['dependency_graph'].edges():
        shared = set(data['qubit_maps'][src].keys()) & set(data['qubit_maps'][dst].keys())
        print(f"Partition {src} → Partition {dst} (Shared qubits: {shared})")
    


if __name__ == "__main__":
    # Run the async main function
    partitions = asyncio.run(main())
