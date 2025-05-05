"""
This example demonstrates using the standard BQSKit compile function.

For a much more detailed tutorial, see the bqskit-tutorial repository.
"""
import asyncio
from bqskit import Circuit
from bqskit.compiler import CompilationTask
from bqskit.passes import QuickPartitioner, ExtendBlockSizePass, ForEachBlockPass, UnfoldPass
from bqskit.passes.util.log import LogPass
from bqskit.compiler import Compiler
from bqskit.ir.gates import U3Gate, CNOTGate

def partition_circuit():
    # Create a larger circuit directly (10 qubits, with multiple gates)
    circuit = Circuit(10)
    
    # Apply U3 and CNOT gates to create a complex circuit
    for i in range(5):
        circuit.append_gate(U3Gate([0.5, 0.5, 0.5]), [i])  # Apply U3 gate on each qubit
        if i < 9:
            circuit.append_gate(CNOTGate(), [i, i+1])  # Apply CNOT between consecutive qubits

    # Step 2: Create the partitioning pass workflow
    block_size = 2  # You can adjust the block_size depending on your circuit and requirements
    workflow = [
        QuickPartitioner(block_size),
        ExtendBlockSizePass(),
        ForEachBlockPass([LogPass('log')]),
        UnfoldPass()
    ]

    # Step 3: Create the compiler object
    compiler = Compiler()

    # Step 4: Compile the circuit with the workflow
    result = compiler.compile(circuit, workflow)  # No need for await

    # Step 5: Print the result of the compilation
    print("Result after running compilation task:")
    print(result)

    # Step 6: Check if result has subcircuits (partitions)
    if hasattr(result, 'subcircuits') and result.subcircuits:
        partitions = result.subcircuits  # Access the subcircuits
        num_partitions = len(partitions)
        print(f"🔹 Number of partitions: {num_partitions}\n")

        for i, subcircuit in enumerate(partitions):
            print(f"📌 Partition {i+1}:")
            print(f"   - Number of gates: {subcircuit.gate_count}")  # Correct method to get number of gates
            print(f"   - Qubits involved: {set(subcircuit.qubits)}\n")  # Correct way to list qubits
    else:
        print("❌ No subcircuits found in the result.")

    print("✅ Partitioning completed.")

# Run the function
partition_circuit()

