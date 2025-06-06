
-----

````markdown
# Fidelipart: A Fidelity-Aware Hypergraph Partitioning Framework for BQSKit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for **Fidelipart**, a novel, fidelity-aware partitioning pass integrated into the BQSKit framework, as presented in the paper "[Your Paper Title Here]".

This project is an extension of the official [Berkeley Quantum Synthesis Toolkit (BQSKit)](https://github.com/BQSKit/bqskit) and includes all of its original functionality, extended with our custom partitioning logic.

## Description

Effective circuit partitioning is critical for Noisy Intermediate-Scale Quantum (NISQ) devices, which are hampered by high error rates and limited qubit connectivity. Standard partitioning heuristics often neglect gate-specific error impacts, leading to suboptimal divisions with significant communication overhead and reduced fidelity. Fidelipart addresses this by transforming quantum circuits into a fidelity-aware hypergraph. In this model, gate error rates and structural dependencies inform the weights of nodes (gates) and hyperedges (representing multi-qubit interactions and qubit timelines), guiding the Mt-KaHyPar partitioner to minimize cuts through error-prone operations and preserve circuit integrity.

## Key Features

- **Fidelity-Aware Hypergraph Model:** Converts quantum circuits into a weighted hypergraph where weights are derived from gate error rates and structural properties.
- **Advanced Partitioning:** Leverages the state-of-the-art Mt-KaHyPar solver to partition the hypergraph, optimizing for minimal inter-partition connectivity using the `km1` metric.
- **Optional Partition Merging:** Includes a heuristic-based merging stage to further reduce cut qubits by combining partitions that share a significant number of global qubits.
- **Dependency Analysis:** Constructs a dependency graph between the final partitions to define a valid execution order based on shared qubits.
- **Local Contiguous Re-mapping:** Standardizes subcircuits by trimming unused qubits and remapping active global qubits to compact, local indices for efficient execution and analysis.

## Prerequisites

- A Unix-like environment (e.g., Linux, or WSL on Windows for compiling Mt-KaHyPar)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management
- A modern C++ compiler (e.g., GCC/g++ or Clang) required for building Mt-KaHyPar.

## Installation and Setup

Follow these steps to set up the project and run the experiments.

**Step 1: Clone this Repository**

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
````

**Step 2: Compile and Install Mt-KaHyPar (External Dependency)**

Fidelipart relies on the Mt-KaHyPar partitioner, which must be compiled from source.

1.  Clone the official Mt-KaHyPar repository:
    ```bash
    git clone [https://github.com/kahypar/mt-kahypar.git](https://github.com/kahypar/mt-kahypar.git)
    cd mt-kahypar
    ```
2.  Follow their official compilation instructions to build the project. A typical workflow is:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
3.  **IMPORTANT:** After successful compilation, you must either:
      * **Option A (Recommended):** Add the directory containing the `mtkahypar` executable (typically `.../mt-kahypar/build/mt-kahypar/application/`) to your system's `PATH` environment variable.
      * **Option B (Alternative):** Manually edit the path in our script. In `bqskit/passes/partitioning/hypergraph.py`, find the `mtkahypar_bin` variable and replace the default value with the full, absolute path to your compiled executable.

**Step 3: Create and Activate Conda Environment**

The provided `environment.yml` file contains all necessary Python packages (including the specific version of BQSKit this fork is based on, Cirq, NetworkX, etc.).

1.  Navigate back to the root directory of this project (`your-repo-name`).
2.  Create the Conda environment from the file:
    ```bash
    conda env create -f environment.yml
    ```
3.  Activate the new environment:
    ```bash
    conda activate QB
    ```

**Step 4: Install This Modified BQSKit**

To ensure your Python environment uses the `Fidelipart` code included in this repository, install this modified version of BQSKit in "editable" mode. This links the installation to this source folder.

```bash
pip install -e .
```

The setup is now complete. Your `QB` conda environment is active and configured to use the `Fidelipart` code from this directory.

## How to Run Experiments

The primary script for reproducing the results in our paper is located at `bqskit/passes/partitioning/compare_partitioning.py`.

To run the full comparison on all benchmark circuits, execute the following command from the root directory of the project:

```bash
python bqskit/passes/partitioning/compare_partitioning.py
```



## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Acknowledgements

This work builds upon the excellent open-source [BQSKit](https://github.com/BQSKit/bqskit) framework from Lawrence Berkeley National Laboratory and utilizes the powerful [Mt-KaHyPar](https://github.com/kahypar/mt-kahypar) hypergraph partitioner. We thank their respective development teams for making these tools available.

```
```