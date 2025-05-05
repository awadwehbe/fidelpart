<<<<<<< HEAD
# Berkeley Quantum Synthesis Toolkit (BQSKit)

The Berkeley Quantum Synthesis Toolkit (BQSKit) \[bis • kit\] is a powerful
and portable quantum compiler framework. It can be used with ease to compile
quantum programs to efficient physical circuits for any QPU.

## Installation

BQSKit is available for Python 3.8+ on Linux, macOS, and Windows. BQSKit
and its dependencies are listed on the [Python Package Index](https://pypi.org),
and as such, pip can easily install it:

```sh
pip install bqskit
```

## Basic Usage

A standard BQSKit workflow loads a program into the framework, models the
target QPU, compiles the program, and exports the resulting circuit. The
below example uses BQSKit to optimize an input circuit provided by a qasm
file:

```python
from bqskit import compile, Circuit

# Load a circuit from QASM
circuit = Circuit.from_file("input.qasm")

# Compile the circuit
compiled_circuit = compile(circuit)

# Save output as QASM
compiled_circuit.save("output.qasm")
```

To learn more about BQSKit, follow the
[tutorial series](https://github.com/BQSKit/bqskit-tutorial/) or refer to
the [documentation](https://bqskit.readthedocs.io/en/latest/).

## How to Cite

You can use the [software disclosure](https://www.osti.gov/biblio/1785933)
to cite the BQSKit package.

Additionally, if you used or extended a specific algorithm, you should cite
that individually. BQSKit passes will include a relevant reference in
their documentation.

## License

The software in this repository is licensed under a **BSD free software
license** and can be used in source or binary form for any purpose as long
as the simple licensing requirements are followed. See the
**[LICENSE](https://github.com/BQSKit/bqskit/blob/master/LICENSE)** file
for more information.

## Copyright

Berkeley Quantum Synthesis Toolkit (BQSKit) Copyright (c) 2021,
The Regents of the University of California, through Lawrence
Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy) and Massachusetts
Institute of Technology (MIT).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to
do so.


"""


"""

## Installation and Setup

To use BQSKIT with its dependencies, including `pymetis` for graph partitioning and `kahypar` for hypergraph partitioning, follow these steps to set up the environment. These instructions assume you have a compatible system (Linux, macOS, or Windows with WSL).

### Prerequisites
- **Miniconda**: Install Miniconda to manage the Python environment. Download it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions for your platform.
- **Git**: Required to clone or work with the repository (if applicable).

### Setting Up the Environment
The recommended way to install dependencies is using Conda with the provided `environment.yml` file, which ensures compatibility with `pymetis` (installed via Conda-Forge) and `kahypar` (installed via PyPI).

#### Step 1: Clone or Extract the Project
If you received this as a Git repository:
```bash
git clone <repository-url>
cd bqskit

#### Step 2: Create the Environment
#Use the environment.yml file to recreate the exact environment:
conda env create -f environment.yml

"""
This sets up a Conda environment named pymetis_env with Python 3.11 and all required packages.

The contents of environment.yml are:
"""
name: pymetis_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.26.0
  - scipy>=1.12.0
  - lark>=1.2.2
  - pymetis>=2022.1
  - pip
  - pip:
    - kahypar>=1.0
    - bqskitrs>=0.4.1
    - dill>=0.3.9
    - typing_extensions>=4.12.2

### Step 3: Activate the Environment
#Activate the environment to use it:
conda activate pymetis_env

### Step 4: Install BQSKIT
# Install the local bqskit package in editable mode:
pip install -e .

###Step 5: Verify Installation
#Test that everything is set up:

python -c "import bqskit, pymetis, kahypar; print('BQSKIT and dependencies installed successfully')"

"""
Alternative Setup with pip
If you prefer not to use Conda or encounter issues:

Create a virtual environment

"""
python -m venv venv
source venv/bin/activate  # Linux/WSL/Mac
venv\Scripts\activate     # Windows CMD

#Install dependencies from requirements.txt:
pip install -r requirements.txt

#Install BQSKIT:Install BQSKIT:
pip install .

#Note: pymetis may require a compatible Python version (e.g., <=3.11) and a C++ compiler (e.g., g++). On Ubuntu, install build tools with:
sudo apt update && sudo apt install -y build-essential g++ cmake

"""

This method might not match the exact Conda-Forge pymetis version (2023.1.1) used in development.

Dependencies
Core BQSKIT Dependencies: numpy, scipy, lark, bqskitrs, dill, typing_extensions.
Partitioning Tools:
pymetis (2023.1.1 via Conda-Forge): Graph partitioning.
kahypar (1.3.5 via PyPI): Hypergraph partitioning.
See requirements.txt for version details if using the pip method.

"""
=======
# bqskit_withHyper
bqskit with new algorithm integrated(hyperGraph partitioning)
>>>>>>> 9f0473543b8ce68fe893bdf0597f130fe4685e2c
