# Installation

GLPBench requires Python 3.10, 3.11, or 3.12.

## Install from PyPI

```bash
python -m pip install glpbench
```

Verify the installation and inspect the available commands:

```bash
python -c "import glpbench; print(glpbench.__version__)"
glpbench --help
```

## Install from source

Clone the repository and install it into an isolated environment:

```bash
git clone https://github.com/yorgancioglukaan/glpbench.git
cd glpbench
python -m pip install .
```

An editable installation is useful when developing GLPBench:

```bash
python -m pip install -e .
```
