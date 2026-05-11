# PSP 6DoF Rocket Simulator

Rocket flight dynamics simulator using Lie group mathematics for geometrically exact integration, also computationally more efficient.

## Installation

```bash
git clone https://github.com/Purdue-Space-Program/psp-6dof.git
cd psp-6dof
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
# For development:
pip install -e ".[dev]"
# Test the code
pytest
```