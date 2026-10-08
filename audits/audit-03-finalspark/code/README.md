# Audit #3 code
- `audit_lib.py`: canonical Stage C pipeline (150-node leaky reservoir, spectral radius 0.9,
  sparsity 0.1, alphas [0.005, 0.01, 0.02], seeds [7, 11, 22], quality-weighted ridge readout
  lambda 1e-3, 20-rep label-shuffle nulls). Copied unchanged from Audit #1.
- Workstream scripts write CSVs to ../results/ and finish with a findings summary.
- Python: ~/workspace/.venv-audit/bin/python
