# ADMET Property Predictor

Predicting toxicity from SMILES strings using Tox21. Starting with multi-task classification (12 assays) and working up to a Streamlit demo + HF Spaces deploy.

Stack: PyTorch, RDKit, W&B, Streamlit

## setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## structure

```
configs/     - experiment configs
data/        - datasets (note: not tracked)
src/admet/   - main package
tests/
```

## todo

- [ ] featurize tox21 (rdkit morgan fingerprints)
- [ ] baseline feedforward model in pytorch
- [ ] training loop + eval
- [ ] wandb logging
- [ ] streamlit app
- [ ] deploy to huggingface
