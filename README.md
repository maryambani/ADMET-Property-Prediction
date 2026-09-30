# ADMET Property Predictor

Multi-task toxicity prediction from SMILES strings on the Tox21 benchmark. A feedforward
network on Morgan fingerprints predicts activity across 12 nuclear-receptor and
stress-response assays, with a Streamlit app for interactive predictions.

Stack: PyTorch, RDKit, scikit-learn, Weights & Biases, Streamlit

## results

Held-out test set, 1,173 molecules, single seed, random split.

| assay         | test ROC-AUC |
| ------------- | -----------: |
| SR-MMP        |        0.875 |
| SR-ATAD5      |        0.872 |
| NR-AR-LBD     |        0.858 |
| NR-AhR        |        0.856 |
| NR-PPAR-gamma |        0.839 |
| NR-Aromatase  |        0.820 |
| NR-ER-LBD     |        0.815 |
| SR-HSE        |        0.805 |
| SR-ARE        |        0.799 |
| SR-p53        |        0.773 |
| NR-AR         |        0.770 |
| NR-ER         |        0.669 |
| **mean**      |    **0.812** |

Model selection used a separate validation split (best epoch 10, val AUC 0.779). The test
set was evaluated once, after training, with the best-on-validation weights.

## methods

### data

Tox21 as distributed by MoleculeNet / DeepChem (`tox21.csv.gz`, 7,831 compounds). Each
compound has binary labels for up to 12 assays; ~17% of labels are missing because not every
compound was screened in every assay. Missing labels are kept as missing (not imputed or
dropped) and masked out of the loss. 8 compounds whose SMILES RDKit could not parse were
dropped, leaving 7,823.

Split: 70 / 15 / 15 train / validation / test, random, seed 42 (5,477 / 1,173 / 1,173).

### featurization

Morgan fingerprints, radius 2, 2,048 bits (`rdkit.Chem.rdFingerprintGenerator`). No other
descriptors. Fingerprint parameters are stored in the checkpoint so inference always matches
training.

### model

Feedforward network: 2048 -> 512 -> 256 -> 12. ReLU activations, dropout after each hidden
layer, a single linear output layer producing one logit per assay. All 12 assays share the
hidden layers (hard parameter sharing); the output layer is the only task-specific part.

### training

- loss: binary cross-entropy with logits, averaged only over labelled (compound, assay) pairs
- optimizer: Adam, lr 1e-3, batch size 64
- dropout: 0.5
- early stopping on mean validation ROC-AUC, patience 5, max 50 epochs
- the checkpoint with the best validation AUC is kept; the test set is scored once with it

Training runs in ~20 s on a laptop CPU.

### evaluation

ROC-AUC per assay, computed only over compounds with a label for that assay, then averaged
across the 12 assays. AUC is used rather than accuracy because actives are rare (3-16% of labelled
compounds depending on the assay), so accuracy is dominated by the negative class.

## what was tried

**Dropout 0.3 vs 0.5.** On an earlier 80/20 split (no test set), raising dropout from 0.3 to
0.5 moved best validation AUC from 0.805 to 0.811 and delayed the best epoch from 4 to 7,
with less separation between train and validation loss at the end of training. The gain is
smaller than the difference observed between two random splits of the same data (~0.03), so
it should not be read as significant without repeating over several seeds. 0.5 was kept.

**Training length.** With 50 fixed epochs the model reaches near-zero training loss while
validation loss more than doubles after the first ~5-10 epochs. Early stopping with patience
5 now ends training at epoch ~15 with no loss in validation AUC.

## limitations

- **Single seed, random split.** Tox21 contains many near-duplicate scaffolds; a random
  split lets close analogues land on both sides, so these numbers are optimistic relative to
  a scaffold split. Variance across seeds has not been measured.
- **NR-ER is weak (0.669).** Estrogen-receptor activity is the hardest assay for
  fingerprint models here, consistent with published Tox21 results.
- **Probabilities are not calibrated.** The app reports raw sigmoid outputs. A value of 0.5
  does not mean a 50% chance of activity.
- **Not a safety tool.** This is a learning project and cannot be used to make decisions about real compounds.

## running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python scripts/prepare_data.py   # download + featurize tox21
PYTHONPATH=src python scripts/train.py          # train, writes checkpoints/best.pt
streamlit run app/app.py                        # interactive predictions
PYTHONPATH=src pytest tests/ -q
```

Experiment tracking is off by default. To log to W&B:

```bash
wandb login
PYTHONPATH=src python scripts/train.py --wandb
```

Hyperparameter sweep:

```bash
wandb sweep configs/sweep.yaml
wandb agent <sweep_id>
```

## structure

```
configs/       default.yaml (all hyperparameters), sweep.yaml (W&B sweep)
data/          raw + processed tox21 (not tracked)
checkpoints/   trained models (not tracked)
scripts/       prepare_data.py, train.py, smoke_model.py
src/admet/
  featurize.py   SMILES -> Morgan fingerprint
  data.py        download, featurize, save X.npy / y.npy
  dataset.py     PyTorch Dataset with label mask
  model.py       feedforward multi-task network
  metrics.py     masked BCE, per-task ROC-AUC
  train.py       split, train loop, early stopping, test eval
  predict.py     load checkpoint, predict from SMILES, draw molecule
app/           Streamlit app + assay descriptions with sources
tests/
```

## next

- repeat training over 5 seeds and report mean +/- std
- scaffold split for a more realistic estimate
- run the W&B sweep
- Dockerfile

## references

- Huang R, et al. (2016). Tox21Challenge to Build Predictive Models of Nuclear Receptor and
  Stress Response Pathways. _Front. Environ. Sci._ 3:85. https://doi.org/10.3389/fenvs.2015.00085
- Wu Z, et al. (2018). MoleculeNet: a benchmark for molecular machine learning. _Chem. Sci._
  9:513-530. https://doi.org/10.1039/C7SC02664A
- Rogers D, Hahn M. (2010). Extended-connectivity fingerprints. _J. Chem. Inf. Model._
  50:742-754. https://doi.org/10.1021/ci100050t

Assay-level references are listed in the app.
