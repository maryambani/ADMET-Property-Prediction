import numpy as np

from admet.featurize import smiles_to_fingerprint


def test_ethanol_fingerprint_shape():
    fp = smiles_to_fingerprint("CCO", radius=2, n_bits=2048)
    assert fp.shape == (2048,)
    assert fp.dtype == np.float32


def test_invalid_smiles_raises():
    try:
        smiles_to_fingerprint("not_a_molecule")
        assert False, "expected ValueError"
    except ValueError:
        pass
