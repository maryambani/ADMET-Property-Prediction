import numpy as np

from admet.featurize import fingerprint_and_bits, smiles_to_fingerprint, smiles_to_mol


def test_ethanol_fingerprint_shape():
    fp = smiles_to_fingerprint("CCO", radius=2, n_bits=2048)
    assert fp.shape == (2048,)
    assert fp.dtype == np.float32


def test_bitinfo_covers_on_bits():
    mol = smiles_to_mol("c1ccccc1O")
    fp, bits = fingerprint_and_bits(mol, radius=2, n_bits=2048)
    on = set(np.flatnonzero(fp).tolist())
    assert set(bits) == on
    assert all(atom >= 0 for hits in bits.values() for atom, _ in hits)


def test_invalid_smiles_raises():
    try:
        smiles_to_fingerprint("not_a_molecule")
        assert False, "expected ValueError"
    except ValueError:
        pass
