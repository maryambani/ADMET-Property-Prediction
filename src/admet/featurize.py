import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_generators = {}


def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return mol


def _get_generator(radius: int, n_bits: int):
    key = (radius, n_bits)
    if key not in _generators:
        _generators[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
    return _generators[key]


def fingerprint_and_bits(mol, radius: int = 2, n_bits: int = 2048):
    # bitInfo maps an on-bit to the atoms that set it: (center atom, radius)
    gen = _get_generator(radius, n_bits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = gen.GetFingerprint(mol, additionalOutput=ao)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    bits = {
        int(bit): [(int(atom), int(rad)) for atom, rad in hits]
        for bit, hits in ao.GetBitInfoMap().items()
    }
    return arr, bits


def mol_to_fingerprint(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    arr, _ = fingerprint_and_bits(mol, radius=radius, n_bits=n_bits)
    return arr


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = smiles_to_mol(smiles)
    return mol_to_fingerprint(mol, radius=radius, n_bits=n_bits)
