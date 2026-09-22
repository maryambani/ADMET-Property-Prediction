from pathlib import Path

import pytest

from admet.predict import Predictor, draw_molecule

needs_checkpoint = pytest.mark.skipif(
    not Path("checkpoints/best.pt").exists(), reason="no trained checkpoint"
)


@needs_checkpoint
def test_predict_returns_prob_per_task():
    predictor = Predictor()
    probs = predictor.predict("CCO")
    assert len(probs) == len(predictor.tasks)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


@needs_checkpoint
def test_predict_invalid_smiles_raises():
    predictor = Predictor()
    with pytest.raises(ValueError):
        predictor.predict("not_a_molecule")


def test_draw_molecule_returns_image():
    img = draw_molecule("CCO", size=(100, 100))
    assert img.size == (100, 100)
