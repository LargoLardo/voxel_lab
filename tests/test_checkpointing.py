import pytest
import torch

from morphovoxel.checkpointing import load_checkpoint, save_checkpoint
from morphovoxel.model_2d import NeuralCA2D


def test_checkpoint_roundtrip(tmp_path):
    model = NeuralCA2D(3, 4)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    save_checkpoint(tmp_path / "model.pt", model, step=9)
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = load_checkpoint(tmp_path / "model.pt", model)
    assert payload["step"] == 9
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())


def test_missing_checkpoint_explains_prerequisite(tmp_path):
    with pytest.raises(FileNotFoundError, match="launch the Full experiment"):
        load_checkpoint(tmp_path / "missing.pt", torch.nn.Linear(2, 2))
