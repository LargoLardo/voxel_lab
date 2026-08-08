import torch
import pytest

from morphovoxel.random_utils import resolve_device, seed_everything
from morphovoxel.rollout import rollout


def test_random_seed_reproduces_values():
    seed_everything(9)
    first = torch.rand(5)
    seed_everything(9)
    assert torch.equal(first, torch.rand(5))


def test_device_selection_and_rollout_observer():
    assert resolve_device("auto").type in {"cpu", "cuda"}
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda")

    seen = []
    final, frames = rollout(lambda state, genome: state + 1, torch.zeros(1), 3, capture_every=2, on_step=lambda step, state: seen.append(step))
    assert seen == [0, 1, 2, 3]
    assert final.item() == 3
    assert [frame.item() for frame in frames] == [0, 2, 3]
