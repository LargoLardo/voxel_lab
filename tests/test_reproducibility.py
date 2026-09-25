import torch
import pytest

from morphovoxel.random_utils import resolve_device, seed_everything
from morphovoxel.rollout import rollout


@pytest.mark.parametrize("deterministic", [True, False])
def test_random_seed_reproduces_values(deterministic):
    seed_everything(9, deterministic=deterministic)
    first = torch.rand(5)
    seed_everything(9, deterministic=deterministic)
    assert torch.equal(first, torch.rand(5))
    assert torch.are_deterministic_algorithms_enabled() == deterministic


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_auto_device_prefers_available_accelerators(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device("auto").type == expected
    assert resolve_device("cpu").type == "cpu"
    if not mps:
        with pytest.raises(ValueError, match="MPS"):
            resolve_device("mps")


def test_device_selection_and_rollout_observer():
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda")

    seen = []
    final, frames = rollout(lambda state, genome: state + 1, torch.zeros(1), 3, capture_every=2, on_step=lambda step, state: seen.append(step))
    assert seen == [0, 1, 2, 3]
    assert final.item() == 3
    assert [frame.item() for frame in frames] == [0, 2, 3]
