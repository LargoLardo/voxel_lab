from morphovoxel.training import train


def test_cpu_smoke_training_2d_and_3d(tmp_path):
    common = {"runs_root": str(tmp_path), "device": "cpu", "iterations": 1, "rollout_steps": [1, 1], "batch_size": 1, "hidden_channels": 2, "model_width": 4, "fire_rate": 1.0, "live_preview": True, "frame_every": 1}
    two = train({**common, "run_name": "2d", "world_size": 12, "materials": 3}, dimensions=2)
    three = train({**common, "run_name": "3d", "world_size": 10, "materials": 4}, dimensions=3)
    assert (two / "checkpoints" / "latest.pt").exists() and (two / "visualizations" / "live.png").exists()
    assert (three / "checkpoints" / "latest.pt").exists() and (three / "visualizations" / "live.png").exists()
