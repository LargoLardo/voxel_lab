from pathlib import Path

from morphovoxel.cleanup import find_disposable, remove_disposable


def test_cleanup_is_dry_by_default_and_preserves_runtime_data(tmp_path: Path):
    disposable = [
        tmp_path / ".tmp-old" / "result.txt",
        tmp_path / ".test-cache" / "result.txt",
        tmp_path / ".pytest_cache" / "state",
        tmp_path / "morphovoxel" / "__pycache__" / "module.pyc",
        tmp_path / "build" / "package.whl",
        tmp_path / ".coverage",
    ]
    preserved = [
        tmp_path / ".venv" / "Lib" / "module.pyc",
        tmp_path / "runs" / ".tmp-active.json",
        tmp_path / "variant_archive" / "tree.npz",
        tmp_path / "graphify-out" / ".pytest_cache" / "state",
        tmp_path / "tests" / "test_model.py",
        tmp_path / "notes.tmp",
    ]
    for path in disposable + preserved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")

    preview, failures = remove_disposable(tmp_path)
    assert failures == []
    assert set(preview) == {
        tmp_path / ".tmp-old",
        tmp_path / ".test-cache",
        tmp_path / ".pytest_cache",
        tmp_path / "morphovoxel" / "__pycache__",
        tmp_path / "build",
        tmp_path / ".coverage",
    }
    assert all(path.exists() for path in disposable + preserved)

    removed, failures = remove_disposable(tmp_path, apply=True)
    assert removed == preview
    assert failures == []
    assert all(not path.exists() for path in disposable)
    assert all(path.exists() for path in preserved)
    assert find_disposable(tmp_path) == []
