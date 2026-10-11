from pathlib import Path

import pytest
from box import ConfigBox

from textSummarizer.utils.common import create_directories, read_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_read_yaml_returns_configbox_with_attribute_access(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("stage:\n  root_dir: artifacts/stage\n  num_samples: 3\n")

    config = read_yaml(path)

    assert isinstance(config, ConfigBox)
    assert config.stage.root_dir == "artifacts/stage"
    assert config.stage.num_samples == 3


@pytest.mark.parametrize("content", ["", "   \n", "# only a comment\n"])
def test_read_yaml_raises_clear_error_on_empty_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text(content)

    with pytest.raises(ValueError, match="empty"):
        read_yaml(path)


def test_read_yaml_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "missing.yaml")


def test_project_params_learning_rate_is_a_float() -> None:
    # PyYAML parses "5e-5" (no dot) as a string; guard against that regression.
    params = read_yaml(PROJECT_ROOT / "params.yaml")

    assert isinstance(params.training.learning_rate, float)


def test_create_directories_creates_nested_dirs(tmp_path: Path) -> None:
    paths = [tmp_path / "a" / "b", tmp_path / "c"]

    create_directories(paths)

    assert all(p.is_dir() for p in paths)


def test_create_directories_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "existing"
    path.mkdir()
    (path / "keep.txt").write_text("data")

    create_directories([path])

    assert (path / "keep.txt").read_text() == "data"
