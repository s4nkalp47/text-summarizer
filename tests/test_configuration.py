from pathlib import Path

import pytest

from textSummarizer.config.configuration import ConfigurationManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a ConfigurationManager from the real YAML files, creating directories under a temp folder."""
    monkeypatch.chdir(tmp_path)

    def build(smoke: bool) -> ConfigurationManager:
        return ConfigurationManager(PROJECT_ROOT / "config/config.yaml", PROJECT_ROOT / "params.yaml", smoke=smoke)

    return build


def test_smoke_training_never_writes_to_the_real_model_folder(manager) -> None:
    full = manager(smoke=False).get_model_trainer_config()
    smoke = manager(smoke=True).get_model_trainer_config()

    assert full.model_path == Path("artifacts/model_trainer/model")
    assert smoke.model_path != full.model_path
    assert smoke.tokenizer_path != full.tokenizer_path
    assert smoke.root_dir != full.root_dir


def test_evaluation_reads_the_model_its_mode_trains(manager) -> None:
    for smoke in (False, True):
        config_manager = manager(smoke=smoke)
        trained = config_manager.get_model_trainer_config()
        evaluated = config_manager.get_model_evaluation_config()

        assert evaluated.model_path == trained.model_path
        assert evaluated.tokenizer_path == trained.tokenizer_path


def test_smoke_evaluation_results_are_kept_separate(manager) -> None:
    full = manager(smoke=False).get_model_evaluation_config()
    smoke = manager(smoke=True).get_model_evaluation_config()

    assert {smoke.metrics_path, smoke.samples_path, smoke.comparison_path}.isdisjoint(
        {full.metrics_path, full.samples_path, full.comparison_path}
    )
    assert (full.num_samples, smoke.num_samples) == (100, 10)
    assert (full.base_model_ckpt, smoke.base_model_ckpt) == ("google/pegasus-cnn_dailymail", "t5-small")
