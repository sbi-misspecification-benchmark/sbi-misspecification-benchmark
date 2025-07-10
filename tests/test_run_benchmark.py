import pandas as pd
from omegaconf import OmegaConf
from src.utils.benchmark_run import run_benchmark




def test_cfg():
    return OmegaConf.create({
        "task": {"name": "test_task"},
        "inference": {
            "method": "npe",
            "num_simulations": 10,
            "num_observations": 2,
            "num_posterior_samples": 5,
        },
        "metric": {"name": "c2st"},
        "random_seed": 42
    })

def test_run_creates_expected_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_benchmark(test_cfg())

    base = tmp_path / "outputs/test_task_NPE/sims_10"
    assert base.exists(), "Output directory was not created"

    metrics_file = base / "metrics.csv"
    assert metrics_file.exists(), "metrics.csv was not created"

def test_no_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_benchmark(test_cfg())
    metrics_file = tmp_path / "outputs/test_task_NPE/sims_10"/"metrics.csv"
    df = pd.read_csv(metrics_file)
    assert len(df) == 2, "Expected two rows for two observation"



