import pandas as pd
from omegaconf import OmegaConf
from src.utils.postprocess_callback import PostProcessCallback, LinePlot


from src.utils.benchmark_run import run_benchmark, task_registry
from tests.test_evaluate import DummyTask

task_registry["test_task"] = DummyTask

def cfg():
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

    task_registry["test_task"] = DummyTask

    run_benchmark(cfg())

    base = tmp_path / "outputs/DummyTask_NPE/sims_10"
    assert base.exists(), "Output directory was not created"

    single_metrics_file = base / "metrics_c2st.csv"
    assert single_metrics_file.exists(), "metrics_c2st.csv was not created"

def multirun_cfg(metric_name: str):
    return OmegaConf.create({
        "task": {"name": "test_task"},
        "inference": {
            "method": "NPE",
            "num_simulations": 10,
            "num_observations": 2,
            "num_posterior_samples": 5,
        },
        "metric": {"name": metric_name},
        "random_seed": 42
    })


def test_postprocess_multirun_consolidation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    task_registry["test_task"] = DummyTask

    # Simulate two runs with two different metrics (as hydra sweeps would do)
    for metric in ["ppc", "c2st"]:
        cfg = multirun_cfg(metric)
        run_benchmark(cfg)

    # Set up Fake Hydra multirun path structure
    sweep_dir = tmp_path / "multirun"
    (sweep_dir / "0" / ".hydra").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "1" / ".hydra").mkdir(parents=True, exist_ok=True)

    # Set up .hydra/config.yaml
    job_cfg = {"task": {"name": "test_task"}, 
               "inference": {"method": "NPE", "num_simulations": 10}}
    
    (sweep_dir / "0" / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(job_cfg)))
    (sweep_dir / "1" / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(job_cfg)))


    # Set up Top Level Config for PostProcessCallback
    top_level_cfg = OmegaConf.create({
        "hydra": {
            "sweep": {
                "dir": str(sweep_dir)
            }
        },
        "task": {"name": "test_task"}
    })

    # Deactivate plotting to avoid Warnings during tests
    monkeypatch.setattr(LinePlot, "run", lambda self, *args, **kwargs: None)


    # Run PostProcessCallback
    PostProcessCallback().on_multirun_end(top_level_cfg)

    # Check consolidated files
    # Check first consolidation step: per metric files for each simulation
    base = tmp_path / "outputs/DummyTask_NPE"
    simulations_path = base / "sims_10"

    assert (simulations_path / "metrics_ppc.csv").exists(), "metrics_ppc.csv was not created"
    assert (simulations_path / "metrics_c2st.csv").exists(), "metrics_c2st.csv was not created"

    assert (simulations_path / "metrics.csv").exists(), "metrics.csv was not created -> Consolidation of metric-specific files failed"
    

    # Check second consolidation step: consolidated file across simulations
    assert (base / "metrics_all.csv").exists(), "metrics_all.csv was not created -> Consolidation across simulations failed"


    # Check contents of one of the metric files 
    metrics_file = tmp_path / "outputs/DummyTask_NPE/sims_10/metrics_c2st.csv"
    df = pd.read_csv(metrics_file)
    assert len(df) == 2, "Expected two rows for two observations"



    