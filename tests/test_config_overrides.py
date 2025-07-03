import os
import sys
import tempfile
import shutil
import pytest
from hydra import initialize, compose
from omegaconf import OmegaConf
import torch
import src.utils.benchmark_run as benchmark_run_mod

@pytest.mark.parametrize("override_key,override_value", [
    ("inference.num_simulations", 123),
    ("inference.num_observations", 7),
    ("inference.num_posterior_samples", 33),
])
def test_config_override(monkeypatch, override_key, override_value):
    # Setup hydra config
    with initialize(version_base="1.3", config_path="../src/configs"):
        cfg = compose(config_name="main", overrides=[f"{override_key}={override_value}"])
        # Run the benchmark, which should use the overridden config value
        # Monkeypatch run_inference to check the value it receives
        called = {}

        def fake_run_inference(task, method_name, num_simulations, num_posterior_samples, num_observations, seed=None, config=None):
            called['num_simulations'] = num_simulations
            called['num_observations'] = num_observations
            called['num_posterior_samples'] = num_posterior_samples
            return torch.ones(10, 2)  # fake samples

        monkeypatch.setattr(benchmark_run_mod, "run_inference", fake_run_inference)
        # Also monkeypatch evaluation to avoid running metrics
        import src.evaluation.evaluate_inference as eval_mod
        monkeypatch.setattr(eval_mod, "evaluate_inference", lambda *a, **kw: None)

        benchmark_run_mod.run_benchmark(cfg)

        # Figure out which key we are testing
        param_name = override_key.split(".")[-1]
        assert called[param_name] == override_value, f"{param_name} not overridden: expected {override_value}, got {called[param_name]}"