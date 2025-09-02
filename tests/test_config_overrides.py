import pytest
from hydra import initialize, compose
import torch
import src.utils.benchmark_run as benchmark_run_mod

def _parse_hydra_float(val):
    # Helper for tests: get first float from comma-separated Hydra string
    if isinstance(val, str) and "," in val:
        return float(val.split(",")[0])
    return float(val)

@pytest.mark.parametrize("override_key,override_value", [
    ("inference.num_simulations", 123),
    ("inference.num_observations", 7),
    ("inference.num_posterior_samples", 33),
])
def test_config_override(monkeypatch, override_key, override_value):
    with initialize(version_base="1.3", config_path="../src/configs"):
        cfg = compose(config_name="main", overrides=[f"{override_key}={override_value}"])

        # Patch tau_m and lambda_val
        if hasattr(cfg.task, "tau_m"):
            cfg.task.tau_m = _parse_hydra_float(cfg.task.tau_m)
        if hasattr(cfg.task, "lambda_val"):
            cfg.task.lambda_val = _parse_hydra_float(cfg.task.lambda_val)

        called = {}
        def fake_run_inference(task, method_name, num_simulations, num_posterior_samples, num_observations, seed=None, config=None,observations=None):
            called['num_simulations'] = num_simulations
            called['num_observations'] = num_observations
            called['num_posterior_samples'] = num_posterior_samples
            return torch.ones(10, 2)

        monkeypatch.setattr(benchmark_run_mod, "run_inference", fake_run_inference)
        monkeypatch.setattr(benchmark_run_mod, "evaluate_inference", lambda *a, **kw: None)

        benchmark_run_mod.run_benchmark(cfg)
        param_name = override_key.split(".")[-1]
        assert called[param_name] == override_value, f"{param_name} not overridden: expected {override_value}, got {called[param_name]}"