import hydra
from hydra.core.global_hydra import GlobalHydra
import os

def test_override_task_tau_m():
    # Wechsel ins Projekt-Root zur Laufzeit
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize(config_path="src/configs", version_base="1.3"):
        cfg = hydra.compose(config_name="main", overrides=["task.tau_m=0.99"])
        assert cfg.task.tau_m == 0.99