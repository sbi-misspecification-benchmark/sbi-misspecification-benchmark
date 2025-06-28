import os
import hydra
from hydra.core.global_hydra import GlobalHydra

def test_override_task_tau_m():
    # Ermittle den absoluten Pfad zum Projekt-Root
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # Ermittle den absoluten Pfad zum Config-Verzeichnis
    config_abs = os.path.join(project_root, "src", "configs")
    # Berechne den RELATIVEN Pfad von CWD zu config_abs
    config_path = os.path.relpath(config_abs, os.getcwd())

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize(config_path=config_path, version_base="1.3"):
        cfg = hydra.compose(config_name="main", overrides=["task.tau_m=0.99"])
        assert cfg.task.tau_m == 0.99