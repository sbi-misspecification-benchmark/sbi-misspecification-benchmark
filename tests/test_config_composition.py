import hydra
from hydra.core.global_hydra import GlobalHydra

def test_config_composition():
    # Clear Hydra state for safety if running multiple Hydra jobs
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    @hydra.main(config_path="src/configs", config_name="main", version_base="1.3")
    def cfg_test(cfg):
        assert cfg.task.name == "misspecified_likelihood"
        assert cfg.task.tau_m == 0.2
        assert cfg.inference.method == "NPE"
        assert cfg.inference.num_simulations == 100
        assert "c2st" in cfg.metrics.metrics
        assert "ppc" in cfg.metrics.metrics
        assert cfg.random_seed == 42

    cfg_test()