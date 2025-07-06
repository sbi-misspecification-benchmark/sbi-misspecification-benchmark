import subprocess
import shutil
from pathlib import Path

# tests if benchmark.py creates a folder for each observation
def test_run_script_creates_expected_outputs():
    project_root = Path(__file__).parent.parent


    result = subprocess.run(
        ["python", "src/run.py", "--config-path", "configs", "--config-name", "test_config"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    print(result.stdout)
    print(result.stderr)

    base = project_root / "outputs/TestTask_NPE/sims_100/"
    for idx in range(3):
        obs_dir = base / f"obs_{idx}"
        assert obs_dir.exists(), f"Observation folder {obs_dir} was not created"
        assert (obs_dir / "posterior_samples.pt").exists(), f"posterior_samples.pt missing in {obs_dir}"

    # deletes files
    shutil.rmtree(project_root / "outputs", ignore_errors=True)
