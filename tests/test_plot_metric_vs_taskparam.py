import csv, os
from pathlib import Path
from src.utils.plot_metric_vs_taskparam import plot_metric_vs_taskparam


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","value","task","method","num_simulations","observation_idx","tau_m"])
        w.writerows(rows)


def test_creates_png(tmp_path):
    csv_path = tmp_path / "metrics_all.csv"
    rows = [
        ["C2ST", 0.70, "Task", "Method", 1000, 0, 0.5],
        ["C2ST", 0.73, "Task", "Method", 1000, 1, 1.0],
        ["C2ST", 0.76, "Task", "Method", 1000, 2, 1.5],
    ]
    _write_csv(csv_path, rows)

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)  # ensure outputs/ is under tmp
        plot_metric_vs_taskparam(str(csv_path), "C2ST", "tau_m")
        out = tmp_path / "outputs" / "Task_Method" / "plots" / "C2ST__vs__tau_m.png"
        assert out.exists() and out.stat().st_size > 0

    finally:
        os.chdir(cwd)
