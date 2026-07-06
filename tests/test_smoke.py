import subprocess, sys, os, glob

def test_synthetic_end_to_end(tmp_path):
    out = tmp_path / "res"
    cmd = [sys.executable, "src/calibration_experiment.py", "--synthetic",
           "--seeds", "2", "--no-tune", "--outdir", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(out / "metrics_baseline_agg.csv")
    assert glob.glob(str(out / "fig*_*.png"))
    assert os.path.exists(out / "raw_splits")
