from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mnist_pca_muon as mpm
root = Path(__file__).resolve().parents[1]
cfg = mpm.load_config(root / "configs" / "smoke.json")
cmpm = mpm
cmpm.run_experiment(cfg, groups=("development",), rebuild_problem=True, force_runs=True)
print(mpm.analyze_experiment(cfg, groups=("development",)))
