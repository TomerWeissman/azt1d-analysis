"""Train the plain-loss and GLIMMER fixed-weight CNN-LSTMs on one HUPA-UCM patient's full record.

Usage: python scripts/train_hupa_patient.py 27
Checkpoints land in data/processed/checkpoints/hupa_ucm_cnn_lstm_{v0,v1}/ and are reused if present.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import hupa
from azt1d import reference as ref
from azt1d.glimmer.train import run_with_checkpoints

subject_id = int(sys.argv[1])
df = hupa.load_subject(subject_id, ROOT / hupa.DEFAULT_DIR)
print(f"HUPA{subject_id:04d}P: {len(df):,} rows, {df['EventDateTime'].min()} to {df['EventDateTime'].max()}", flush=True)

ckpt_root = ROOT / "data" / "processed" / "checkpoints"
runs = {
    "hupa_ucm_cnn_lstm_v0": None,
    "hupa_ucm_cnn_lstm_v1": ref.GLIMMER_PAPER_WEIGHTS["cnn_lstm"],
}
for name, weights in runs.items():
    start = time.time()
    summary, _ = run_with_checkpoints(df, ckpt_root / name, epochs=30, region_weights=weights)
    print(f"{name}: RMSE {summary['rmse'].iloc[0]:.2f}  MAE {summary['mae'].iloc[0]:.2f}  ({time.time() - start:.0f}s)", flush=True)
