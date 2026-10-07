"""Train the plain per-patient CNN-LSTM (v0, ordinary loss) for every HUPA-UCM patient.

Same code and settings as the AZT1D and OhioT1DM plain models (run_with_checkpoints, 30 epochs,
early stopping on validation). Checkpoints go to data/processed/checkpoints/hupa_ucm_cnn_lstm_v0/,
and patients already there (27) are reused, not retrained.

Run:    python work/5_10/scripts/train_hupa_all_v0.py
"""
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from azt1d import hupa  # noqa: E402
from azt1d.glimmer.train import run_with_checkpoints  # noqa: E402

t0 = time.time()
ids = hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR)
df = pd.concat([hupa.load_subject(s, ROOT / hupa.DEFAULT_DIR) for s in ids], ignore_index=True)
print(f"{len(ids)} patients, {len(df):,} rows", flush=True)
summary, _ = run_with_checkpoints(df, ROOT / "data" / "processed" / "checkpoints" / "hupa_ucm_cnn_lstm_v0",
                                  epochs=30, region_weights=None)
print(summary.round(2).to_string(), flush=True)
print(f"runtime {time.time() - t0:.0f}s", flush=True)
