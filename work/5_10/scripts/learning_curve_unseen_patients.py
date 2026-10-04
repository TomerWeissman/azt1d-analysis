"""H18 and H20: which uncertainty behaves like epistemic uncertainty?

Trains a 3-member deep ensemble (Gaussian mean and variance per member) on the pooled
HUPA-UCM patients, then measures two variances on held-out windows:

  - epistemic: variance of the members' means (disagreement between members)
  - aleatoric: average of the members' predicted noise variances

H18 (learning curve): train on 10%, 30%, and 100% of the training windows. Epistemic should
    fall as data grows (at least 20% from smallest to largest). Aleatoric should stay put
    (within 10%).
H20 (unseen patients): on the full-data ensemble, compare windows from the source patients'
    held-out validation split (familiar) with windows from five patients the models never saw
    (unseen). Epistemic should be at least 20% higher on unseen windows. Aleatoric should be
    within 10%.

The thresholds are fixed here, before any run. Patient split and data handling match the
28_9 fine-tuning experiment. The model code is a copy of that script's, not an import, because
only its results were carried forward into this week.

Run:    python work/5_10/scripts/learning_curve_unseen_patients.py
Writes: work/5_10/results/learning_curve.csv, work/5_10/results/verdicts.txt,
        work/5_10/figures/learning_curve.png
"""
from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import hupa, plotting  # noqa: E402
from azt1d.glimmer.sequences import make_windows  # noqa: E402
from azt1d.glimmer.train import _iter_batches  # noqa: E402

RNG_SEED = 42
N_TARGETS = 5
FEATURES = ["glucose", "bolus_volume_delivered", "carb_input", "heart_rate"]
TARGET = "glucose"
LOOKBACK, HORIZON = 36, 6
HIDDEN, FC_UNITS, DROPOUT = 128, 64, 0.1
LR, BATCH, EPOCHS, PATIENCE = 1e-4, 1024, 50, 5
N_MEMBERS = 3
FRACTIONS = (0.1, 0.3, 1.0)
ROWS_PER_DAY, TEST_DAYS, SOURCE_CAP_DAYS = 288, 3, 30
DECIDE_DROP, DECIDE_UNSEEN_RATIO, DECIDE_FLAT = 0.20, 1.20, 0.10


# ---------------------------------------------------------------- data

def load_patient(sid: int) -> pd.DataFrame:
    path = ROOT / hupa.DEFAULT_DIR / f"HUPA{sid:04d}P.csv"
    raw = pd.read_csv(path, sep=";", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df = raw[["time", "glucose", "bolus_volume_delivered", "carb_input", "heart_rate"]].copy()
    df["bolus_volume_delivered"] = df["bolus_volume_delivered"].fillna(0.0)
    df["carb_input"] = df["carb_input"].fillna(0.0)
    return df


def windows_for(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return make_windows(df, FEATURES, TARGET, LOOKBACK, HORIZON)


@dataclass
class Scaler:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.x_mean) / self.x_std).astype(np.float32)


def build_pool(source_ids: list[int]):
    """Each source patient: first 80% of windows for training, last 20% for validation. Pooled.
    Scaler is fit on the pooled training windows only."""
    cap = SOURCE_CAP_DAYS * ROWS_PER_DAY
    Xtr, ytr, Xva, yva = [], [], [], []
    for sid in source_ids:
        X, y = windows_for(load_patient(sid).iloc[:cap])
        n_val = max(1, int(round(len(X) * 0.2)))
        cut = len(X) - n_val
        Xtr.append(X[:cut]); ytr.append(y[:cut]); Xva.append(X[cut:]); yva.append(y[cut:])
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    Xva, yva = np.concatenate(Xva), np.concatenate(yva)
    xs = StandardScaler().fit(Xtr.reshape(-1, len(FEATURES)))
    scaler = Scaler(xs.mean_.astype(np.float32), xs.scale_.astype(np.float32), float(ytr.mean()), float(ytr.std()))
    ys = lambda y: ((y - scaler.y_mean) / scaler.y_std).astype(np.float32)  # noqa: E731
    return (scaler.transform_x(Xtr), ys(ytr), scaler.transform_x(Xva), ys(yva), scaler, Xtr)


def unseen_windows(target_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Test-block windows (last 3 days) of the five target patients. Never used in training."""
    n_test = TEST_DAYS * ROWS_PER_DAY
    Xs, ys = [], []
    for sid in target_ids:
        df = load_patient(sid)
        test_df = df.iloc[len(df) - n_test - LOOKBACK - HORIZON + 1:].reset_index(drop=True)
        X, y = windows_for(test_df)
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# ---------------------------------------------------------------- model

class GaussianLSTM(nn.Module):
    """LSTM -> final hidden state -> 64-unit layer -> dropout -> (mean, log variance)."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURES), HIDDEN, num_layers=1, batch_first=True)
        self.fc = nn.Linear(HIDDEN, FC_UNITS)
        self.dropout = nn.Dropout(DROPOUT)
        self.out = nn.Linear(FC_UNITS, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        z = self.dropout(F.relu(self.fc(h_n[-1])))
        mu, log_var = self.out(z).unbind(-1)
        return mu, log_var


def gaussian_nll(mu, log_var, y, eps: float = 1e-6):
    var = torch.exp(log_var).clamp_min(eps)
    return (0.5 * torch.log(2 * math.pi * var) + 0.5 * (y - mu) ** 2 / var).mean()


def train_model(model, X_tr, y_tr, X_va, y_va, seed: int, epochs: int, patience: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va)
    best, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in _iter_batches(X_tr, y_tr, BATCH, shuffle=True, rng=rng):
            opt.zero_grad()
            loss = gaussian_nll(*model(torch.from_numpy(xb)), torch.from_numpy(yb))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = gaussian_nll(*model(Xv), yv).item()
        if v < best:
            best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def ensemble_variances(members, X: np.ndarray, scaler: Scaler):
    """Mean prediction, epistemic variance (spread of member means), aleatoric variance (mean of
    member noise variances). Everything in mg/dL."""
    mus, alea = [], []
    with torch.no_grad():
        for m in members:
            mu, log_var = m(torch.from_numpy(X))
            mus.append(mu.numpy() * scaler.y_std + scaler.y_mean)
            alea.append(np.exp(log_var.numpy()) * scaler.y_std ** 2)
    mus, alea = np.stack(mus), np.stack(alea)
    return mus.mean(0), mus.var(0), alea.mean(0)


# ---------------------------------------------------------------- experiment

def run(fractions=FRACTIONS, n_members=N_MEMBERS, out_dir=WEEK, epochs=EPOCHS, patience=PATIENCE) -> pd.DataFrame:
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    all_ids = hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR)
    targets = sorted(rng.choice(all_ids, N_TARGETS, replace=False).tolist())
    sources = [s for s in all_ids if s not in targets]
    print(f"targets {targets}; pretraining pool from {len(sources)} patients", flush=True)

    X_tr, y_tr, X_va, y_va, scaler, _ = build_pool(sources)
    X_un_raw, y_un = unseen_windows(targets)
    X_un = scaler.transform_x(X_un_raw)
    y_va_raw = y_va * scaler.y_std + scaler.y_mean
    print(f"train {len(X_tr):,}  held-out source {len(X_va):,}  unseen target {len(X_un):,}", flush=True)

    rows = []
    for k, frac in enumerate(fractions):
        n = int(round(frac * len(X_tr)))
        idx = np.random.default_rng(RNG_SEED + k).choice(len(X_tr), n, replace=False)
        members = []
        for i in range(n_members):
            m = train_model(GaussianLSTM(), X_tr[idx], y_tr[idx], X_va, y_va,
                            seed=RNG_SEED + 1000 * k + i, epochs=epochs, patience=patience)
            members.append(m)
            print(f"  fraction {frac:.0%}: member {i + 1}/{n_members} done ({time.time() - t0:.0f}s)", flush=True)
        for name, X, y_raw in (("held-out source", X_va, y_va_raw), ("unseen target", X_un, y_un)):
            mean, epi, alea = ensemble_variances(members, X, scaler)
            rows.append({
                "train_fraction": frac, "n_train_windows": n, "eval_set": name, "n_eval_windows": len(X),
                "rmse": float(np.sqrt(np.mean((y_raw - mean) ** 2))),
                "epistemic_sd": float(np.mean(np.sqrt(epi))),
                "aleatoric_sd": float(np.mean(np.sqrt(alea))),
            })

    res = pd.DataFrame(rows)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "results" / "learning_curve.csv", index=False)
    verdicts = decide(res)
    (out_dir / "results" / "verdicts.txt").write_text(verdicts + "\n")
    print(verdicts, flush=True)
    plot(res, out_dir / "figures" / "learning_curve.png")
    print(f"runtime {time.time() - t0:.0f}s", flush=True)
    return res


def decide(res: pd.DataFrame) -> str:
    insample = res[res.eval_set == "held-out source"].sort_values("train_fraction")
    small, large = insample.iloc[0], insample.iloc[-1]
    drop = (small.epistemic_sd - large.epistemic_sd) / small.epistemic_sd
    alea_change = abs(large.aleatoric_sd - small.aleatoric_sd) / small.aleatoric_sd
    h18 = "SUPPORTED" if drop >= DECIDE_DROP and alea_change < DECIDE_FLAT else "NOT SUPPORTED"
    full = res[res.train_fraction == large.train_fraction].set_index("eval_set")
    ratio = full.loc["unseen target", "epistemic_sd"] / full.loc["held-out source", "epistemic_sd"]
    alea_gap = abs(full.loc["unseen target", "aleatoric_sd"] / full.loc["held-out source", "aleatoric_sd"] - 1)
    h20 = "SUPPORTED" if ratio >= DECIDE_UNSEEN_RATIO and alea_gap < DECIDE_FLAT else "NOT SUPPORTED"
    return "\n".join([
        f"H18 (epistemic shrinks with data, aleatoric flat): {h18}",
        f"  epistemic {small.epistemic_sd:.3f} -> {large.epistemic_sd:.3f} SD, drop {drop:.1%} (needs >= {DECIDE_DROP:.0%})",
        f"  aleatoric {small.aleatoric_sd:.3f} -> {large.aleatoric_sd:.3f} SD, change {alea_change:.1%} (needs < {DECIDE_FLAT:.0%})",
        f"H20 (epistemic higher on unseen patients, aleatoric same): {h20}",
        f"  epistemic unseen / familiar at full data = {ratio:.2f} (needs >= {DECIDE_UNSEEN_RATIO:.2f})",
        f"  aleatoric unseen vs familiar differ by {alea_gap:.1%} (needs < {DECIDE_FLAT:.0%})",
        "  single run: one ensemble per training size, so no error bars",
    ])


def plot(res: pd.DataFrame, path: Path) -> None:
    plotting.apply_style()
    insample = res[res.eval_set == "held-out source"].sort_values("train_fraction")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(insample.train_fraction * 100, insample.epistemic_sd, marker="o", color=plotting.CATEGORICAL[0], label="epistemic SD")
    ax1.plot(insample.train_fraction * 100, insample.aleatoric_sd, marker="s", color=plotting.CATEGORICAL[1], label="aleatoric SD")
    ax1.set_xlabel("Training data used (% of pooled training windows)")
    ax1.set_ylabel("Mean SD (mg/dL), held-out source windows")
    ax1.set_title("H18: does epistemic shrink with data?", loc="left", fontsize=10)
    ax1.legend(frameon=False)
    full = res[res.train_fraction == insample.train_fraction.max()]
    x = np.arange(2)
    w = 0.36
    ax2.bar(x - w / 2, full[full.eval_set == "held-out source"][["epistemic_sd", "aleatoric_sd"]].values[0], w,
            color=plotting.CATEGORICAL[2], label="familiar (held-out source)")
    ax2.bar(x + w / 2, full[full.eval_set == "unseen target"][["epistemic_sd", "aleatoric_sd"]].values[0], w,
            color=plotting.CATEGORICAL[3], label="unseen (target patients)")
    ax2.set_xticks(x, ["epistemic SD", "aleatoric SD"])
    ax2.set_ylabel("Mean SD (mg/dL), full training data")
    ax2.set_title("H20: does epistemic rise on unseen patients?", loc="left", fontsize=10)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
