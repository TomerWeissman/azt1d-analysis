"""Quick experiment: does a pretrained model's uncertainty behave sensibly when fine-tuned on a
new patient, for two methods (deep ensemble, deep evidential regression)?

Pretrains on 20 of HUPA-UCM's 25 patients, then fine-tunes on 5 held-out target patients at
increasing adaptation budgets (0/1/3/7 days), and checks whether RMSE falls, whether epistemic
uncertainty shrinks with more adaptation data while aleatoric stays roughly flat, and whether
calibration holds up at low budgets.

Run with: python scripts/run_finetune_uncertainty_experiment.py
Writes results/quick_metrics.csv and figures/quick_results.png.

Assumptions not pinned down by the spec, made explicit here since they affect the numbers:
  - Dropout rate: 0.1 (unspecified). A ReLU sits between the 64-unit linear layer and dropout,
    since a linear -> dropout -> linear stack with no nonlinearity between the two linears would
    collapse to one linear map in expectation.
  - Early-stopping patience: 5 epochs (unspecified; this project's other training code uses the
    same default).
  - Ensemble interval: the 3 members' predictions are combined into one Gaussian (mean = mean of
    member means, variance = epistemic + aleatoric) and intervals come from its normal quantiles.
    The spec gives epistemic/aleatoric formulas for the ensemble but only says "intervals from the
    Student-t" for the evidential method, so this combination rule is a choice, not given.
  - "1 seed" is read as one fixed seed for the experiment's random choices (which 20/5 patients,
    model init base), not one literal seed shared by all 3 ensemble members. Giving the 3 members
    identical seeds would make them identical models -- an ensemble in name only, and exactly the
    kind of bug this project already found once in its own genetic-algorithm search (every patient
    started from the same seed there too, and it silently made "personalization" meaningless).
    Members get seeds base_seed, base_seed+1, base_seed+2 instead.
  - Pretraining pools each of the 20 source patients' own chronological first 80%/last 20% split
    (train/val) rather than a single global cut, since the 20 series don't share a calendar.
  - Patients 26-28 have 89-574 days of data (the other 22 have 7-14). To keep "fast" honest, each
    pretrain-source patient's contribution to the pooled pretraining set is capped at its first
    30 days; if one of the long patients is a target instead, its budgets/test block use its real
    length (no cap; a budget only ever needs at most 7 days from the start of its own pool).
"""
from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import hupa  # noqa: E402
from azt1d.glimmer.sequences import make_windows  # noqa: E402
from azt1d.glimmer.train import _iter_batches  # noqa: E402

RNG_SEED = 42
N_TARGETS = 5
BUDGET_DAYS = (0, 1, 3, 7)
FEATURES = ["glucose", "bolus_volume_delivered", "carb_input", "heart_rate"]
TARGET = "glucose"
LOOKBACK = 36  # 3 hours at 5-minute sampling
HORIZON = 6  # 30 minutes ahead
HIDDEN = 128
FC_UNITS = 64
DROPOUT = 0.1
PRETRAIN_LR, PRETRAIN_BATCH, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE = 1e-4, 1024, 50, 5
FINETUNE_LR, FINETUNE_BATCH, FINETUNE_EPOCHS, FINETUNE_PATIENCE = 1e-5, 1024, 50, 5
N_MEMBERS = 3
EVID_LAMBDA = 0.01
COVERAGE_LEVELS = np.round(np.arange(0.1, 0.901, 0.1), 2)
ROWS_PER_DAY = 24 * 60 // 5  # 288
TEST_DAYS = 3
PRETRAIN_SOURCE_CAP_DAYS = 30  # see module docstring
MIN_ADAPT_WINDOWS = 20  # a budget with fewer usable windows than this is skipped


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_patient(subject_id: int) -> pd.DataFrame:
    """Raw 5-minute HUPA-UCM series for one patient, the 4 columns this experiment needs. No NaNs
    were found in any of them across all 25 patients (checked directly), so no imputation logic."""
    path = hupa.DEFAULT_DIR / f"HUPA{subject_id:04d}P.csv"
    raw = pd.read_csv(ROOT / path, sep=";", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df = raw[["time", "glucose", "bolus_volume_delivered", "carb_input", "heart_rate"]].copy()
    df["bolus_volume_delivered"] = df["bolus_volume_delivered"].fillna(0.0)
    df["carb_input"] = df["carb_input"].fillna(0.0)
    return df


def windows_for(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """X: (n, LOOKBACK, 4); y: (n,), the scalar glucose value HORIZON steps after each window."""
    return make_windows(df, FEATURES, TARGET, LOOKBACK, HORIZON)


def chrono_holdout(n: int, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Indices [0, n) split into a chronological (train, val) pair, val = the last val_frac."""
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    return np.arange(0, n - n_val), np.arange(n - n_val, n)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GaussianLSTM(nn.Module):
    """One deep-ensemble member: LSTM -> final hidden state -> 64-unit layer -> dropout -> (mu,
    log_var). Trained with Gaussian NLL (Lakshminarayanan et al. 2017)."""

    def __init__(self, n_features: int, hidden: int = HIDDEN, fc: int = FC_UNITS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, fc)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(fc, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        z = self.dropout(F.relu(self.fc(h_n[-1])))
        mu, log_var = self.out(z).unbind(-1)
        return mu, log_var


class EvidentialLSTM(nn.Module):
    """Same backbone, a Normal-Inverse-Gamma head instead: (gamma, nu, alpha, beta), Amini et al.
    2020. nu, beta > 0 and alpha > 1 via softplus, so aleatoric = beta/(alpha-1) and epistemic =
    beta/(nu*(alpha-1)) are always defined."""

    def __init__(self, n_features: int, hidden: int = HIDDEN, fc: int = FC_UNITS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, fc)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(fc, 4)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        z = self.dropout(F.relu(self.fc(h_n[-1])))
        gamma, nu_raw, alpha_raw, beta_raw = self.out(z).unbind(-1)
        nu = F.softplus(nu_raw) + 1e-6
        alpha = F.softplus(alpha_raw) + 1.0 + 1e-6
        beta = F.softplus(beta_raw) + 1e-6
        return gamma, nu, alpha, beta


def gaussian_nll(mu, log_var, y, eps: float = 1e-6):
    var = torch.exp(log_var).clamp_min(eps)
    return (0.5 * torch.log(2 * math.pi * var) + 0.5 * (y - mu) ** 2 / var).mean()


def nig_nll(y, gamma, nu, alpha, beta, eps: float = 1e-6):
    """Negative log-likelihood of the NIG model's Student-t marginal (Amini et al. 2020, Eq. 8)."""
    omega = (2 * beta * (1 + nu)).clamp_min(eps)
    return (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def evidential_loss(gamma, nu, alpha, beta, y, lam: float = EVID_LAMBDA):
    reg = torch.abs(y - gamma) * (2 * nu + alpha)
    return (nig_nll(y, gamma, nu, alpha, beta) + lam * reg).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(model, loss_fn, X_train, y_train, X_val, y_val, lr, epochs, batch_size, patience, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    best_val, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in _iter_batches(X_train, y_train, batch_size, shuffle=True, rng=rng):
            opt.zero_grad()
            loss = loss_fn(*model(torch.from_numpy(xb).to(device)), torch.from_numpy(yb).to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(*model(X_val_t), y_val_t).item()
        if val_loss < best_val:
            best_val, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction, intervals, metrics
# ---------------------------------------------------------------------------


def ensemble_predict(members, X, device, y_mean, y_std):
    mus, alea = [], []
    with torch.no_grad():
        for m in members:
            mu, log_var = m(torch.from_numpy(X).to(device))
            mus.append(mu.cpu().numpy() * y_std + y_mean)
            alea.append(np.exp(log_var.cpu().numpy()) * y_std**2)
    mus, alea = np.stack(mus), np.stack(alea)
    mean = mus.mean(0)
    epistemic_var = mus.var(0)  # variance of member means
    aleatoric_var = alea.mean(0)  # mean of member variances
    return mean, epistemic_var, aleatoric_var


def ensemble_interval(mean, epistemic_var, aleatoric_var, level):
    z = stats.norm.ppf(0.5 + level / 2)
    sd = np.sqrt(epistemic_var + aleatoric_var)
    return mean - z * sd, mean + z * sd


def evidential_predict(model, X, device, y_mean, y_std):
    with torch.no_grad():
        gamma, nu, alpha, beta = model(torch.from_numpy(X).to(device))
    gamma = gamma.cpu().numpy() * y_std + y_mean
    nu, alpha, beta = (t.cpu().numpy() for t in (nu, alpha, beta))
    beta = beta * y_std**2  # beta scales with the target's variance; nu, alpha are scale-free
    aleatoric_var = beta / (alpha - 1)
    epistemic_var = beta / (nu * (alpha - 1))
    return gamma, epistemic_var, aleatoric_var, nu, alpha, beta


def evidential_interval(gamma, nu, alpha, beta, level):
    df = 2 * alpha
    scale = np.sqrt(beta * (1 + nu) / (nu * alpha))
    t_q = stats.t.ppf(0.5 + level / 2, df)
    return gamma - t_q * scale, gamma + t_q * scale


def mce(y, interval_fn, levels=COVERAGE_LEVELS) -> float:
    gaps = [abs(np.mean((y >= interval_fn(lv)[0]) & (y <= interval_fn(lv)[1])) - lv) for lv in levels]
    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class Scaler:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def transform(self, X, y=None):
        Xs = ((X - self.x_mean) / self.x_std).astype(np.float32)
        if y is None:
            return Xs
        return Xs, ((y - self.y_mean) / self.y_std).astype(np.float32)


def build_pretrain_pool(source_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Scaler]:
    """Each source patient's own first-80%/last-20% chronological split, capped at
    PRETRAIN_SOURCE_CAP_DAYS of data per patient, then pooled. Scaler is fit on the pooled
    training windows only."""
    cap_rows = PRETRAIN_SOURCE_CAP_DAYS * ROWS_PER_DAY
    X_train_parts, y_train_parts, X_val_parts, y_val_parts = [], [], [], []
    for sid in source_ids:
        df = load_patient(sid).iloc[:cap_rows]
        X, y = windows_for(df)
        train_idx, val_idx = chrono_holdout(len(X))
        X_train_parts.append(X[train_idx]); y_train_parts.append(y[train_idx])
        X_val_parts.append(X[val_idx]); y_val_parts.append(y[val_idx])
    X_train = np.concatenate(X_train_parts); y_train = np.concatenate(y_train_parts)
    X_val = np.concatenate(X_val_parts); y_val = np.concatenate(y_val_parts)

    n_features = X_train.shape[-1]
    xs = StandardScaler().fit(X_train.reshape(-1, n_features))
    scaler = Scaler(
        x_mean=xs.mean_.astype(np.float32), x_std=xs.scale_.astype(np.float32),
        y_mean=float(y_train.mean()), y_std=float(y_train.std()),
    )
    X_train_s, y_train_s = scaler.transform(X_train, y_train)
    X_val_s, y_val_s = scaler.transform(X_val, y_val)
    return X_train_s, y_train_s, X_val_s, y_val_s, scaler


def budget_windows(df: pd.DataFrame, budget_days: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Windows built from just the first budget_days of df (the adaptation pool, already excludes
    the test block) -- sliced before windowing rather than after, so a long pool (patients 26-28
    have 89-574 days) is never windowed further than a budget actually needs. None if too little
    data to be worth fine-tuning on."""
    cutoff = budget_days * ROWS_PER_DAY
    sub = df.iloc[: min(len(df), cutoff)]
    try:
        X, y = windows_for(sub)
    except ValueError:  # not enough rows for even one window
        return None
    if len(X) < MIN_ADAPT_WINDOWS:
        return None
    return X, y


def run_target_patient(sid: int, pretrain, device, scaler: Scaler) -> list[dict]:
    df = load_patient(sid)
    n_test_rows = TEST_DAYS * ROWS_PER_DAY
    pool_df = df.iloc[: len(df) - n_test_rows].reset_index(drop=True)
    test_df = df.iloc[len(df) - n_test_rows - LOOKBACK - HORIZON + 1 :].reset_index(drop=True)  # keep lookback context
    X_test, y_test = windows_for(test_df)
    X_test_s = scaler.transform(X_test)

    rows = []
    for budget in BUDGET_DAYS:
        adapt = None if budget == 0 else budget_windows(pool_df, budget)
        if budget != 0 and adapt is None:
            print(f"    patient {sid}, budget {budget}d: skipped (fewer than {MIN_ADAPT_WINDOWS} adaptation windows)")
            continue

        if adapt is not None:
            X_adapt, y_adapt = adapt
            train_idx, val_idx = chrono_holdout(len(X_adapt))
            X_tr_s, y_tr_s = scaler.transform(X_adapt[train_idx], y_adapt[train_idx])
            X_va_s, y_va_s = scaler.transform(X_adapt[val_idx], y_adapt[val_idx])

        # --- deep ensemble ---
        members = []
        for i, base in enumerate(pretrain["ensemble"]):
            m = copy.deepcopy(base)
            if adapt is not None:
                m = train_model(m, lambda mu, lv, y: gaussian_nll(mu, lv, y), X_tr_s, y_tr_s, X_va_s, y_va_s,
                                 FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_BATCH, FINETUNE_PATIENCE, device,
                                 seed=RNG_SEED + 100 * budget + i)
            members.append(m)
        mean, epi_var, alea_var = ensemble_predict(members, X_test_s, device, scaler.y_mean, scaler.y_std)
        rmse = float(np.sqrt(np.mean((y_test - mean) ** 2)))
        rows.append({
            "patient": sid, "method": "Deep Ensemble", "budget_days": budget, "rmse": rmse,
            "epistemic_sd": float(np.mean(np.sqrt(epi_var))), "aleatoric_sd": float(np.mean(np.sqrt(alea_var))),
            "mce": mce(y_test, lambda lv: ensemble_interval(mean, epi_var, alea_var, lv)),
        })

        # --- deep evidential regression ---
        evid = copy.deepcopy(pretrain["evidential"])
        if adapt is not None:
            evid = train_model(evid, evidential_loss, X_tr_s, y_tr_s, X_va_s, y_va_s,
                                FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_BATCH, FINETUNE_PATIENCE, device,
                                seed=RNG_SEED + 100 * budget + 900)
        gamma, epi_var, alea_var, nu, alpha, beta = evidential_predict(evid, X_test_s, device, scaler.y_mean, scaler.y_std)
        rmse = float(np.sqrt(np.mean((y_test - gamma) ** 2)))
        rows.append({
            "patient": sid, "method": "Deep Evidential", "budget_days": budget, "rmse": rmse,
            "epistemic_sd": float(np.mean(np.sqrt(epi_var))), "aleatoric_sd": float(np.mean(np.sqrt(alea_var))),
            "mce": mce(y_test, lambda lv: evidential_interval(gamma, nu, alpha, beta, lv)),
        })
        print(f"    patient {sid}, budget {budget}d: done")
    return rows


def main():
    t0 = time.time()
    # CPU measured faster than MPS for this small LSTM (per-op dispatch overhead dominates on MPS
    # at this scale), checked directly: ~8.3s/epoch vs ~9.5s/epoch on the real pretraining pool.
    device = torch.device("cpu")
    rng = np.random.default_rng(RNG_SEED)
    all_ids = hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR)
    target_ids = sorted(rng.choice(all_ids, N_TARGETS, replace=False).tolist())
    source_ids = [s for s in all_ids if s not in target_ids]
    print(f"Device: {device}. Target patients: {target_ids}. Pretraining on {len(source_ids)} patients.")

    X_train_s, y_train_s, X_val_s, y_val_s, scaler = build_pretrain_pool(source_ids)
    print(f"Pretraining pool: {len(X_train_s):,} train windows, {len(X_val_s):,} val windows.")

    ensemble = []
    for i in range(N_MEMBERS):
        m = GaussianLSTM(len(FEATURES))
        m = train_model(m, lambda mu, lv, y: gaussian_nll(mu, lv, y), X_train_s, y_train_s, X_val_s, y_val_s,
                         PRETRAIN_LR, PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_PATIENCE, device, seed=RNG_SEED + i)
        ensemble.append(m)
        print(f"  pretrained ensemble member {i + 1}/{N_MEMBERS}")

    evid_model = EvidentialLSTM(len(FEATURES))
    evid_model = train_model(evid_model, evidential_loss, X_train_s, y_train_s, X_val_s, y_val_s,
                              PRETRAIN_LR, PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_PATIENCE, device,
                              seed=RNG_SEED + 900)
    print("  pretrained evidential model")
    pretrain = {"ensemble": ensemble, "evidential": evid_model}

    all_rows = []
    for sid in target_ids:
        print(f"  target patient {sid}")
        all_rows.extend(run_target_patient(sid, pretrain, device, scaler))

    results = pd.DataFrame(all_rows)
    (WEEK / "results").mkdir(exist_ok=True)
    (WEEK / "figures").mkdir(exist_ok=True)
    results.to_csv(WEEK / "results" / "quick_metrics.csv", index=False)

    summary = results.groupby(["method", "budget_days"]).agg(
        rmse_mean=("rmse", "mean"), rmse_sem=("rmse", "sem"),
        epistemic_mean=("epistemic_sd", "mean"), epistemic_sem=("epistemic_sd", "sem"),
        aleatoric_mean=("aleatoric_sd", "mean"), aleatoric_sem=("aleatoric_sd", "sem"),
        mce_mean=("mce", "mean"), mce_sem=("mce", "sem"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"Deep Ensemble": "#2a78d6", "Deep Evidential": "#eb6834"}
    for method, color in colors.items():
        s = summary[summary.method == method].sort_values("budget_days")
        axes[0].errorbar(s.budget_days, s.rmse_mean, yerr=s.rmse_sem.fillna(0), color=color, marker="o", label=method)
        axes[1].errorbar(s.budget_days, s.epistemic_mean, yerr=s.epistemic_sem.fillna(0), color=color, marker="o",
                          linestyle="-", label=f"{method}, epistemic")
        axes[1].errorbar(s.budget_days, s.aleatoric_mean, yerr=s.aleatoric_sem.fillna(0), color=color, marker="s",
                          linestyle="--", label=f"{method}, aleatoric")
        axes[2].errorbar(s.budget_days, s.mce_mean, yerr=s.mce_sem.fillna(0), color=color, marker="o", label=method)
    axes[0].set_title("RMSE vs. adaptation budget"); axes[0].set_ylabel("RMSE (mg/dL)")
    axes[1].set_title("Uncertainty vs. adaptation budget"); axes[1].set_ylabel("SD (mg/dL)")
    axes[2].set_title("Calibration vs. adaptation budget"); axes[2].set_ylabel("MCE")
    for ax in axes:
        ax.set_xlabel("Days of adaptation data")
        ax.set_xticks(list(BUDGET_DAYS))
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "quick_results.png", dpi=150)

    elapsed = time.time() - t0
    print(f"\nWrote results/quick_metrics.csv ({len(results)} rows) and figures/quick_results.png")
    print(f"Runtime: {elapsed / 60:.1f} minutes")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
