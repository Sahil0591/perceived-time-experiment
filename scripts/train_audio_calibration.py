"""
Fit a linear calibration mapping from audio change (sum_dL2 or rate_dL2)
to human-perceived duration (seconds), following Roseboom-style accumulation.

Inputs:
- results/audio/summary.csv (from scripts/audio_extract_features.py)
- labels CSV with at least: clip_id, perceived_s (human judgments in seconds)

Outputs:
- results/audio/calibration.json with {alpha, beta, feature, r2}
- results/audio/calibration_table.csv (merged training table)

Usage (Windows cmd):
  python -m scripts.train_audio_calibration ^
    --summary d:\Sahil\Capstone_Project\results\audio\summary.csv ^
    --labels d:\Sahil\Capstone_Project\data\labels\audio_perceived.csv ^
    --feature sum_dL2 ^
    --label-col perceived_s
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def main():
    ap = argparse.ArgumentParser(description="Train audio subjective-time calibration")
    ap.add_argument("--summary", type=Path, required=True, help="Path to results/audio/summary.csv")
    ap.add_argument("--labels", type=Path, required=True, help="Path to labels CSV with clip_id and perceived duration")
    ap.add_argument("--label-col", type=str, default="perceived_s", help="Column name in labels for perceived duration (seconds)")
    ap.add_argument("--feature", type=str, default="sum_dL2", choices=["sum_dL2", "rate_dL2"], help="Feature to use for calibration")
    ap.add_argument("--out-json", type=Path, default=Path("results/audio/calibration.json"), help="Output JSON with alpha/beta")
    ap.add_argument("--out-table", type=Path, default=Path("results/audio/calibration_table.csv"), help="Merged training table output")
    args = ap.parse_args()

    # Load data
    summary_df = pd.read_csv(args.summary)
    labels_df = pd.read_csv(args.labels)

    # Check required columns
    for col in ["clip_id", args.feature]:
        if col not in summary_df.columns:
            raise ValueError(f"summary.csv missing required column: {col}")
    if "clip_id" not in labels_df.columns or args.label_col not in labels_df.columns:
        raise ValueError("labels CSV must have columns: clip_id and the specified label column")

    # Merge
    df = summary_df.merge(labels_df[["clip_id", args.label_col]], on="clip_id", how="inner")
    df = df.dropna(subset=[args.feature, args.label_col])
    if len(df) == 0:
        raise ValueError("No overlapping clips between summary and labels after merge")

    X = df[[args.feature]].values.astype(np.float64)
    y = df[args.label_col].values.astype(np.float64)

    # Fit linear mapping: y = alpha * X + beta
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = float(r2_score(y, y_pred))

    alpha = float(reg.coef_[0])
    beta = float(reg.intercept_)

    # Save calibration
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"alpha": alpha, "beta": beta, "feature": args.feature, "r2": r2}, f, indent=2)

    # Save merged table for inspection
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["pred_subjective_s"] = y_pred
    df_out.to_csv(args.out_table, index=False)

    print(f"Saved calibration to {args.out_json} (feature={args.feature}, alpha={alpha:.6g}, beta={beta:.6g}, R^2={r2:.3f})")


if __name__ == "__main__":
    main()
