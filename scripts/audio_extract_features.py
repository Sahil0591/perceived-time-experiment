"""
Audio feature extraction using mel-spectrogram and temporal segmentation.

Inputs (produced by scripts/preprocess_data.py):
- data/processed/**/<clip_id>_processed.wav (or other audio files)

Outputs:
- results/audio/summary.csv                     # one row per clip (means/stds)
- results/audio/segments/<clip_id>_segments.csv # per-segment change metrics
- Optional: results/audio/embeddings/<clip_id>.npy  # if --save-embeddings

Usage (Windows cmd):
    python -m scripts.audio_extract_features ^
        --audio-root data/processed ^
        --out-dir results/audio ^
        --segment-duration 0.5 ^
        --hop-duration 0.25 ^
        --sr 0 ^
        --alpha 1.0 ^
        --beta 0.0

Notes:
- Set --sr 0 to keep the native sample rate (no resampling). This avoids double-processing when audio was already standardized by preprocess_data.py.
 - Subjective time prediction follows Roseboom-style accumulation: pred_s = alpha * sum(dL2) + beta.

Dependencies: librosa, numpy, pandas, tqdm
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from sklearn.linear_model import LinearRegression

try:
    import librosa
    import librosa.feature
except ImportError:
    raise ImportError("Please install librosa: pip install librosa")


def extract_mel_features(
    audio_path: Path,
    sr: int = 0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extract mel-spectrogram features from audio file.

    Args:
        sr: Target sample rate. Use 0 to keep native sample rate (no resampling).

    Returns:
        features: [n_frames, n_mels] array of mel-spectrogram features
        times: [n_frames] array of time points in seconds
        sr_used: The actual sampling rate used
    """
    target_sr = None if sr == 0 else sr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr_used = librosa.load(audio_path, sr=target_sr, mono=True)

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr_used,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Transpose to [n_frames, n_mels]
    features = mel_spec_db.T

    # Compute time points for each frame
    times = librosa.frames_to_time(
        np.arange(features.shape[0]),
        sr=sr_used,
        hop_length=hop_length,
    )

    return features, times, sr_used


def segment_features(
    features: np.ndarray,
    times: np.ndarray,
    segment_duration: float = 0.5,
    hop_duration: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate features into fixed-duration segments.
    
    Returns:
        embeddings: [n_segments, n_mels] mean features per segment
        segment_times: [n_segments] center time of each segment
    """
    if len(features) == 0 or len(times) == 0:
        return np.array([]), np.array([])
    
    total_duration = times[-1]
    segment_centers = []
    segment_feats = []
    
    current_time = segment_duration / 2
    while current_time <= total_duration:
        # Find frames within this segment window
        start_time = current_time - segment_duration / 2
        end_time = current_time + segment_duration / 2
        
        mask = (times >= start_time) & (times < end_time)
        if mask.sum() > 0:
            # Average features in this segment
            seg_feat = features[mask].mean(axis=0)
            segment_feats.append(seg_feat)
            segment_centers.append(current_time)
        
        current_time += hop_duration
    
    if len(segment_feats) == 0:
        return np.array([]), np.array([])
    
    embeddings = np.stack(segment_feats, axis=0)
    segment_times = np.array(segment_centers)
    
    return embeddings, segment_times


def compute_change(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-segment change metrics from 2D embeddings [N, D].
    
    Returns (dL2, dCos), each shaped [N], with a 0 prepended so they align to segments.
    """
    if embeddings.shape[0] == 0:
        return np.array([]), np.array([])
    if embeddings.shape[0] == 1:
        return np.array([0.0]), np.array([0.0])
    
    diffs = embeddings[1:] - embeddings[:-1]
    dL2 = np.linalg.norm(diffs, axis=1)
    
    a = embeddings[1:]
    b = embeddings[:-1]
    an = np.linalg.norm(a, axis=1) + 1e-8
    bn = np.linalg.norm(b, axis=1) + 1e-8
    cos_sim = (a * b).sum(axis=1) / (an * bn)
    dCos = 1.0 - cos_sim
    
    # Prepend a 0 to align with the first segment
    dL2 = np.concatenate([[0.0], dL2])
    dCos = np.concatenate([[0.0], dCos])
    return dL2, dCos


def extract_clip_features(
    audio_path: Path,
    segment_duration: float = 0.5,
    hop_duration: float = 0.25,
    sr: int = 0,
    n_mels: int = 128,
    save_embeddings: bool = False,
) -> Dict:
    """Extract audio features and compute change metrics for a single clip."""
    clip_id = audio_path.stem
    
    # Extract mel-spectrogram features
    features, frame_times, sr_used = extract_mel_features(
        audio_path,
        sr=sr,
        n_mels=n_mels,
    )
    
    if len(features) == 0:
        return {
            "clip_id": clip_id,
            "times": np.array([]),
            "dL2": np.array([]),
            "dCos": np.array([]),
            "summary": {
                "clip_id": clip_id,
                "n_segments": 0,
                "duration_s": 0.0,
                "mean_dL2": np.nan,
                "std_dL2": np.nan,
                "max_dL2": np.nan,
                "mean_dCos": np.nan,
                "std_dCos": np.nan,
                "max_dCos": np.nan,
                "sr_used": None,
            },
            "embeddings": None,
        }
    
    # Aggregate into segments
    embeddings, segment_times = segment_features(
        features,
        frame_times,
        segment_duration=segment_duration,
        hop_duration=hop_duration,
    )
    
    if len(embeddings) == 0:
        return {
            "clip_id": clip_id,
            "times": np.array([]),
            "dL2": np.array([]),
            "dCos": np.array([]),
            "summary": {
                "clip_id": clip_id,
                "n_segments": 0,
                "duration_s": float(frame_times[-1]) if len(frame_times) > 0 else 0.0,
                "mean_dL2": np.nan,
                "std_dL2": np.nan,
                "max_dL2": np.nan,
                "mean_dCos": np.nan,
                "std_dCos": np.nan,
                "max_dCos": np.nan,
                "sr_used": int(sr_used) if sr_used else None,
            },
            "embeddings": None,
        }
    
    # Compute change metrics
    dL2, dCos = compute_change(embeddings)
    
    # Aggregate change for subjective time proxy
    sum_dL2 = float(np.nansum(dL2)) if len(dL2) > 0 else 0.0
    sum_dCos = float(np.nansum(dCos)) if len(dCos) > 0 else 0.0
    duration_s = float(frame_times[-1]) if len(frame_times) > 0 else 0.0
    rate_dL2 = (sum_dL2 / duration_s) if duration_s > 0 else np.nan
    rate_dCos = (sum_dCos / duration_s) if duration_s > 0 else np.nan

    summary = {
        "clip_id": clip_id,
        "n_segments": int(len(segment_times)),
        "duration_s": duration_s,
        "mean_dL2": float(np.nanmean(dL2)) if len(dL2) > 0 else np.nan,
        "std_dL2": float(np.nanstd(dL2)) if len(dL2) > 0 else np.nan,
        "max_dL2": float(np.nanmax(dL2)) if len(dL2) > 0 else np.nan,
        "mean_dCos": float(np.nanmean(dCos)) if len(dCos) > 0 else np.nan,
        "std_dCos": float(np.nanstd(dCos)) if len(dCos) > 0 else np.nan,
        "max_dCos": float(np.nanmax(dCos)) if len(dCos) > 0 else np.nan,
        "sr_used": int(sr_used),
        "sum_dL2": sum_dL2,
        "sum_dCos": sum_dCos,
        "rate_dL2": float(rate_dL2) if not np.isnan(rate_dL2) else np.nan,
        "rate_dCos": float(rate_dCos) if not np.isnan(rate_dCos) else np.nan,
    }
    
    return {
        "clip_id": clip_id,
        "times": segment_times,
        "dL2": dL2,
        "dCos": dCos,
        "summary": summary,
        "embeddings": embeddings if save_embeddings else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Extract audio features and change metrics")
    ap.add_argument("--audio-root", type=Path, default=Path("data/processed"), help="Root directory containing processed audio files (searches recursively)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/audio"), help="Output directory for CSVs and optional embeddings")
    ap.add_argument("--segment-duration", type=float, default=0.5, help="Duration of each audio segment in seconds")
    ap.add_argument("--hop-duration", type=float, default=0.25, help="Hop between segments in seconds")
    ap.add_argument("--sr", type=int, default=0, help="Target sample rate (0 = keep native; avoids double-processing)")
    ap.add_argument("--n-mels", type=int, default=128, help="Number of mel bands")
    ap.add_argument("--save-embeddings", action="store_true", help="Also save per-segment embeddings as .npy per clip")
    ap.add_argument("--alpha", type=float, default=1.0, help="Scale factor for subjective time: pred_s = alpha * sum(dL2) + beta")
    ap.add_argument("--beta", type=float, default=0.0, help="Intercept for subjective time: pred_s = alpha * sum(dL2) + beta")
    ap.add_argument("--feature", type=str, default="sum_dL2", choices=["sum_dL2", "rate_dL2", "sum_dCos", "rate_dCos"], help="Feature to use for prediction/calibration")
    ap.add_argument("--auto-calibrate", action="store_true", help="Fit alpha/beta inside this run")
    ap.add_argument("--labels", type=Path, default=None, help="CSV with clip_id and label column for calibration")
    ap.add_argument("--label-col", type=str, default="perceived_s", help="Label column name in --labels")
    ap.add_argument("--calibrate-to-duration", action="store_true", help="Use actual duration_s as labels if no human labels")
    ap.add_argument("--self-calibrate", action="store_true", help="Derive alpha from dataset median rate: alpha = 1/median(rate_feature); beta=0")
    args = ap.parse_args()
    
    # Discover audio files
    audio_files = sorted(args.audio_root.rglob("*.wav"))
    if len(audio_files) == 0:
        audio_files = sorted(args.audio_root.rglob("*.mp3"))
    
    print(f"Found {len(audio_files)} audio file(s) under {args.audio_root}")
    
    if len(audio_files) == 0:
        print("No audio files found. Exiting.")
        return
    
    # Prepare outputs
    (args.out_dir / "segments").mkdir(parents=True, exist_ok=True)
    if args.save_embeddings:
        (args.out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    
    summaries = []
    for audio_path in tqdm(audio_files, desc="Processing clips"):
        res = extract_clip_features(
            audio_path,
            segment_duration=args.segment_duration,
            hop_duration=args.hop_duration,
            sr=args.sr,
            n_mels=args.n_mels,
            save_embeddings=args.save_embeddings,
        )
        # Compute subjective time prediction using Roseboom-style accumulation
        pred_subjective_s = args.alpha * res["summary"]["sum_dL2"] + args.beta
        res_summary = dict(res["summary"])  # copy
        res_summary["pred_subjective_s"] = float(pred_subjective_s)
        summaries.append(res_summary)
        
        # Save per-segment metrics
        if len(res["times"]) > 0:
            df = pd.DataFrame({
                "time_s": res["times"],
                "dL2": res["dL2"],
                "dCos": res["dCos"],
            })
            df.to_csv(args.out_dir / "segments" / f"{res['clip_id']}_segments.csv", index=False)
        
        # Optional embeddings
        if args.save_embeddings and res["embeddings"] is not None:
            np.save(args.out_dir / "embeddings" / f"{res['clip_id']}.npy", res["embeddings"])
    
    # Save summary with optional in-script calibration
    summary_df = pd.DataFrame(summaries)

    # Self calibration: seconds per unit change from median rate
    if args.self_calibrate:
        # Determine the rate column corresponding to chosen feature
        if args.feature.startswith("sum_"):
            rate_col = "rate_" + args.feature[len("sum_"):]
            if rate_col not in summary_df.columns:
                # compute rate from sum/duration if missing
                summary_df[rate_col] = summary_df[args.feature] / np.clip(summary_df["duration_s"], 1e-6, None)
        else:
            rate_col = args.feature

        rates = summary_df[rate_col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(rates) > 0:
            med_rate = float(np.median(rates))
            if med_rate > 0 and not np.isnan(med_rate):
                alpha_self = 1.0 / med_rate
                beta_self = 0.0
                summary_df["pred_subjective_s"] = alpha_self * summary_df[args.feature] + beta_self
                with open(args.out_dir / "calibration.json", "w", encoding="utf-8") as f:
                    json.dump({"method": "self", "feature": args.feature, "alpha": alpha_self, "beta": beta_self, "rate_col": rate_col}, f, indent=2)
                print(f"Self-calibrated using median {rate_col}={med_rate:.6g}; alpha={alpha_self:.6g}, beta=0")
            else:
                print("Self-calibration skipped: median rate is zero/NaN.")
        else:
            print("Self-calibration skipped: no valid rate values.")

    if args.auto_calibrate:
        X = summary_df[[args.feature]].values
        y = None
        if args.calibrate_to_duration:
            y = summary_df[["duration_s"]].values.ravel()
        elif args.labels is not None and args.labels.exists():
            labels_df = pd.read_csv(args.labels)
            if "clip_id" in labels_df.columns and args.label_col in labels_df.columns:
                merged = summary_df.merge(labels_df[["clip_id", args.label_col]], on="clip_id", how="inner")
                if len(merged) > 0:
                    X = merged[[args.feature]].values
                    y = merged[[args.label_col]].values.ravel()
                else:
                    print("No overlapping clips between summary and labels; skipping calibration.")
            else:
                print("Labels CSV missing required columns: clip_id and", args.label_col)
        else:
            if args.labels:
                print(f"Labels file not found: {args.labels}")
            else:
                print("No labels provided; use --calibrate-to-duration or provide --labels.")

        if y is not None and len(X) == len(y) and len(y) > 0:
            reg = LinearRegression()
            reg.fit(X, y)
            alpha = float(reg.coef_[0])
            beta = float(reg.intercept_)
            summary_df["pred_subjective_s"] = alpha * summary_df[args.feature] + beta
            # Save calibration
            (args.out_dir).mkdir(parents=True, exist_ok=True)
            with open(args.out_dir / "calibration.json", "w", encoding="utf-8") as f:
                json.dump({"feature": args.feature, "alpha": alpha, "beta": beta}, f, indent=2)
            print(f"Calibrated alpha={alpha:.6g}, beta={beta:.6g}; saved calibration.json")
        else:
            print("Calibration skipped.")

    summary_df.to_csv(args.out_dir / "summary.csv", index=False)
    print(f"Wrote summary to {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
