"""
Visual feature extraction with AlexNet (fc7) on pre-extracted frames.

Inputs (produced by scripts/preprocess_data.py when --save-frames is used):
- data/processed/frames/<clip_id>/
    - frame_000001.jpg, frame_000002.jpg, ...
    - frame_times.npy  (float64 array of timestamps in seconds)

Outputs:
- results/visual/summary.csv                     # one row per clip (means/stds)
- results/visual/frames/<clip_id>_frames.csv     # per-frame change metrics
- Optional: results/visual/embeddings/<clip_id>.npy  # if --save-embeddings

Usage (Windows cmd):
  python -m scripts.visual_extract_alexnet ^
    --frames-root data/processed/frames ^
    --out-dir results/visual ^
    --batch-size 64 ^
    --device auto

Note: Install CUDA-enabled PyTorch/torchvision separately following README instructions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights


def build_model(device: torch.device) -> nn.Module:
    """AlexNet feature extractor up to fc7 (4096-D)."""
    weights = AlexNet_Weights.IMAGENET1K_V1
    net = alexnet(weights=weights)
    # Keep features through the second-to-last layer (fc7)
    feature_extractor = nn.Sequential(
        net.features,
        net.avgpool,
        nn.Flatten(),
        net.classifier[0],  # fc6
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.0),  # disable dropout at inference
        net.classifier[3],  # fc7
        nn.ReLU(inplace=True),
        # Stop before the final classifier layer
    )
    feature_extractor.eval().to(device)
    return feature_extractor


def build_transform() -> T.Compose:
    """ImageNet preprocessing for 224x224 RGB frames."""
    weights = AlexNet_Weights.IMAGENET1K_V1
    mean = weights.meta["mean"]
    std = weights.meta["std"]
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def compute_change(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-frame change metrics from 2D embeddings [N, D].

    Returns (dL2, dCos), each shaped [N], with a 0 prepended so they align to frames.
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

    # Prepend a 0 to align with the first frame
    dL2 = np.concatenate([[0.0], dL2])
    dCos = np.concatenate([[0.0], dCos])
    return dL2, dCos


def load_frames_and_times(clip_dir: Path) -> Tuple[List[Path], np.ndarray]:
    times = np.load(clip_dir / "frame_times.npy")
    frame_paths = sorted(clip_dir.glob("frame_*.jpg"))
    if len(frame_paths) != len(times):
        # Best-effort alignment: trim to the min length
        n = min(len(frame_paths), len(times))
        frame_paths = frame_paths[:n]
        times = times[:n]
    return frame_paths, times


def extract_clip_features(
    clip_dir: Path,
    model: nn.Module,
    transform: T.Compose,
    device: torch.device,
    batch_size: int = 64,
    save_embeddings: bool = False,
) -> Dict:
    frame_paths, times = load_frames_and_times(clip_dir)
    if len(frame_paths) == 0:
        return {
            "clip_id": clip_dir.name,
            "times": np.array([]),
            "dL2": np.array([]),
            "dCos": np.array([]),
            "summary": {
                "clip_id": clip_dir.name,
                "n_frames": 0,
                "mean_dL2": np.nan,
                "std_dL2": np.nan,
                "max_dL2": np.nan,
                "mean_dCos": np.nan,
                "std_dCos": np.nan,
                "max_dCos": np.nan,
            },
            "embeddings": None,
        }

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frame_paths), batch_size), desc=f"{clip_dir.name}"):
            batch_paths = frame_paths[i : i + batch_size]
            imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(imgs, dim=0).to(device)
            y = model(x)
            feats.append(y.detach().cpu().numpy())
    embeds = np.concatenate(feats, axis=0)

    dL2, dCos = compute_change(embeds)

    summary = {
        "clip_id": clip_dir.name,
        "n_frames": int(len(times)),
        "mean_dL2": float(np.nanmean(dL2)) if len(dL2) else np.nan,
        "std_dL2": float(np.nanstd(dL2)) if len(dL2) else np.nan,
        "max_dL2": float(np.nanmax(dL2)) if len(dL2) else np.nan,
        "mean_dCos": float(np.nanmean(dCos)) if len(dCos) else np.nan,
        "std_dCos": float(np.nanstd(dCos)) if len(dCos) else np.nan,
        "max_dCos": float(np.nanmax(dCos)) if len(dCos) else np.nan,
    }

    return {
        "clip_id": clip_dir.name,
        "times": times,
        "dL2": dL2,
        "dCos": dCos,
        "summary": summary,
        "embeddings": embeds if save_embeddings else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Extract AlexNet features and change metrics from frames")
    ap.add_argument("--frames-root", type=Path, default=Path("data/processed/frames"), help="Root containing per-clip frame folders")
    ap.add_argument("--out-dir", type=Path, default=Path("results/visual"), help="Output directory for CSVs and optional embeddings")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--save-embeddings", action="store_true", help="Also save per-frame embeddings as .npy per clip")
    args = ap.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Discover clips (directories with frame_*.jpg)
    clips = [p for p in sorted(args.frames_root.iterdir()) if p.is_dir() and any(p.glob("frame_*.jpg"))]
    print(f"Found {len(clips)} clip(s) under {args.frames_root}")

    # Prepare outputs
    (args.out_dir / "frames").mkdir(parents=True, exist_ok=True)
    if args.save_embeddings:
        (args.out_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    # Build model & transform
    transform = build_transform()
    model = build_model(device)

    summaries = []
    for clip_dir in clips:
        print(f"Processing {clip_dir.name} ...")
        res = extract_clip_features(
            clip_dir,
            model,
            transform,
            device,
            batch_size=args.batch_size,
            save_embeddings=args.save_embeddings,
        )
        summaries.append(res["summary"])

        # Save per-frame metrics
        df = pd.DataFrame({
            "time_s": res["times"],
            "dL2": res["dL2"],
            "dCos": res["dCos"],
        })
        df.to_csv(args.out_dir / "frames" / f"{clip_dir.name}_frames.csv", index=False)

        # Optional embeddings
        if args.save_embeddings and res["embeddings"] is not None:
            np.save(args.out_dir / "embeddings" / f"{clip_dir.name}.npy", res["embeddings"])

    # Save summary
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_dir / "summary.csv", index=False)
    print(f"Wrote summary to {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
