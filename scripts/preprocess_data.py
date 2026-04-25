"""
Preprocess local video files (e.g., OBS recordings) for time-perception modeling.

Pipeline:
- Read local video files from a directory and/or explicit list (or data/raw by default)
- Extract audio and normalize loudness to a target RMS level (no trimming)
- Recombine synchronized A/V and save to data/processed

Notes:
- The full clip duration is preserved; no silence trimming is performed.
- We target a fixed sample rate and RMS dB level to standardize loudness.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Media utils
from moviepy.editor import VideoFileClip, AudioFileClip

# Audio processing
import librosa
import cv2


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class PreprocessConfig:
    raw_dir: Path
    output_dir: Path
    target_sr: int = 48_000
    target_db: float = -20.0  # target RMS loudness (dB)
    # Visual frame extraction options
    save_frames: bool = False
    frames_dir: Path = Path("data/processed/frames")
    frames_fps: int = 30
    frame_size: int = 224


# -----------------------------
# Utility functions
# -----------------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def collect_input_files(input_dir: Optional[Path], files: List[str]) -> List[Path]:
    """Gather input video files from a directory and/or explicit list.

    Supported extensions: .mp4, .mov, .mkv, .avi, .webm
    """
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    paths: List[Path] = []
    if input_dir and input_dir.exists():
        for p in sorted(input_dir.rglob("*")):
            if p.suffix.lower() in exts:
                paths.append(p)
    for f in files:
        p = Path(f)
        if p.exists() and p.suffix.lower() in exts:
            paths.append(p)
    # Deduplicate preserving order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def rms_db(y: np.ndarray) -> float:
    eps = 1e-10
    rms = np.sqrt(np.mean(np.square(y), dtype=np.float64) + eps)
    return 20.0 * math.log10(rms + eps)


def normalize_to_db(y: np.ndarray, target_db: float) -> np.ndarray:
    cur_db = rms_db(y)
    gain_db = target_db - cur_db
    gain = 10 ** (gain_db / 20.0)
    y_out = y * gain
    # avoid clipping
    max_abs = np.max(np.abs(y_out))
    if max_abs > 1.0:
        y_out = y_out / max_abs
    return y_out.astype(np.float32)


    


# -----------------------------
# Frame extraction helpers
# -----------------------------

def center_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square then resize to (size, size)."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def extract_and_save_frames(video_path: Path, out_dir: Path, target_fps: int, frame_size: int) -> dict:
    """Sample frames at exact timestamps (1/target_fps), save JPGs, frame_times.npy, and meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0 for frame extraction")

    times: List[float] = []
    saved = 0
    with VideoFileClip(str(video_path)) as clip:
        duration = float(clip.duration)
        step = 1.0 / float(target_fps)
        t = 0.0
        while t < duration - 1e-6:
            frame_rgb = clip.get_frame(t)  # RGB ndarray
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            proc = center_crop_resize(frame_bgr, frame_size)
            cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), proc, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            times.append(t)
            saved += 1
            t += step

    # Save timestamps for alignment with audio features
    np.save(str(out_dir / "frame_times.npy"), np.array(times, dtype=np.float32))

    meta = {
        "video_path": str(video_path),
        "target_fps": int(target_fps),
        "saved_frames": int(saved),
        "duration_sec": float(duration),
        "frame_size": int(frame_size),
        "timeline": {"start_sec": 0.0, "step_sec": float(step)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# -----------------------------
# Core processing
# -----------------------------

def copy_to_raw(src: Path, raw_dir: Path) -> Path:
    """Copy input file into raw_dir for a consistent workflow."""
    dst = raw_dir / src.name
    if dst.resolve() == src.resolve():
        return dst
    data = src.read_bytes()
    dst.write_bytes(data)
    return dst


def process_clip(in_path: Path, cfg: PreprocessConfig) -> Tuple[Path, Path]:
    """Create processed video and wav with normalized audio (no trimming).

    Returns (processed_video_path, processed_wav_path).
    """
    video_id = in_path.stem
    out_video = cfg.output_dir / f"{video_id}_processed.mp4"
    out_audio = cfg.output_dir / f"{video_id}_processed.wav"

    # Load video and extract audio as ndarray
    with VideoFileClip(str(in_path)) as vclip:
        duration = float(vclip.duration)
        if vclip.audio is None:
            raise RuntimeError(f"No audio track found in: {in_path}")
        # Write temp wav to read with librosa for consistent resampling
        tmp_wav = cfg.output_dir / f".{video_id}_tmp_input.wav"
        vclip.audio.write_audiofile(
            str(tmp_wav),
            fps=cfg.target_sr,
            verbose=False,
            logger=None,
        )

    # Load audio and normalize loudness (no trimming)
    y, sr = librosa.load(str(tmp_wav), sr=cfg.target_sr, mono=True)
    y = normalize_to_db(y, cfg.target_db)

    # Save normalized (and possibly trimmed) wav
    sf.write(str(out_audio), y, sr)

    # Reconstruct final video with the processed audio, preserving full duration
    with VideoFileClip(str(in_path)) as vclip, AudioFileClip(str(out_audio)) as aclip:
        aclip = aclip.subclip(0, vclip.duration)
        vfinal = vclip.set_audio(aclip)
        vfinal.write_videofile(
            str(out_video),
            codec="libx264",
            audio_codec="aac",
            fps=vclip.fps,
            verbose=False,
            logger=None,
        )

    # Cleanup temp
    try:
        tmp_wav.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    return out_video, out_audio


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess local videos (OBS clips)")
    p.add_argument("files", nargs="*", help="Video file paths (optional if --input-dir provided)")
    p.add_argument("--input-dir", type=str, default=None, help="Directory to search recursively for video files")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Directory to store raw inputs")
    p.add_argument("--output-dir", type=str, default="data/processed", help="Directory for processed outputs")
    p.add_argument("--target-sr", type=int, default=48_000, help="Target sample rate for audio")
    p.add_argument("--target-db", type=float, default=-20.0, help="Target RMS loudness in dB")
    # Frame extraction options
    p.add_argument("--save-frames", action="store_true", help="Extract standardized frames for vision model")
    p.add_argument("--frames-dir", type=str, default="data/processed/frames", help="Directory root for frames")
    p.add_argument("--frames-fps", type=int, default=15, help="Target FPS for frame sampling")
    p.add_argument("--frame-size", type=int, default=224, help="Output frame size (square)")
    return p.parse_args(argv)


essential_usage = """
Examples:
    # Default: scan data/raw if no inputs provided
    python scripts\preprocess_data.py

    # Specify a folder or files
    python scripts\preprocess_data.py --input-dir OBS_Clips --save-frames --frames-fps 15
    python scripts\preprocess_data.py videos\clip1.mp4 videos\clip2.mov --target-db -18
"""


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    ensure_dirs(raw_dir, out_dir)

    cfg = PreprocessConfig(
        raw_dir=raw_dir,
        output_dir=out_dir,
        target_sr=args.target_sr,
        target_db=args.target_db,
        save_frames=bool(args.save_frames),
        frames_dir=Path(args.frames_dir),
        frames_fps=int(args.frames_fps),
        frame_size=int(args.frame_size),
    )

    inputs = collect_input_files(Path(args.input_dir) if args.input_dir else None, list(args.files))
    # If none provided, fall back to scanning data/raw
    if not inputs:
        inputs = collect_input_files(raw_dir, [])
    if not inputs:
        print("No input videos found. Place clips in data/raw or provide --input-dir / file paths.")
        print(essential_usage)
        return

    for src in tqdm(inputs, desc="Processing videos"):
        try:
            raw_path = copy_to_raw(src, cfg.raw_dir)
            processed_video, _ = process_clip(raw_path, cfg)
            if cfg.save_frames:
                clip_id = raw_path.stem
                frames_out_dir = cfg.frames_dir / clip_id
                ensure_dirs(frames_out_dir)
                extract_and_save_frames(processed_video, frames_out_dir, cfg.frames_fps, cfg.frame_size)
        except Exception as e:
            print(f"Error processing {src}: {e}")


if __name__ == "__main__":
    main()
