# Time Perception Modeling from Audio-Visual Cues

This project explores how accumulated visual and auditory change relate to perceived time in short, naturalistic travel POV videos (10–25 seconds). It prepares a dataset of synchronized audio/video clips, then (next steps) trains:

- A visual model (AlexNet) to measure accumulated visual change over time.
- An audio model (OpenL3) to measure accumulated auditory change.
- A late-fusion model to study multisensory cues predicting perceived time.

Human judgments will later be collected via jsPsych and compared against model predictions.

## Project structure

```text
project_root/
  data/
    raw/         # raw copies of original videos
    processed/   # normalized A/V clips (+ optional frames)
  models/        # model weights/checkpoints (future)
  notebooks/     # exploratory analysis (future)
  scripts/
    preprocess_data.py  # read local files + normalize + optional frame extraction
  results/      # figures, metrics, CSVs
  main.py       # env + dependency sanity checks (versions, CUDA)
  requirements.txt
  README.md
```

## Setup and installation

You can use a virtualenv or conda. On Windows (cmd.exe):

### Option A: venv (recommended)

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bat
conda create -n time-perception python=3.10 -y
conda activate time-perception
pip install -r requirements.txt
```

Notes:

- MoviePy uses FFmpeg under the hood. If FFmpeg isn’t found, imageio will auto-download a compatible binary on first use. If you prefer a system FFmpeg, install and add it to PATH.
- OpenL3 may install TensorFlow; PyTorch is also included for the AlexNet baseline. Installing GPU builds is optional but recommended if you have a CUDA-capable GPU.

## Quick start

1. Sanity-check your environment and see versions/CUDA status:

```bat
python main.py
```

2. Put your clips in `data/raw/` (default), or place them in another folder.

3. Run preprocessing to normalize audio (no trimming) and optionally extract frames for the visual model:

```bat
# Default: scans data/raw
python scripts\preprocess_data.py --save-frames --frames-fps 15 --frame-size 224

# Or specify a folder or files
python scripts\preprocess_data.py --input-dir OBS_Clips --save-frames
python scripts\preprocess_data.py videos\clip1.mp4 videos\clip2.mov --target-db -18
```

Key options:

- `--input-dir`: directory to search recursively for video files; if omitted, the script scans `data/raw`.
- `--output-dir`: where to save processed clips (default: `data/processed`).
- `--raw-dir`: where raw copies live (default: `data/raw`).
- `--target-sr`: audio sample rate for processing (default: 48000 Hz).
- `--target-db`: loudness normalization target in dB RMS (default: -20 dB).
- `--save-frames`: extract standardized frames for the vision model.
- `--frames-dir`: root directory for frames (default: `data/processed/frames`).
- `--frames-fps`: frame sampling rate for extraction (default: 15).
- `--frame-size`: output frame size (square, default: 224).

Outputs:

- `data/raw/<filename>.mp4` – raw copy of the original video
- `data/processed/<name>_processed.mp4` – processed video with normalized audio
- `data/processed/<name>_processed.wav` – processed audio waveform
- `data/processed/frames/<name>/` – standardized frames (+ frame_times.npy, meta.json) when `--save-frames` is used

## Next steps (roadmap)

- Implement visual change model using AlexNet features and frame-wise difference accumulation.
- Implement audio change model using OpenL3 embeddings and temporal accumulation.
- Late fusion (e.g., learned linear/MLP) to predict perceived duration.
- Add evaluation code to compare against jsPsych human judgments.

## Troubleshooting

- If MoviePy complains about FFmpeg, ensure `imageio-ffmpeg` auto-download completes or install FFmpeg and add it to PATH.
- If CUDA isn’t detected in `main.py`, verify you installed a CUDA-enabled PyTorch build compatible with your driver.
