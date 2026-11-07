# Data

Media and participant-level data are not published. The clips are third-party footage, and the individual responses are covered by the study's ethics terms. To reproduce the pipeline with your own clips, use this layout:

```
data/
  raw/                         original video clips (<clip_id>.mp4)
  processed/                   output of scripts/preprocess_data.py
    <clip_id>_processed.mp4
    <clip_id>_processed.wav
    frames/<clip_id>/          frame_000001.jpg …, frame_times.npy, meta.json
  labels/                      output of scripts/generate_labels.py
    video_perceived.csv        clip_id, perceived_s  (video-only condition means)
    audio_perceived.csv        clip_id, perceived_s  (audio-only condition means)
    av_perceived.csv           clip_id, perceived_s  (audio-visual condition means)
```

Per-clip aggregated human estimates for all three conditions are in [`results/fusion/fusion_predictions.csv`](../results/fusion/fusion_predictions.csv).
