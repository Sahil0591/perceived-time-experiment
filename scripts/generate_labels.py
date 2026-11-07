import csv
import statistics
from pathlib import Path
from collections import defaultdict

# Paths and output location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMBINED_DIR = PROJECT_ROOT / "results" / "Combined csv"
OUT_DIR      = PROJECT_ROOT / "data" / "labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Read every CSV in the combined folder
all_rows = []
csv_files = sorted(COMBINED_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    with open(f, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  {f.name} — {len(rows)} rows")
    all_rows.extend(rows)

# Keep only rows that have the fields we need and a numeric response
clean = []
for r in all_rows:
    if r.get('clip_id') and r.get('condition') and r.get('response') and r.get('response') != 'null':
        try:
            clean.append({
                'participant_id':  r['participant_id'],
                'clip_id':         r['clip_id'],
                'condition':       r['condition'],
                'response':        float(r['response']),
                'actual_duration': float(r['actual_duration_s']) if r.get('actual_duration_s') else None,
                'rt':              float(r['rt']) if r.get('rt') and r['rt'] not in ('', 'null') else None,
            })
        except:
            pass

print(f"\nTotal clean responses: {len(clean)}")
print(f"Participants: {len(set(r['participant_id'] for r in clean))}")

for cond in ['A', 'V', 'AV']:
    n     = len([r for r in clean if r['condition'] == cond])
    clips = len(set(r['clip_id'] for r in clean if r['condition'] == cond))
    print(f"  Condition {cond}: {n} responses across {clips} clips")

# Check whether any participant is imbalanced across conditions
per_participant = defaultdict(lambda: defaultdict(list))
for r in clean:
    per_participant[r['participant_id']][r['condition']].append(r['clip_id'])

print("\nPer-participant clip counts (showing imbalanced participants only):")
found = False
for pid, conds in sorted(per_participant.items()):
    counts = {c: len(clips) for c, clips in conds.items()}
    if len(set(counts.values())) > 1:
        found = True
        print(f"  {pid}: A={counts.get('A',0)}, V={counts.get('V',0)}, AV={counts.get('AV',0)}")
        for cond, clips in conds.items():
            print(f"    {cond}: {sorted(clips, key=lambda x: int(x))}")
if not found:
    print("  None — all participants have equal counts across conditions")

# Write one averaged row per clip
def write_clip_labels(rows, out_path, label):
    """Write clip-level averages for perceived duration."""
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r['clip_id']].append(r['response'])

    results = []
    for clip_id, responses in sorted(by_clip.items(), key=lambda x: int(x[0])):
        results.append({
            'clip_id':      clip_id,
            'perceived_s':  round(statistics.mean(responses), 3),
            'n_responses':  len(responses),
            'std_s':        round(statistics.stdev(responses), 3) if len(responses) > 1 else 0.0,
        })

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['clip_id', 'perceived_s', 'n_responses', 'std_s'])
        writer.writeheader()
        writer.writerows(results)

    mean_perceived = statistics.mean(r['perceived_s'] for r in results)
    print(f"\n{label} (clip-level averages) → {out_path.name}")
    print(f"  {len(results)} clips, mean perceived = {mean_perceived:.2f}s")


# Write one row per trial response
def write_trial_labels(rows, out_path, label):
    """Write trial-level responses in the format used for evaluation."""
    rows_sorted = sorted(rows, key=lambda r: (r['participant_id'], int(r['clip_id'])))

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'participant_id', 'clip_id', 'condition',
            'response', 'actual_duration', 'rt'
        ])
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"{label} (trial-level) → {out_path.name}")
    print(f"  {len(rows_sorted)} individual trials")


# Create clip-level averages for calibration
print("\n── Clip-level averages ──────────────────────────────────────────")
write_clip_labels(
    [r for r in clean if r['condition'] == 'A'],
    OUT_DIR / "audio_perceived.csv",
    "Audio-only (A)"
)
write_clip_labels(
    [r for r in clean if r['condition'] == 'V'],
    OUT_DIR / "video_perceived.csv",
    "Video-only (V)"
)
write_clip_labels(
    [r for r in clean if r['condition'] == 'AV'],
    OUT_DIR / "av_perceived.csv",
    "Audio+Video (AV)"
)

# Create trial-level files for evaluation
print("\n── Trial-level individual responses ─────────────────────────────")
write_trial_labels(
    [r for r in clean if r['condition'] == 'A'],
    OUT_DIR / "audio_trials.csv",
    "Audio-only (A)"
)
write_trial_labels(
    [r for r in clean if r['condition'] == 'V'],
    OUT_DIR / "video_trials.csv",
    "Video-only (V)"
)
write_trial_labels(
    [r for r in clean if r['condition'] == 'AV'],
    OUT_DIR / "av_trials.csv",
    "Audio+Video (AV)"
)

# Save all cleaned responses in one file
all_out = OUT_DIR / "all_responses.csv"
with open(all_out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'participant_id', 'clip_id', 'condition',
        'response', 'actual_duration', 'rt'
    ])
    writer.writeheader()
    writer.writerows(sorted(clean, key=lambda r: (r['participant_id'], int(r['clip_id']))))
print(f"\nAll responses → all_responses.csv ({len(clean)} trials)")
print("\nDone.")