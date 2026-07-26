# AI Okey Replay Viewer

Static, dependency-free spectator UI for deterministic replay JSON files.

Generate a trained checkpoint replay:

```powershell
python -m benchmarks.replay `
  --model-checkpoint training_runs\checkpoint-40.npz `
  --seeds 42 `
  --output-dir replay_runs\checkpoint-40 `
  --top-candidates 5
```

```powershell
python -m http.server 4173 --directory viewer
```

Open `http://127.0.0.1:4173`, then drag a generated replay JSON into the
right-hand drop zone. The built-in four-frame demo intentionally shows a dumb
checkpoint discarding a real Okey so the viewer is useful before training
artifacts exist.

The policy never receives spectator data. Full hands are a rendering-only
feature and can be hidden with the `Seyirci` toggle.

## Live training dashboard

Run a fresh 200-episode experiment, open the dashboard automatically, and
create videos at episodes 0, 20, 80, and 200:

```powershell
python -m benchmarks.train_progression `
  --episodes 200 `
  --checkpoints 0,20,80,200 `
  --seed 42 `
  --replay-seed 42 `
  --evaluation-episodes 20 `
  --output-dir training_runs\progression-200 `
  --serve-port 4173 `
  --open-dashboard `
  --keep-serving
```

The trainer publishes `status.json` atomically after every episode. The
dashboard polls that file, shows real loss/score/action telemetry, and adds a
video player only after a checkpoint replay and MP4 are fully written.
`--keep-serving` leaves the completed dashboard available until `Ctrl+C`.
Use a new or empty `--output-dir` for each run; existing run data is never
overwritten.

Create a shareable H.264 video (Pillow and FFmpeg required):

```powershell
python -m benchmarks.render_replay `
  replay_runs\checkpoint-40\checkpoint-40-seed-42.json `
  --output replay_runs\checkpoint-40\checkpoint-40-seed-42.mp4 `
  --fps 2
```
