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

Create a shareable H.264 video (Pillow and FFmpeg required):

```powershell
python -m benchmarks.render_replay `
  replay_runs\checkpoint-40\checkpoint-40-seed-42.json `
  --output replay_runs\checkpoint-40\checkpoint-40-seed-42.mp4 `
  --fps 2
```
