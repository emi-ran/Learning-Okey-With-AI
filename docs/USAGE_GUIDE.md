# 101 Okey AI - Detaylı Kullanım ve Komut Rehberi

Bu belge; yerel eğitim, video üretimi, benchmark'lar, stres testleri ve detaylı CLI parametrelerini içerir.

---

## 1. Yerel Eğitim Komutları

### Temel CLI ile Hızlı Eğitim
```powershell
python -m okey101.training.cli `
  --episodes 1000 `
  --checkpoint artifacts/training/model-1k.npz `
  --evaluate 50 `
  --json
```

### Canlı İlerleme Paneli ile Eğitim (200 Episode)
```powershell
python -m benchmarks.train_progression `
  --episodes 200 `
  --checkpoints 0,20,80,200 `
  --seed 42 `
  --replay-seed 42 `
  --evaluation-episodes 20 `
  --output-dir artifacts/training/progression-200 `
  --serve-port 4173 `
  --open-dashboard `
  --keep-serving
```
*Not: Video üretimi istemiyorsanız komuta `--no-video` ekleyebilirsiniz.*

### Uzun Soluklu Eğitim (10.000+ Episode)
```powershell
python -m benchmarks.train_progression `
  --episodes 10000 `
  --checkpoints 0,100,1000,2000,5000,10000 `
  --seed 42 `
  --replay-seed 42 `
  --evaluation-episodes 100 `
  --output-dir artifacts/training/progression-10000 `
  --serve-port 4173 `
  --open-dashboard `
  --keep-serving
```

---

## 2. Replay Görüntüleme ve Video Üretimi

### A. Replay Viewer'ı Başlatma
```powershell
python -m http.server 4173
```
Tarayıcınızda `http://127.0.0.1:4173/viewer/` adresini açıp üretilen `.json` replay dosyasını seçin.

### B. Adım Adım MP4 Video Oluşturma

1. **Model Eğit:**
   ```powershell
   python -m okey101.training.cli --episodes 40 --seed 0 --checkpoint artifacts/training/checkpoint-40.npz
   ```
2. **Replay JSON Üret:**
   ```powershell
   python -m benchmarks.replay --model-checkpoint artifacts/training/checkpoint-40.npz --seeds 42 --output-dir artifacts/replays/checkpoint-40
   ```
3. **H.264 MP4 Olarak Render Al (FFmpeg gerekir):**
   ```powershell
   python -m benchmarks.render_replay artifacts/replays/checkpoint-40/checkpoint-40-seed-42.json --output artifacts/replays/checkpoint-40/checkpoint-40-seed-42.mp4 --fps 2
   ```

---

## 3. Test ve Stres Koşuları

```powershell
# Tüm testleri çalıştır (200 test)
python -m pytest

# Motor hız testi
python -m benchmarks.engine --rounds 100 --seed 0 --json

# Solver bellek ve performans testi
python -m benchmarks.solver --seeds 100 --measure-memory

# Çoklu çekirdek stres testi
python -m benchmarks.stress --rounds 1000 --workers 8
```
