# AI Okey Viewer

Bu klasör iki dependency-free web arayüzü içerir:

- `index.html`: doğrulanmış replay'leri hamle hamle izleyen spectator masası
- `training.html`: episode telemetrisi ve checkpoint videolarını gösteren
  canlı eğitim paneli

Kullanıcı kurulumu ve uçtan uca eğitim komutları için kökteki
[`README.md`](../README.md) dosyasına bakın.

## Bağımsız replay viewer

Proje kökünde:

```powershell
python -m http.server 4173
```

Sonra `http://127.0.0.1:4173/viewer/` adresini açın ve replay JSON'unu
**Replay aç** düğmesiyle seçin.

Viewer elleri seri/per/çift adaylarına göre yalnızca görsel olarak gruplar;
oyun state'i veya policy girdisi değiştirilmez. Tam rakip elleri de yalnızca
spectator katmanında bulunur.

## Canlı eğitim paneli

```powershell
python -m benchmarks.train_progression `
  --episodes 200 `
  --checkpoints 0,20,80,200 `
  --seed 42 `
  --replay-seed 42 `
  --evaluation-episodes 20 `
  --output-dir artifacts\training\progression-200 `
  --serve-port 4173 `
  --open-dashboard `
  --keep-serving
```

Trainer her episode sonunda `status.json` dosyasını atomik olarak yayımlar.
Panel bu dosyayı izler; checkpoint replay ve MP4 dosyaları tamamen yazılmadan
video bağlantısını göstermez. Var olan dolu bir output klasörünün üzerine
yazılmaz.

## Tarayıcı smoke testleri

Sunucu çalışırken:

```powershell
python viewer\smoke_test.py `
  --url http://127.0.0.1:4173/viewer/ `
  --replay artifacts\replays\random\random-0-seed-0.json
```

Canlı panel testi:

```powershell
python viewer\training_smoke_test.py `
  --base-url http://127.0.0.1:4173 `
  --status /artifacts/training/live-200/status.json
```
