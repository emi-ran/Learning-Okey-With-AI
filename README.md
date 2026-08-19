# 🀄 101 Okey Self-Play AI

Deterministik bir 101 Okey motoru üzerinde dört yapay zekânın aynı nöral politikayı kullanarak kendi kendine öğrendiği (**Self-Play Reinforcement Learning**), canlı izlenebildiği ve replay/video olarak analiz edilebildiği açık kaynaklı araştırma projesi.

<p align="center">
  <img
    src="docs/assets/replay-viewer.webp"
    alt="101 Okey AI Spectator Replay Ekranı"
    width="100%"
  />
</p>

---

## ✨ Temel Özellikler

- **Deterministik 101 Okey Motoru:** 106 fiziksel taş, seri/çift perler, katlamalı açılış, yere işleme, okey alma ve ceza kurallarının eksiksiz simülasyonu.
- **Hafif ve Hızlı RL Mimarisi:** NumPy tabanlı Actor-Critic modeli ile tek çekirdekte saniyede 20+ el, çoklu çekirdekte 50-80+ el paralel eğitim hızı.
- **Canlı Takip Paneli & Spectator Viewer:** Eğitimi tarayıcıdan canlı izleme, kayıp (loss) grafikleri ve hamleleri adım adım inceleme.
- **Doğrulanabilir Replay ve Video Üretimi:** Sabit seed ile oynanan elleri JSON, poster veya H.264 MP4 videoya dönüştürme.
- **Google Colab Desteği:** Google Drive otomatik kayıtlı ve çoklu çekirdek hızlandırmalı hazır Jupyter Notebook (`colab_training.ipynb`).

<p align="center">
  <img
    src="docs/assets/training-dashboard.webp"
    alt="Canlı Eğitim Takip Paneli"
    width="100%"
  />
</p>

---

## ⚡ Hızlı Başlangıç

### 1. Kurulum
```powershell
# Sanal ortamı oluştur ve aktif et
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Bağımlılıkları kur
python -m pip install --upgrade pip
python -m pip install -e ".[dev,video]"
```

### 2. Testleri Çalıştır (200 Test)
```powershell
python -m pytest
```

### 3. Canlı Panelli Eğitimi Başlat (200 El)
```powershell
python -m benchmarks.train_progression `
  --episodes 200 `
  --checkpoints 0,20,80,200 `
  --seed 42 `
  --open-dashboard
```

---

## ☁️ Google Colab ile Bulutta Eğitim

Eğitimi bilgisayarınızı yormadan Google Colab üzerinde çalıştırmak ve modelleri doğrudan Google Drive'ınıza kaydetmek için [colab_training.ipynb](colab_training.ipynb) dosyasını kullanabilirsiniz:

- Çoklu çekirdek paralel hızlandırma devrededir.
- Her 5.000 elde bir otomatik `okey_model_5k.npz`, `10k.npz` checkpoint'i ve en yüksek kazanma oranına sahip `okey_model_best.npz` kaydedilir.
- RAM, işlemci yükü ve karar sürelerini anlık olarak raporlar.

---

## 🎬 Checkpoint Karşılaştırması

Farklı eğitim seviyelerindeki modellerin aynı elde nasıl oynadığını görsel olarak kıyaslayabilirsiniz:

<p align="center">
  <img
    src="docs/assets/checkpoint-comparison.webp"
    alt="0, 20, 80 ve 200 Episode Checkpoint Karşılaştırması"
    width="100%"
  />
</p>

---

## 📁 Proje Yapısı

```text
├── benchmarks/         # Performans, stres ve eğitim senaryoları
├── docs/               # Kural şartnameleri, mimari ve kılavuzlar
├── src/okey101/
│   ├── agents/         # Temel botlar (Random vb.)
│   ├── engine/         # 101 Okey kuralları ve oyun döngüsü
│   ├── replay/         # Deterministik replay kaydedici
│   ├── rl/             # RL ortamı, candidate encoder ve maskeleme
│   ├── solver/         # Per ve açılış algoritmaları
│   ├── training/       # NumPy Actor-Critic, Adam ve Paralel Trainer
│   └── visualization/  # MP4 render ve görselleştirme araçları
├── tests/              # 200 adet birim ve entegrasyon testi
├── viewer/             # Tarayıcı tabanlı Replay & Canlı Panel UI
├── colab_training.ipynb# Google Colab hazır eğitim laboratuvarı
└── pyproject.toml
```

---

## 📖 Detaylı Dokümantasyon

- 🛠️ [Detaylı Kullanım ve CLI Rehberi](docs/USAGE_GUIDE.md)
- 📜 [101 Okey Oyun Kuralları Şartnamesi](docs/02_RULES_SPEC.md)
- 🏗️ [Oyun Motoru Mimarisi](docs/03_ENGINE_ARCHITECTURE.md)
- 🤖 [Self-Play RL ve Nöral Ağ Tasarımı](docs/06_SELF_PLAY_AI_PLAN.md)
