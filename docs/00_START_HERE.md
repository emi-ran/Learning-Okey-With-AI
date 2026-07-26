# 101 Okey Self-Play AI — Start Here

> Kurulum, eğitim ve replay kullanımı için kökteki
> [`README.md`](../README.md) dosyasına bakın. Bu belge teknik tasarımın
> başlangıç noktasıdır.

## Projenin ana hedefi

Bu projenin amacı, **101 Okey'i kendi kendine oynayarak öğrenen bir yapay zekâ** geliştirmektir.

Dört oyuncu aynı öğrenen politika/model ailesini kullanarak birbirlerine karşı oynayacak. Model, insan tarafından yazılmış "iyi strateji" kurallarıyla yönetilmeyecek. Stratejiyi self-play sonucunda öğrenmesi hedeflenecek.

Ancak **oyunun kuralları AI tarafından keşfedilmeyecek**. Kurallar, legal hamleler, puanlama ve oyun akışı tamamen deterministik bir oyun motoru tarafından uygulanacak.

En önemli öncelik:

> **Önce kusursuz, hızlı ve test edilmiş 101 Okey engine. Sonra AI.**

Yanlış bir engine üzerinde milyonlarca oyun oynatmak, yanlış oyunu çok iyi oynayan bir model üretir.

---

# Belge sırası

Agent bu belgeleri aşağıdaki sırayla okumalıdır:

1. `01_GOALS_AND_SCOPE.md`
2. `02_RULES_SPEC.md`
3. `03_ENGINE_ARCHITECTURE.md`
4. `04_ACTION_AND_STATE_DESIGN.md`
5. `05_TEST_AND_VALIDATION_PLAN.md`
6. `06_SELF_PLAY_AI_PLAN.md`
7. `07_FUTURE_HAND_ADVISOR.md`
8. `08_AGENT_WORKFLOW.md`
9. `09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md`

Güncel implementation ve doğrulama durumu:

- `11_ENGINE_IMPLEMENTATION_STATUS.md`

---

# Temel tasarım kararı: AI taşları fiziksel olarak dizmeyi öğrenmeyecek

AI'ya raftaki taşları soldan sağa dizme problemi verilmemelidir.

Örneğin aynı el:

```text
Kırmızı 3, Kırmızı 5, Mavi 9, Kırmızı 4
```

ve

```text
Mavi 9, Kırmızı 4, Kırmızı 3, Kırmızı 5
```

AI için aynı oyun durumu olmalıdır.

Engine/Solver otomatik olarak:

```text
Kırmızı 3-4-5
```

aday serisini bulmalıdır.

AI'nın görevi:

- bu seriyi kullanıp kullanmamak,
- ne zaman açmak,
- hangi kombinasyonu seçmek,
- Okeyi nerede kullanmak,
- hangi taşı atmak,
- yandan taşı almak veya almamak,
- rakibe taş vermemek,
- risk/ödül dengesini kurmak

olmalıdır.

AI'nın görevi **"3-4-5 seri midir?"** sorusunu öğrenmek değildir. Bu kesin bir oyun kuralıdır.

---

# Geliştirme sırası

```text
Rule Specification
        ↓
Deterministic Engine
        ↓
Rule / Invariant Tests
        ↓
Legal Action Generator
        ↓
Random Agents
        ↓
100k–1M Random Simulation Stress Test
        ↓
Fast RL Environment
        ↓
Self-Play Baseline
        ↓
Neural Policy / Value Model
        ↓
Evaluation League
        ↓
Human Hand Advisor
        ↓
Possible Real-Game Integration
```

Engine testleri geçmeden RL eğitimine başlanmamalıdır.

---

# Teknoloji yönü

İlk tercih:

- Python
- NumPy
- PyTorch
- Gymnasium benzeri environment arayüzü
- pytest
- multiprocessing / vectorized environments
- checkpoint tabanlı self-play

Kod mümkün olduğunca framework bağımsız tasarlanmalıdır.

Engine, PyTorch'a bağımlı olmamalıdır.

---

# Agent için birinci görev

İlk implementation hedefi AI değildir.

İlk hedef:

> Kurallara göre eksiksiz çalışan, dört legal-random oyuncuyla yüz binlerce eli hatasız simüle edebilen headless 101 Okey motoru.

Bu milestone tamamlanmadan neural network yazma.
