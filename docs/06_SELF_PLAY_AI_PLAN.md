# Self-Play AI Plan

## 1. Temel hedef

Model 101 Okey stratejisini insan heuristiklerini doğrudan kopyalamadan self-play ile öğrenmelidir.

Engine legal actionları garanti eder.

Policy yalnızca legal seçenekler arasından karar verir.

---

# 2. Baseline sırası

## Baseline 0 — RandomAgent

Uniform random legal actions.

Amaç:

- engine validation,
- minimum performance baseline.

## Baseline 1 — SimpleHeuristicAgent

Yalnızca değerlendirme için basit bir bot oluşturulabilir.

Örneğin:

- mümkünse aç,
- yüksek tek taşları at,
- tamamlanmış meldleri koru.

Bu agent nihai model değildir.

Neural modelin random'dan ve basit heuristic'ten üstün olup olmadığını ölçmek için kullanılır.

---

# 3. Neural policy

İlk model küçük tutulmalıdır.

Observation encoder:

```text
own hand
public table
discard history summary
player statuses
thresholds
stock count
scores
```

çıktıları:

```text
policy
value
```

Model mimarisi ilk sürümde transformer olmak zorunda değildir.

MLP / structured encoder ile başlanabilir.

Doğru environment ve action formulation, model büyüklüğünden daha önemlidir.

---

# 4. RL algoritması

Başlangıç adayı:

- PPO + action masking + vectorized self-play

Alternatifler daha sonra:

- recurrent PPO,
- IMPALA-style,
- V-trace,
- actor-critic,
- policy/value + search,
- population based self-play.

İlk milestone için gereksiz algoritma karmaşıklığı oluşturma.

---

# 5. Reward

Nihai amaç round/match skorudur.

Ana reward terminal score'a dayanmalıdır.

Örnek normalize edilmiş form:

```text
reward = relative_final_score
```

Anlık cezalar da reward sinyaline yansıtılabilir.

Ancak aşırı reward shaping modelin gerçek hedef yerine yazılmış heuristikleri öğrenmesine neden olabilir.

Bu nedenle:

1. sparse terminal reward baseline,
2. gerekirse küçük ve kurala dayalı shaping

şeklinde ilerle.

"Seri tutmak iyidir" gibi stratejik shaping yazma.

---

# 6. Multi-agent self-play

İlk sürümde dört seat aynı policy ağırlıklarını kullanabilir.

```text
Policy θ vs Policy θ vs Policy θ vs Policy θ
```

Daha sonra league:

```text
current policy
recent checkpoint
older strong checkpoint
heuristic baseline
```

karışımı kullanılabilir.

Amaç self-play döngüsündeki stratejik unutmayı azaltmak.

---

# 7. Seat symmetry

Model dört seat'in hepsinde oynayabilmelidir.

Observation current-player perspektifine canonicalize edilmelidir.

Örneğin:

```text
self
left opponent
across opponent
right opponent
```

şeklinde relative seat representation kullanılabilir.

---

# 8. Hidden information

Opponent hands hiçbir zaman policy input'una verilmemelidir.

Training code'da accidental leakage testi yazılmalıdır.

Centralized critic kullanılırsa bile bunun deployment davranışını bozmayacağından emin olunmalı; ilk sürümde kaçınılması daha güvenlidir.

---

# 9. Checkpointing

Colab/free compute gibi kesintili ortamlarda training devam edebilmelidir.

Kaydet:

```text
model weights
optimizer state
training step
RNG states
league/checkpoint pool
config
metrics
git commit hash if available
```

Resume:

```text
load checkpoint
continue
```

---

# 10. Evaluation

Sadece training reward'a bakma.

Her checkpoint:

```text
vs RandomAgent
vs HeuristicAgent
vs previous checkpoints
```

çok sayıda el oynamalıdır.

Metrikler:

- average score differential,
- win/finish rate,
- unopened rate,
- penalties per game,
- Okey discard frequency,
- playable discard penalty frequency,
- average opening turn,
- pair strategy rate,
- stock-exhaustion outcome,
- illegal actions = always zero.

---

# 11. Training curriculum

Engine kuralını değiştirerek sahte kolay oyun öğretmek zorunlu değildir.

Ancak computational curriculum mümkün:

### Stage A
Single round.

### Stage B
More diverse rule configs.

### Stage C
Progressive/katlamalı.

### Stage D
Multi-round match strategy.

Modelin aynı architecture ile config bilgisini observation'dan alması tercih edilir.

---

# 12. Colab hedefi

İlk sistem Colab Free üzerinde çalışabilir olmalıdır.

CPU:

- environment simulation,
- legal action generation.

GPU:

- batched policy/value inference,
- neural training.

Environment'lar batch/vectorized tasarlanmalıdır.

---

# 13. En önemli araştırma sorusu

Action space çok büyürse doğrudan flat categorical policy yerine hierarchical policy gerekebilir.

Örneğin:

```text
phase head
candidate-type head
candidate-index head
discard head
```

Ancak bunu profiling ve candidate-count istatistikleri görmeden gereksiz yere karmaşıklaştırma.

Önce gerçek distribution ölç.
