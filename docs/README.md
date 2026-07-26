# Teknik Dokümantasyon

Bu klasör projenin kural, mimari, doğrulama ve araştırma belgelerini içerir.
Projeyi çalıştırmak isteyen kullanıcılar önce kökteki
[`README.md`](../README.md) dosyasını okumalıdır.

## Okuma sırası

1. [`00_START_HERE.md`](00_START_HERE.md) — proje ilkeleri
2. [`01_GOALS_AND_SCOPE.md`](01_GOALS_AND_SCOPE.md) — hedef ve kapsam
3. [`02_RULES_SPEC.md`](02_RULES_SPEC.md) — authoritative oyun kuralları
4. [`03_ENGINE_ARCHITECTURE.md`](03_ENGINE_ARCHITECTURE.md) — motor mimarisi
5. [`04_ACTION_AND_STATE_DESIGN.md`](04_ACTION_AND_STATE_DESIGN.md) — state,
   action ve observation tasarımı
6. [`05_TEST_AND_VALIDATION_PLAN.md`](05_TEST_AND_VALIDATION_PLAN.md) — kalite
   kapıları
7. [`06_SELF_PLAY_AI_PLAN.md`](06_SELF_PLAY_AI_PLAN.md) — öğrenme planı
8. [`07_FUTURE_HAND_ADVISOR.md`](07_FUTURE_HAND_ADVISOR.md) — gelecek advisor
   katmanı
9. [`08_AGENT_WORKFLOW.md`](08_AGENT_WORKFLOW.md) — geliştirme iş akışı
10. [`09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md`](09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md)
    — kesin kararlar ve kontrollü belirsizlikler

Ek belgeler:

- [`10_MASTER_AGENT_PROMPT.md`](10_MASTER_AGENT_PROMPT.md) — tarihsel uygulama
  yönlendirmesi
- [`11_ENGINE_IMPLEMENTATION_STATUS.md`](11_ENGINE_IMPLEMENTATION_STATUS.md)
  — güncel tamamlanma, benchmark ve sınırlamalar

## Kaynak önceliği

Oyun kuralları için `02_RULES_SPEC.md`, kesin/açık proje kararları için
`09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md` authoritative kaynaktır.
`11_ENGINE_IMPLEMENTATION_STATUS.md` yalnızca mevcut uygulama durumunu raporlar.
