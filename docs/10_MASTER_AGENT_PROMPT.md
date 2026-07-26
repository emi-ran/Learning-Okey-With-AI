# Master Prompt for the Coding Agent

You are implementing a high-performance, test-driven 101 Okey engine and later a self-play reinforcement-learning system.

The repository contains the authoritative project documentation:

```text
00_START_HERE.md
01_GOALS_AND_SCOPE.md
02_RULES_SPEC.md
03_ENGINE_ARCHITECTURE.md
04_ACTION_AND_STATE_DESIGN.md
05_TEST_AND_VALIDATION_PLAN.md
06_SELF_PLAY_AI_PLAN.md
07_FUTURE_HAND_ADVISOR.md
08_AGENT_WORKFLOW.md
09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md
```

Read all of them before making architecture decisions.

## Core priority

Do **not** begin by building a neural network.

First build a correct, deterministic, fast, headless 101 Okey engine.

The engine is the foundation for self-play. A model trained on an incorrect engine is useless.

## Critical separation

The solver/engine must know deterministic rules and enumerate legal candidates.

The AI must learn strategy.

Do not force the AI to discover whether 3-4-5 is a valid run, how 101 is calculated, whether an action is legal, or how physical rack sorting works.

The AI should learn which legal action is strategically best.

## Required development order

1. Tile/deck/Okey primitives.
2. Meld/pair/joker validation.
3. Round state machine.
4. Opening rules.
5. Table attachments.
6. Discard pickup constraints.
7. Joker replacement.
8. Discard and penalties.
9. Finish and scoring.
10. Legal action generation.
11. Invariants and stress tests.
12. Performance benchmark.
13. RL environment.
14. Random baseline.
15. Neural self-play baseline.
16. Evaluation league.
17. Human hand advisor compatibility.

## Correctness

Every physical tile must be conserved.

No hidden opponent hand information may leak into policy observations.

All illegal actions should be masked/omitted from RL action candidates.

Legal-but-penalized actions such as playable-tile discard and normal Okey discard must remain available to the policy.

## Unknown rules

Never silently invent behavior for an unresolved scoring edge case.

Consult `09_LOCKED_DECISIONS_AND_OPEN_ITEMS.md`.

Represent unresolved scoring behavior in explicit config with regression tests.

## Testing

Use pytest.

After rule tests pass, run RandomAgent stress simulations and save the seed/action trace of every failure.

Target at least 100,000 complete rounds before calling the engine stable; aim for 1,000,000.

## Performance

Correctness first.

Then profile.

Focus on:

- meld enumeration,
- opening candidate generation,
- legal action generation,
- state copies,
- vectorized simulation.

Training should be able to run headlessly and later on Colab with CPU simulation plus batched GPU inference.

## Deliverables before RL

Before starting neural training, provide:

- passing test report,
- rule coverage summary,
- random stress-test results,
- benchmark results,
- known limitations,
- reproducible commands.

Only then proceed to the self-play AI phase.
