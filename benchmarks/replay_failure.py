"""Reproduce the failing action captured by ``benchmarks.stress``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.engine.invariants import validate_invariants
from okey101.engine.round import RoundEngine, deserialize_action


def reproduce_failure(path: Path) -> Exception:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state_payload = payload.get("state")
    action_payload = payload.get("failed_action")
    if not isinstance(state_payload, dict):
        raise ValueError("Failure artifact has no serialized pre-action state")
    if not isinstance(action_payload, dict):
        raise ValueError("Failure artifact has no failed_action")

    engine = RoundEngine()
    engine.load_state(state_payload)
    action = deserialize_action(action_payload)
    try:
        state, _events = engine.step(action)
        validate_invariants(state)
    except Exception as error:
        return error
    raise RuntimeError("Captured action no longer reproduces a step/invariant failure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    error = reproduce_failure(args.artifact)
    print(f"reproduced: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
