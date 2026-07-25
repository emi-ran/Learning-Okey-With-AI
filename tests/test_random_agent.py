from __future__ import annotations

import pytest

from okey101.agents.base import NoLegalActionsError
from okey101.agents.random_agent import RandomAgent


def test_random_agent_only_returns_supplied_legal_actions() -> None:
    legal_actions = ("draw", "discard-7", "open")
    agent: RandomAgent[object, str] = RandomAgent(seed=19)

    selected = {
        agent.select_action(object(), legal_actions)
        for _ in range(100)
    }

    assert selected <= set(legal_actions)
    assert selected == set(legal_actions)


def test_random_agent_is_reproducible_after_reset() -> None:
    legal_actions = tuple(range(8))
    agent: RandomAgent[None, int] = RandomAgent(seed=73)
    first = [agent.select_action(None, legal_actions) for _ in range(20)]

    agent.reset(seed=73)
    second = [agent.select_action(None, legal_actions) for _ in range(20)]

    assert second == first


def test_random_agent_rejects_empty_legal_action_list() -> None:
    agent: RandomAgent[None, object] = RandomAgent(seed=1)

    with pytest.raises(NoLegalActionsError, match="no legal actions"):
        agent.select_action(None, ())
