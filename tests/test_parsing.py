"""Tests for observation parsing from green agent messages."""

import pytest
from strategy.models import parse_observation_from_text, determine_action_type, Observation


SAMPLE_PROPOSE_MESSAGE = '''You are participating in the AgentBeats bargaining meta-game as 'challenger'.
Action: PROPOSE.
Return ONLY JSON. Preferred: {"allocation_self":[...],"allocation_other":[...],"reason":"..."}.
Observation:
```json
{
  "pair": "challenger_vs_soft",
  "game_index": 0,
  "role": "row",
  "valuations_self": [45, 72, 13],
  "batna_self": 120,
  "discount": 0.98,
  "max_rounds": 5,
  "quantities": [7, 4, 1],
  "round_index": 1,
  "player_index": 0,
  "action": "propose",
  "pending_offer": {}
}
```'''

SAMPLE_ACCEPT_MESSAGE = '''You are participating in the AgentBeats bargaining meta-game as 'challenger'.
Action: ACCEPT_OR_REJECT.
Return ONLY JSON: {"accept": true|false, "reason": "..."}.
Observation:
```json
{
  "pair": "challenger_vs_tough",
  "game_index": 3,
  "role": "col",
  "valuations_self": [88, 15, 67],
  "batna_self": 200,
  "discount": 0.98,
  "max_rounds": 5,
  "quantities": [7, 4, 1],
  "round_index": 2,
  "player_index": 1,
  "action": "ACCEPT_OR_REJECT",
  "pending_offer": {
    "proposer": "row",
    "offer_allocation_self": [1, 0, 0],
    "offer_allocation_other": [6, 4, 1]
  },
  "offer_value": 88,
  "batna_value": 200
}
```'''


def test_parse_propose_message():
    obs = parse_observation_from_text(SAMPLE_PROPOSE_MESSAGE)
    assert obs.valuations_self == [45, 72, 13]
    assert obs.batna_self == 120
    assert obs.discount == 0.98
    assert obs.max_rounds == 5
    assert obs.quantities == [7, 4, 1]
    assert obs.round_index == 1
    assert obs.is_propose is True


def test_parse_accept_message():
    obs = parse_observation_from_text(SAMPLE_ACCEPT_MESSAGE)
    assert obs.valuations_self == [88, 15, 67]
    assert obs.batna_self == 200
    assert obs.is_accept_or_reject is True
    assert obs.pending_offer is not None
    assert obs.pending_offer["offer_allocation_other"] == [6, 4, 1]
    assert obs.offer_value == 88


def test_total_value():
    obs = Observation(
        valuations_self=[45, 72, 13],
        quantities=[7, 4, 1],
        batna_self=120,
    )
    assert obs.total_value == 45 * 7 + 72 * 4 + 13 * 1  # 315 + 288 + 13 = 616


def test_determine_action_type_propose():
    assert determine_action_type('Action: PROPOSE.') == "propose"


def test_determine_action_type_accept():
    assert determine_action_type('Action: ACCEPT_OR_REJECT.') == "accept_or_reject"


def test_parse_invalid_message():
    with pytest.raises(ValueError):
        parse_observation_from_text("This is not a valid message at all")
