"""Tests for heuristic negotiation strategy."""

from strategy.models import Observation, ProposalResponse, QUANTITIES
from strategy.heuristics import aspiration_propose, aspiration_accept_or_reject


def test_proposal_valid_allocation():
    obs = Observation(
        valuations_self=[45, 72, 13],
        batna_self=120,
        discount=0.98,
        max_rounds=5,
        quantities=[7, 4, 1],
        round_index=1,
        action="propose",
    )
    proposal = aspiration_propose(obs)
    # Check allocations sum to quantities
    for s, o, q in zip(proposal.allocation_self, proposal.allocation_other, QUANTITIES):
        assert s + o == q
        assert s >= 0
        assert o >= 0


def test_proposal_exceeds_batna():
    obs = Observation(
        valuations_self=[45, 72, 13],
        batna_self=120,
        discount=0.98,
        max_rounds=5,
        quantities=[7, 4, 1],
        round_index=1,
        action="propose",
    )
    proposal = aspiration_propose(obs)
    self_value = sum(v * a for v, a in zip(obs.valuations_self, proposal.allocation_self))
    assert self_value >= obs.batna_self


def test_accept_good_offer():
    obs = Observation(
        valuations_self=[45, 72, 13],
        batna_self=120,
        discount=0.98,
        max_rounds=5,
        quantities=[7, 4, 1],
        round_index=2,
        action="ACCEPT_OR_REJECT",
        offer_value=400,  # Way above BATNA
    )
    resp = aspiration_accept_or_reject(obs)
    assert resp.accept is True


def test_reject_bad_offer():
    obs = Observation(
        valuations_self=[45, 72, 13],
        batna_self=500,
        discount=0.98,
        max_rounds=5,
        quantities=[7, 4, 1],
        round_index=1,
        action="ACCEPT_OR_REJECT",
        offer_value=50,  # Way below BATNA
    )
    resp = aspiration_accept_or_reject(obs)
    assert resp.accept is False


def test_proposal_concedes_over_rounds():
    obs1 = Observation(
        valuations_self=[50, 50, 50],
        batna_self=50,
        discount=0.98,
        max_rounds=5,
        quantities=[7, 4, 1],
        round_index=1,
        action="propose",
    )
    obs3 = Observation(
        valuations_self=[50, 50, 50],
        batna_self=50,
        discount=0.98,
        max_rounds=5,
        quantities=[7, 4, 1],
        round_index=3,
        action="propose",
    )
    p1 = aspiration_propose(obs1)
    p3 = aspiration_propose(obs3)
    val1 = sum(v * a for v, a in zip(obs1.valuations_self, p1.allocation_self))
    val3 = sum(v * a for v, a in zip(obs3.valuations_self, p3.allocation_self))
    # Should concede over rounds (keep less in later rounds)
    assert val3 <= val1


def test_proposal_response_validation():
    # Valid proposal
    p = ProposalResponse(allocation_self=[5, 3, 1], allocation_other=[2, 1, 0])
    assert p.allocation_self == [5, 3, 1]

    # Auto-compute allocation_other
    p2 = ProposalResponse(allocation_self=[4, 2, 0])
    assert p2.allocation_other == [3, 2, 1]
