# services/tests/test_engine.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "conversation_engine", "src"))

import pytest
from core.conversation_engine import handle_message


@pytest.fixture
def sku():
    # This must exist in your product_catalog.json
    return "sku-lip-001"


def test_greeting(sku):
    reply = handle_message("t1", "u1", "Hello there", product_id=sku)
    # reply is EngineOutput, so use attributes not dict subscripting
    assert hasattr(reply, "response_text")
    assert isinstance(reply.response_text, str)


def test_negotiation_midpoint(sku):
    reply = handle_message("t2", "u1", "I can pay 5000", product_id=sku)
    assert reply.meta["intent"] == "NEGOTIATE"
    assert isinstance(reply.response_text, str)


def test_low_offer_rejects_or_counters(sku):
    reply = handle_message("t3", "u1", "I pay 1000", product_id=sku)
    assert reply.meta["intent"] == "NEGOTIATE"
    assert reply.action in ["REJECT", "COUNTER"]


def test_buy_intent(sku):
    reply = handle_message("t4", "u1", "I'll buy", product_id=sku)
    assert isinstance(reply.response_text, str)
    assert reply.action in ["ACCEPT", "CONFIRM"]
