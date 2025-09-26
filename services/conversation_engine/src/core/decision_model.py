from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import math
from .models import DecisionOutput, Product, ConversationState

@dataclass
class DecisionModel:
    def predict_action(self, *, offer: int | None, product: Product, state: ConversationState) -> DecisionOutput:
        base = product.base_price
        if not offer:
            return DecisionOutput(action="ASK_CLARIFY", price=None, confidence=0.55, strategy_tag="clarify")
        ratio = offer / max(base, 1)
        if ratio >= 0.9:
            return DecisionOutput(action="ACCEPT", price=offer, confidence=0.9, strategy_tag="high_offer")
        if 0.6 <= ratio < 0.9:
            counter = math.floor((offer + base) / 2 / 50) * 50
            return DecisionOutput(action="COUNTER", price=counter, confidence=0.75, strategy_tag="midpoint")
        counter = max(int(base * 0.95), offer)
        return DecisionOutput(action="COUNTER", price=counter, confidence=0.65, strategy_tag="anchor_high")
