from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, timezone

timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Expanded ActionType to include CONFIRM
ActionType = Literal[
    "ACCEPT",
    "COUNTER",
    "REJECT",
    "ASK_CLARIFY",
    "OFFER_ADDON",
    "CONFIRM",          # new
]

class Product(BaseModel):
    id: str
    name: str
    category: str
    base_price: int
    min_price: Optional[int] = None
    currency: str = "NGN"
    stock: int = 100

class MerchantConfig(BaseModel):
    min_discount_ratio: float = 0.6   # do not go below 60% of base by default
    auto_approve_above_ratio: float = 0.9
    require_approval_below_ratio: float = 0.65
    language_default: str = "en"

class Turn(BaseModel):
    speaker: Literal["buyer", "bot"]
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parsed_offer: Optional[int] = None
    action: Optional[ActionType] = None

class ConversationState(BaseModel):
    conversation_id: str
    user_id: str
    product_id: Optional[str] = None
    turns: List[Turn] = Field(default_factory=list)
    last_offer: Optional[int] = None
    last_counter: Optional[int] = None
    stage: Literal["idle", "negotiating", "closed", "pending_merchant"] = "idle"
    buyer_score: float = 0.0

class DecisionOutput(BaseModel):
    action: ActionType
    price: Optional[int] = None
    confidence: float = 0.0
    strategy_tag: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class EngineOutput(BaseModel):
    action: ActionType
    response_text: str
    lang: str
    price: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
