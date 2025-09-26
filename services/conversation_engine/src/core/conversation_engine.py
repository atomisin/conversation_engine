from __future__ import annotations
from .utils import normalize_text
from .amount_parser import parse_amount
from .language_detection import detect_language
from .intent_classifier import classify_intent
from .product_catalog import get_product, load_catalog
from .state_store import STATE
from .decision_model import DecisionModel
from .llm_phraser import LLMPhraser
from .price_guard import enforce_price_bounds, requires_approval
from .models import ConversationState, Turn, EngineOutput, MerchantConfig, DecisionOutput
from pathlib import Path
import os

_DECISION = DecisionModel()
_PHRASER = LLMPhraser()
_DEFAULT_MERCHANT = MerchantConfig()


def _ensure_catalog_loaded():
    """
    Try to lazily load the product catalog if it's empty. This helps tests
    that call handle_message directly without running the FastAPI startup hook.
    """
    # get_product will return None if catalog empty; we attempt to load a catalog
    # from a repo-local path: services/conversation_engine/data/product_catalog.json
    # Pathing: current file is .../src/core/conversation_engine.py
    base = Path(__file__).resolve().parents[2]  # services/conversation_engine
    candidate = base / "data" / "product_catalog.json"
    if candidate.exists():
        try:
            load_catalog(str(candidate))
        except Exception:
            # If loading fails, silently continue; get_product will still return None
            pass


def handle_message(conversation_id: str, user_id: str, text: str, product_id: str | None = None) -> EngineOutput:
    text_norm = normalize_text(text)
    lang, lang_conf = detect_language(text_norm)
    intent, intent_conf = classify_intent(text_norm, lang)
    offer = parse_amount(text_norm)

    # load or create conversation state
    state = STATE.get(conversation_id) or ConversationState(conversation_id=conversation_id, user_id=user_id, product_id=product_id)
    # ensure user_id present on state
    state.user_id = state.user_id or user_id
    if product_id:
        state.product_id = product_id

    # ensure catalog loaded (useful for tests)
    _ensure_catalog_loaded()

    product = get_product(state.product_id) if state.product_id else None
    if not product:
        # product missing: include intent in meta so tests can assert meta["intent"]
        reply = "Kindly select a product, please."
        STATE.append_turn(conversation_id, Turn(speaker="buyer", text=text_norm, parsed_offer=offer))
        STATE.append_turn(conversation_id, Turn(speaker="bot", text=reply))
        return EngineOutput(
            action="ASK_CLARIFY",
            response_text=reply,
            lang=lang,
            price=None,
            meta={"reason": "missing_product", "intent": intent, "lang_conf": lang_conf, "intent_conf": intent_conf},
        )

    # append buyer turn
    STATE.append_turn(conversation_id, Turn(speaker="buyer", text=text_norm, parsed_offer=offer))

    # explicit BUY intent handler (confirm/order path)
    if intent == "BUY":
        # If buyer says "I'll buy" but provided no explicit offer, confirm and return price.
        price_token = product.base_price
        reply = f"Thanks — I'll reserve {product.name} for you at ₦{price_token:,}. Please confirm payment."
        STATE.append_turn(conversation_id, Turn(speaker="bot", text=reply, parsed_offer=price_token, action="CONFIRM"))
        STATE.upsert(state)
        return EngineOutput(
            action="CONFIRM",
            response_text=reply,
            lang=lang,
            price=price_token,
            meta={"intent": intent, "lang_conf": lang_conf, "intent_conf": intent_conf},
        )

    # negotiation / price questions
    if intent in ("NEGOTIATE", "ASK_PRICE") or offer:
        decision = _DECISION.predict_action(offer=offer, product=product, state=state)
        decision = enforce_price_bounds(product, _DEFAULT_MERCHANT, decision)
        response_text = _PHRASER.generate(decision, product, STATE.get(conversation_id), lang=lang)
        meta = {"intent": intent, "lang_conf": lang_conf, "intent_conf": intent_conf}
        if requires_approval(product, _DEFAULT_MERCHANT, decision):
            meta["pending_merchant"] = True
            state.stage = "pending_merchant"
        STATE.append_turn(conversation_id, Turn(speaker="bot", text=response_text, parsed_offer=decision.price, action=decision.action))
        STATE.upsert(STATE.get(conversation_id))
        return EngineOutput(action=decision.action, response_text=response_text, lang=lang, price=decision.price, meta=meta)

    # general response (greeting, other)
    from .cultural_intelligence import opening, closing

    msg = opening(lang) + f"{product.name} base price is ₦{product.base_price:,}. " + closing(lang)
    # include intent in meta
    STATE.append_turn(conversation_id, Turn(speaker="bot", text=msg))
    STATE.upsert(STATE.get(conversation_id))
    return EngineOutput(action="ASK_CLARIFY", response_text=msg, lang=lang, price=product.base_price, meta={"intent": intent, "lang_conf": lang_conf, "intent_conf": intent_conf})
