# services/negotiation_service/app.py
import os
import re
import uuid
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple

import requests
from fastapi import FastAPI
from pydantic import BaseModel

from llm_phraser import phrase

logger = logging.getLogger("negotiation")
logging.basicConfig(level=logging.INFO)

# Config
LANG_DETECT_URL = os.getenv("LANGUAGE_DETECTION_URL", "http://language-detection:8000/detect")
LANG_DETECT_TIMEOUT = float(os.getenv("LANG_DETECT_TIMEOUT", "0.5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
DEFAULT_MIN_PRICE_RATIO = float(os.getenv("DEFAULT_MIN_PRICE_RATIO", "0.5"))

# Heuristic override threshold (if service confidence below this, heuristic may override)
LANG_DETECT_OVERRIDE_THRESHOLD = float(os.getenv("LANG_DETECT_OVERRIDE_THRESHOLD", "0.75"))

# Requests session (shared)
_session = requests.Session()

# Models
class ProductPayload(BaseModel):
    id: str
    name: str
    base_price: int
    min_price: Optional[int] = None

class DecideRequest(BaseModel):
    offer: Optional[int]
    product: ProductPayload
    # avoid dangerous mutable default; use Optional and default None
    state: Optional[Dict[str, Any]] = None

class DecideResponse(BaseModel):
    action: str
    price: Optional[int]
    confidence: float = 1.0
    strategy_tag: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    reply: Optional[str] = None

# -----------------------
# Language detection with small heuristic override
# -----------------------
def _heuristic_language_from_text(text: Optional[str]) -> Optional[str]:
    """Very small heuristic detector to catch short Nigerian-language phrases.
    Returns one of: 'yo','ha','ig','pcm' or None.
    """
    if not text:
        return None
    t = text.lower()

    # pidgin cues (common tokens/phrases)
    if re.search(r'\b(wetin|wat|gud|abeg|oya|bros|sis|no wahala|i fit|fit pay|make we|abeg|tori|how far|abeg,|abeg\.)\b', t):
        return "pcm"

    # yoruba cues
    if re.search(r'\b(mo|mo le|mo fẹ|mo fẹ́|eleyi|ra|jẹ|ẹ̀|ọrẹ́|ẹ ṣe|ọjọ|ọmọ|ẹ jọwọ|mo san|mo san|jo|bayi|elo)\b', t):
        return "yo"

    # hausa cues
    if re.search(r'\b(zan iya|ina so|za mu|nagode|na gode|sai|yau|don allah|don Allah|ka saye|za ka)\b', t):
        return "ha"

    # igbo cues
    if re.search(r'\b(biko|m nwere|m ga|zụta|nna|nne|kedu|ị ga|ị nwere|m nwere ike)\b', t):
        return "ig"

    return None

def detect_language_via_service(text: Optional[str]) -> Tuple[str, float]:
    """
    Call external language detection service, but use a small heuristic override
    when confidence is low or service returns 'en' for clearly-local text.

    Returns (language_code, confidence).
    Defensive: returns ('en', 0.5) on error or empty input.
    """
    if not text:
        return "en", 0.5

    # call external service
    try:
        resp = _session.post(LANG_DETECT_URL, json={"text": text}, timeout=LANG_DETECT_TIMEOUT)
        resp.raise_for_status()
        j = resp.json()
        svc_lang = (j.get("language") or "en").lower()
        svc_conf = float(j.get("confidence", 0.5))
    except Exception as e:
        logger.warning("[negotiation] language detect service failed: %s", e)
        svc_lang, svc_conf = "en", 0.5

    # heuristic override when confidence is low OR service returned 'en' but text contains local tokens
    heuristic = _heuristic_language_from_text(text)
    final_lang = svc_lang
    final_conf = svc_conf
    override_used = False

    if heuristic:
        if svc_conf < LANG_DETECT_OVERRIDE_THRESHOLD or svc_lang == "en":
            final_lang = heuristic
            # bump confidence to indicate heuristic preference (not a statistical confidence)
            final_conf = max(svc_conf, 0.85)
            override_used = True

    logger.info(
        "[negotiation] lang_detect service=%s conf=%.3f heuristic=%s override=%s final=%s final_conf=%.3f",
        svc_lang, svc_conf, heuristic, override_used, final_lang, final_conf
    )
    return final_lang, float(final_conf)

# Helpers
def _enforce_price_guard(resp: DecideResponse, product: ProductPayload) -> DecideResponse:
    try:
        proposed = resp.price
        base = product.base_price
        if proposed is None:
            return resp
        if product.min_price is not None and proposed < product.min_price:
            return DecideResponse(
                action="ESCALATE", price=proposed, confidence=resp.confidence,
                strategy_tag="merchant_min_violation", meta={"min_price": product.min_price}
            )
        default_min = int(base * DEFAULT_MIN_PRICE_RATIO)
        if proposed < default_min:
            return DecideResponse(
                action="ESCALATE", price=proposed, confidence=resp.confidence,
                strategy_tag="min_ratio_violation", meta={"min_price": default_min}
            )
        return resp
    except Exception as e:
        logger.exception("price guard error: %s", e)
        return DecideResponse(action="ESCALATE", price=resp.price, confidence=resp.confidence, strategy_tag="guard_error")

def _generate_decision_id(user_id: Optional[str] = None) -> str:
    base = str(uuid.uuid4())
    if user_id:
        hash_id = hashlib.sha256(str(user_id).encode()).hexdigest()[:8]
        return f"{base}-{hash_id}"
    return base

# FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    """
    Lightweight ping: checks language detection service status.
    """
    try:
        resp = _session.post(LANG_DETECT_URL, json={"text": "ping"}, timeout=0.5)
        if resp.status_code == 200:
            return {"status": "ok"}
    except Exception:
        pass
    return {"status": "fail"}

@app.post("/decide")
def decide(req: DecideRequest):
    decision_id = _generate_decision_id((req.state or {}).get("user_id") if req.state else None)

    buyer_text = (req.state or {}).get("meta", {}).get("buyer_text") if req.state else None
    # use offer+product name when buyer_text missing
    lang_input = buyer_text or f"{req.offer or ''} {req.product.name}"
    lang, lang_conf = detect_language_via_service(lang_input)

    # Log detected language and input for observability
    logger.info("[negotiation] decide: user_id=%s offer=%s product=%s detected_lang=%s lang_conf=%.3f",
                (req.state or {}).get("user_id"), req.offer, req.product.name, lang, lang_conf)

    # Decision logic (simple heuristics; keep backward-compatible)
    action = "COUNTER" if req.offer else "ASK_CLARIFY"
    price = req.offer if req.offer else req.product.base_price
    resp = DecideResponse(action=action, price=price)

    # Enforce guard rails (min price checks, etc.)
    resp = _enforce_price_guard(resp, req.product)

    # Build reply via phraser and gracefully handle phraser failures
    reply_text = None
    try:
        reply_text = phrase(resp.dict(), req.product.dict(), lang=lang, context=buyer_text)
    except Exception as e:
        logger.exception("phrase() failed: %s", e)

    if not reply_text:
        # fallback message if template generation somehow fails
        fallback_price = resp.price or req.product.base_price
        reply_text = f"Our counter price is ₦{int(fallback_price):,} for {req.product.name}."

    resp.reply = reply_text

    # Structured logging
    logger.info("negotiation_log: %s", {
        "decision_id": decision_id,
        "model_version": MODEL_VERSION,
        "user_id_hash": hashlib.sha256(str((req.state or {}).get("user_id") or "").encode()).hexdigest()[:8],
        "action": resp.action,
        "price": resp.price,
        "lang": lang,
        "lang_conf": lang_conf,
        "strategy_tag": resp.strategy_tag
    })

    return resp
