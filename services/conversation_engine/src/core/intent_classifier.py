import re
from typing import Tuple

INTENTS = ("GREETING", "ASK_PRICE", "NEGOTIATE", "BUY", "OTHER")

def classify_intent(text: str, lang: str = "en") -> Tuple[str, float]:
    """
    Simple heuristic-based intent classifier.
    Returns (intent, confidence).
    """
    t = (text or "").lower().strip()

    # GREETING
    if any(g in t for g in ("hi", "hello", "hey", "how far", "howfa")):
        return "GREETING", 0.85

    # ASK_PRICE
    if any(k in t for k in ("price", "how much", "cost", "how much be")):
        return "ASK_PRICE", 0.8

    # NEGOTIATE
    if (
        any(k in t for k in ("can i pay", "i fit pay", "last price", "offer", "take", "accept", "i pay", "i can pay"))
        or re.search(r"\b\d{3,}\b", t)  # detect numeric offers like 5000, 1000
    ):
        return "NEGOTIATE", 0.9

    # BUY
    if any(k in t for k in ("i'll buy", "i go buy", "pay now", "buy now", "i will buy", "i wan buy")):
        return "BUY", 0.9

    # Default fallback
    return "OTHER", 0.4
