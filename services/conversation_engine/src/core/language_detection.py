from typing import Tuple

def detect_language(text: str) -> Tuple[str, float]:
    t = (text or "").lower()
    if any(w in t for w in ("abeg", "dey", "na", "wahala", "oga")):
        return "pcm", 0.8
    if any(w in t for w in ("ẹ", "ṣ", "ọ", "ń", "fẹ́", "jọ̀ọ́")):
        return "yo", 0.8
    return "en", 0.6
