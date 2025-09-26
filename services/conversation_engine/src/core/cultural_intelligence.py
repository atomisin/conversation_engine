from typing import Literal

def opening(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang == "pcm": return "How far! "
    if lang == "yo": return "Ẹ káàbọ̀! "
    return "Hello! "

def closing(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang == "pcm": return "Make we run am."
    if lang == "yo": return "Ẹ ṣé gan an."
    return "Shall we proceed?"
