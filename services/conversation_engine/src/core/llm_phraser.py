from __future__ import annotations
from typing import Optional, Dict
import os
from .models import DecisionOutput, Product, ConversationState
from .utils import contains_required_price

_USE_MODEL = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    _MODEL_ID = os.getenv("YARNAI_LLM_MODEL", "")
    if _MODEL_ID:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, device_map="auto")
        _pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer)
        _USE_MODEL = True
except Exception:
    _USE_MODEL = False

TEMPLATES = {
    "en": {
        "ACCEPT": "Great news! I can accept {price_token} for {product}. Shall I proceed?",
        "COUNTER": "Thanks for the offer. I can do {price_token} for {product}. Can we shake on it?",
        "REJECT": "I wish I could match that. For {product}, I can’t go that low right now.",
        "ASK_CLARIFY": "Could you confirm your offer for {product}? What price do you have in mind?",
        "OFFER_ADDON": "I can add a little gift with {product}. Interested?"
    },
    "pcm": {
        "ACCEPT": "Na good news! I fit accept {price_token} for {product}. Make I run am?",
        "COUNTER": "I hail your offer. Make we do {price_token} for {product}. You gree?",
        "REJECT": "I for like match am but e low. For {product}, I no fit reach that level now.",
        "ASK_CLARIFY": "You fit confirm your price for {product}? How much you wan pay?",
        "OFFER_ADDON": "I fit add small jara for {product}. You dey interested?"
    },
    "yo": {
        "ACCEPT": "Ìròyìn ayọ̀! Mo lè gba {price_token} fún {product}. Ṣe kí n tẹ̀síwájú?",
        "COUNTER": "Ẹ ṣé fún ìfilọ́. Ẹ jọ̀ọ́ ẹ jẹ́ ká ṣe {price_token} fún {product}. Ṣé ó bá yín mu?",
        "REJECT": "Ó ṣòro kí n dé ibẹ̀. Fún {product}, mi ò lè lọ kéré bẹ́ẹ̀ báyìí.",
        "ASK_CLARIFY": "Ẹ lè jẹ́ kí n mọ owó tí ẹ ń ráyè fún {product}? Ẹ̀ẹ̀wo ni ẹ ní lórí?",
        "OFFER_ADDON": "Mo lè fi díẹ̀ ẹ̀bùn kún {product}. Ṣe ẹ nífẹ̀ẹ́ si?"
    }
}

def format_price(price: Optional[int], currency: str = "NGN") -> str:
    if price is None:
        return ""
    if currency.upper() == "NGN":
        return f"₦{price:,}"
    return f"{price:,} {currency}"

class LLMPhraser:
    def __init__(self, default_lang: str = "en"):
        self.default_lang = default_lang

    def generate(
        self,
        decision: DecisionOutput,
        product: Product,
        state: ConversationState,
        lang: str = "en",
        persona: Optional[str] = None,
    ) -> str:
        lang = (lang or self.default_lang).lower()
        if lang not in TEMPLATES:
            lang = "en"

        price_token = format_price(decision.price, product.currency)
        template = TEMPLATES[lang].get(decision.action, TEMPLATES[lang]["COUNTER"])
        text = template.format(price_token=price_token, product=product.name)

        if _USE_MODEL:
            prompt = (
                f"System: You are a polite marketplace seller. \n"
                f"NEVER change numeric prices. Use language code: {lang}. Keep it ≤40 words.\n"
                f"Action: {decision.action}\n"
                f"Price: {price_token}\n"
                f"Product: {product.name}\n"
                f"LastBuyerMsg: {(state.turns[-1].text if state.turns else '')}\n"
                f"Draft: {text}\n"
                f"Output a friendly, persuasive message that includes the exact price."
            )
            out = _pipe(prompt, max_new_tokens=64, temperature=0.2, do_sample=False)[0]["generated_text"]
            if decision.price is None or contains_required_price(out, decision.price):
                return out.strip()
        return text
