# services/negotiation_service/llm_phraser.py
from __future__ import annotations
import os
import re
import threading
import logging
import random
import math
from typing import Dict, Any, Optional, Tuple, List
import requests

logger = logging.getLogger("llm_phraser")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ------------------ Config (env) ------------------
LLM_MODE = os.getenv("LLM_MODE", "REMOTE").upper()            # REMOTE | LOCAL | TEMPLATE
LLM_REMOTE_PROVIDER = os.getenv("LLM_REMOTE_PROVIDER", "GROQ").upper()  # GROQ | HF | OPENAI
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-7b-instruct")
LLM_REMOTE_URL = os.getenv("LLM_REMOTE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "80"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# new: make remote timeout configurable
LLM_REMOTE_TIMEOUT = int(os.getenv("LLM_REMOTE_TIMEOUT", "20"))

# negotiation default min ratio (used when product.min_price missing)
DEFAULT_MIN_PRICE_RATIO = float(os.getenv("DEFAULT_MIN_PRICE_RATIO", "0.5"))

# ------------------ startup diagnostic (masked) ------------------
def _mask_key(s: Optional[str]) -> str:
    if not s:
        return "<missing>"
    s = str(s)
    return (s[:6] + "..." + s[-4:]) if len(s) > 12 else "<present>"

logger.info(
    "[llm_phraser] startup: LLM_MODE=%s LLM_PROVIDER=%s LLM_REMOTE_URL=%s LLM_MODEL=%s LLM_REMOTE_TIMEOUT=%s",
    LLM_MODE, LLM_REMOTE_PROVIDER, LLM_REMOTE_URL or "<none>", LLM_MODEL, LLM_REMOTE_TIMEOUT
)
logger.info(
    "[llm_phraser] startup keys: GROQ=%s LLM_API=%s OPENAI=%s HF=%s DEFAULT_MIN_PRICE_RATIO=%s",
    _mask_key(GROQ_API_KEY), _mask_key(LLM_API_KEY), _mask_key(OPENAI_API_KEY), _mask_key(HF_TOKEN), DEFAULT_MIN_PRICE_RATIO
)

# ------------------ requests session w/ retries ------------------
from requests.adapters import HTTPAdapter, Retry

_session = requests.Session()
_retries = Retry(total=2, backoff_factor=0.25, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST", "GET"])
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# ------------------ runtime helpers ------------------
def _auth_headers() -> Dict[str, str]:
    """
    Build Authorization headers: prefer provider-specific key, fall back to generic.
    Always include Content-Type.
    """
    h = {"Content-Type": "application/json"}
    provider = LLM_REMOTE_PROVIDER.upper()
    # Provider-specific priority
    if provider == "GROQ" and GROQ_API_KEY:
        h["Authorization"] = f"Bearer {GROQ_API_KEY}"
        return h
    # Generic fallbacks
    if LLM_API_KEY:
        h["Authorization"] = f"Bearer {LLM_API_KEY}"
        return h
    if OPENAI_API_KEY:
        h["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        return h
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
        return h
    # No auth found — return headers without Authorization; callers will log
    return h

_model_lock = threading.Lock()
_local_ready = False
_tokenizer = None
_model = None

# ------------------ context sanitizer ------------------
def _sanitize_context(ctx: Optional[str]) -> str:
    if not ctx:
        return ""
    s = re.sub(r'[\x00-\x1f\x7f]+', ' ', ctx)
    s = re.sub(r'\s+', ' ', s).strip()
    # remove currency symbols to avoid double-embedding them in templates
    s = re.sub(r'[₦$€£]', '', s)
    return s[:800]

# ------------------ local model loader (optional) ------------------
def _try_load_local_model():
    global _local_ready, _tokenizer, _model
    if _local_ready:
        return
    with _model_lock:
        if _local_ready:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            logger.info("[llm_phraser] Loading local model: %s", LLM_MODEL)
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
            try:
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL, load_in_4bit=True, device_map="auto", trust_remote_code=True
                )
                logger.info("[llm_phraser] Loaded local model in 4-bit mode.")
            except Exception:
                logger.warning("[llm_phraser] 4-bit load failed, trying standard load.")
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            _local_ready = True
            logger.info("[llm_phraser] Local model ready.")
        except Exception as e:
            logger.exception("[llm_phraser] Local model load failed: %s", e)
            _local_ready = False
            _tokenizer = None
            _model = None

# ------------------ response extractor (generic) ------------------
def _extract_text_from_remote_response(j: Any) -> Optional[str]:
    """General extractor used for HF/Groq/OpenAI-style shapes."""
    try:
        # dict shapes
        if isinstance(j, dict):
            # HF inference api common shape
            if "generated_text" in j and isinstance(j["generated_text"], str):
                return j["generated_text"].strip()
            # OpenAI/Groq style: choices -> message/content or text
            if "choices" in j and j["choices"]:
                c0 = j["choices"][0]
                if isinstance(c0, dict):
                    # chat-style
                    if "message" in c0 and isinstance(c0["message"], dict):
                        msg = c0["message"]
                        if "content" in msg and isinstance(msg["content"], str):
                            return msg["content"].strip()
                        # some providers use 'content' as list
                        if "content" in msg and isinstance(msg["content"], list):
                            parts = []
                            for p in msg["content"]:
                                if isinstance(p, dict) and "text" in p:
                                    parts.append(p["text"])
                                elif isinstance(p, str):
                                    parts.append(p)
                            if parts:
                                return " ".join(parts).strip()
                    # completions-style
                    if "text" in c0 and isinstance(c0["text"], str):
                        return c0["text"].strip()
            # groq/hf outputs list
            if "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
                out0 = j["outputs"][0]
                if isinstance(out0, dict):
                    for key in ("generated_text", "text", "content", "prediction"):
                        if key in out0 and isinstance(out0[key], str):
                            return out0[key].strip()
                    cont = out0.get("content")
                    if isinstance(cont, list):
                        texts = []
                        for block in cont:
                            if isinstance(block, dict):
                                if "text" in block and isinstance(block["text"], str):
                                    texts.append(block["text"])
                                elif "content" in block and isinstance(block["content"], str):
                                    texts.append(block["content"])
                        if texts:
                            return " ".join(texts).strip()
        # list fallback
        if isinstance(j, list) and j:
            if isinstance(j[0], str):
                return j[0].strip()
            if isinstance(j[0], dict):
                if "generated_text" in j[0] and isinstance(j[0]["generated_text"], str):
                    return j[0]["generated_text"].strip()
                if "text" in j[0] and isinstance(j[0]["text"], str):
                    return j[0]["text"].strip()
    except Exception:
        pass
    return None

# ------------------ remote caller ------------------
def _call_remote_llm(prompt: str, timeout: int = None) -> Optional[str]:
    """
    Calls the configured remote LLM provider.
    Supports:
      - GROQ: OpenAI-compatible chat/completions shape (messages + max_tokens)
      - HF: Hugging Face Inference API (inputs + parameters)
      - OPENAI (when OPENAI_API_KEY used against openai-compatible endpoints)
    Returns generated text or None.
    """
    provider = LLM_REMOTE_PROVIDER.upper()
    headers = _auth_headers()
    url = LLM_REMOTE_URL or None
    timeout = timeout or LLM_REMOTE_TIMEOUT

    logger.debug("[llm_phraser] _call_remote_llm provider=%s url=%s model=%s timeout=%s", provider, url, LLM_MODEL, timeout)
    if not url:
        # sensible defaults for providers
        if provider == "GROQ":
            url = "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "HF":
            url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
        elif provider == "OPENAI":
            url = "https://api.openai.com/v1/chat/completions"
        else:
            logger.warning("[llm_phraser] No LLM_REMOTE_URL configured and no sane default for provider=%s", provider)
            return None

    # Warn when no Authorization header present for providers that expect it
    if "Authorization" not in headers:
        logger.warning("[llm_phraser] No Authorization header set for provider=%s — remote call may be rejected", provider)

    def _try_post(cur_url: str) -> Optional[requests.Response]:
        try:
            if provider == "GROQ" or provider == "OPENAI":
                payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a polite Nigerian market seller and negotiator."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P
                }
                return _session.post(cur_url, json=payload, headers=headers, timeout=timeout)

            if provider == "HF":
                payload = {"inputs": prompt, "parameters": {"temperature": LLM_TEMPERATURE, "max_new_tokens": LLM_MAX_TOKENS, "top_p": LLM_TOP_P}}
                return _session.post(cur_url, json=payload, headers=headers, timeout=timeout)

            logger.warning("[llm_phraser] Unknown provider in runtime: %s", provider)
            return None
        except Exception as e:
            logger.exception("[llm_phraser] _try_post exception for url=%s: %s", cur_url, e)
            return None

    tried_urls = []
    candidate_urls: List[str] = [url]
    # add common alternates for OpenAI/Groq endpoints
    if provider == "GROQ" or provider == "OPENAI":
        if "chat/completions" in url:
            candidate_urls.append(url.replace("chat/completions", "chat"))
            candidate_urls.append(url.replace("chat/completions", "completions"))
        if url.endswith("/chat"):
            candidate_urls.append(url + "/completions")
    for cur in candidate_urls:
        if cur in tried_urls:
            continue
        tried_urls.append(cur)
        logger.debug("[llm_phraser] Attempting remote LLM POST to %s", cur)
        resp = _try_post(cur)
        if resp is None:
            continue
        body_preview = (resp.text or "")[:4000]
        logger.debug("[llm_phraser] remote status=%s preview=%s", resp.status_code, body_preview[:1000])
        if resp.status_code >= 400:
            logger.warning("[llm_phraser] remote returned HTTP %s for %s: %s", resp.status_code, cur, body_preview[:1000])
            continue
        try:
            j = resp.json()
        except Exception:
            logger.exception("[llm_phraser] failed to parse JSON from remote response for %s", cur)
            j = None
        if j is not None:
            txt = _extract_text_from_remote_response(j)
            if txt:
                return txt.strip()
            # maybe the endpoint returns a simple string body
            if isinstance(j, str):
                return j.strip()
        # fallback: raw text
        if resp.text and len(resp.text) > 0:
            raw = resp.text.strip()
            if raw:
                return raw
    logger.warning("[llm_phraser] remote LLM call failed after trying %d URLs", len(tried_urls))
    return None

# ------------------ local generation helper ------------------
def _run_local_generation(prompt: str) -> Optional[str]:
    global _local_ready, _tokenizer, _model
    if not _local_ready:
        _try_load_local_model()
    if not _local_ready or _model is None or _tokenizer is None:
        return None
    try:
        import torch
        inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        gen = _model.generate(**inputs, max_new_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE, do_sample=True)
        out = _tokenizer.decode(gen[0], skip_special_tokens=True)
        if out.startswith(prompt):
            out = out[len(prompt):].strip()
        return out.strip()
    except Exception as e:
        logger.exception("[llm_phraser] local generation failed: %s", e)
        return None

# ------------------ templates (unchanged structure, lists allowed) ------------------
_TEMPLATES = {
    "en": {
        "accept": [
            "My dear, deal done! ₦{price:,} for {product}, you’ll shine! Pay now!",
            "Thank you, we’re set! ₦{price:,} for {product}, grab it quick!"
        ],
        "counter": [
            "Nice try, but let’s do ₦{price:,} for {product}. Take it now!",
            "I hear you! Best I can do is ₦{price:,} for {product}. Buy now?"
        ],
        "reject": [
            "Ouch, too low for {product}! Best is ₦{price:,}, don’t miss it!",
            "That price no fit o! ₦{price:,} for {product}, let’s deal!"
        ],
        "clarify": [
            "What price are you thinking for {product}? Share now, let’s deal!",
            "My friend, what’s your budget for {product}? Let me know!"
        ]
    },
    "pcm": {
        "accept": [
            "Correct deal! ₦{price:,} for {product}, e go sweet you! Pay sharp!",
            "Na so! ₦{price:,} for {product}, you go love am! Buy now!"
        ],
        "counter": [
            "You try, but I fit do ₦{price:,} for {product}. Oya, take am!",
            "No wahala, I go give ₦{price:,} for {product}. You dey in?",
        ],
        "reject": [
            "Haba, dat one too small for {product}! ₦{price:,}, abeg grab am!",
            "No way o, {product} worth ₦{price:,}! Oya, make we talk business!"
        ],
        "clarify": [
            "Abeg, wetin price you dey reason for {product}? Talk quick!",
            "Bros, which price you dey eye for {product}? Tell me now!"
        ]
    },
    "yo": {
        "accept": [
            "O ṣeun! ₦{price:,} fún {product}, ẹ rẹwà pẹ̀lú rẹ̀! Ra báyìí!",
            "Ẹ ṣé, a ti ṣe! ₦{price:,} fún {product}, kíá ra ni!",
        ],
        "counter": [
            "Ó dá, ṣùgbọ́n mo lè gba ₦{price:,} fún {product}. Ṣé ẹ rà?",
            "Ẹ̀gbọ́n, mo lè fun ọ ni ₦{price:,} fún {product}. Ra kíákíá!",
        ],
        "reject": [
            "Há, kéré jù fún {product}! ₦{price:,} ni mo lè gba, má ṣe yọ̀ọ̀!",
            "Kò tó fún {product}! ₦{price:,} ni, jọ̀ọ́, ṣé ẹ rà?",
        ],
        "clarify": [
            "Jọ̀ọ́, mélòó ni ẹ rò fún {product}? Sọ fún mi kíákíá!",
            "Ẹ̀gbọ́n, kí ni ìdíwọ̀n rẹ fún {product}? Sọ báyìí!"
        ]
    },
    "ha": {
        "accept": [
            "Nagode! ₦{price:,} domin {product}, zai sa ka haske! Saya yanzu!",
            "Na gode, mun gama! ₦{price:,} don {product}, ka saya da sauri!",
        ],
        "counter": [
            "Ka ji, amma zan iya ₦{price:,} don {product}. Ka saya yanzu?",
            "Nagode, zan iya baka ₦{price:,} don {product}. Shin za ka saya?",
        ],
        "reject": [
            "Haba, wannan ƙasa ga {product}! ₦{price:,}, ka ji yanzu!",
            "Farashin bai isa ba ga {product}! ₦{price:,}, ka saya ko?",
        ],
        "clarify": [
            "Don Allah, wane farashi kake tunani don {product}? Fada min yanzu!",
            "Aboki, wane farashi kake so don {product}? Ka gaya min!",
        ]
    },
    "ig": {
        "accept": [
            "Daalụ! ₦{price:,} maka {product}, ọ ga-eme gị ka ọgaranya! Zụta ugbu a!",
            "Ekele, anyị kwụsịrị! ₦{price:,} maka {product}, zụta ngwa ngwa!",
        ],
        "counter": [
            "Ị dị mma, mana m nwere ike inye ₦{price:,} maka {product}. Zụta ugbu a?",
            "Daalụ, enwere m ike ịnye ₦{price:,} maka {product}. Ị dị njikere?",
        ],
        "reject": [
            "Haba, ọnụahịa a dị ala maka {product}! ₦{price:,}, biko zụta ya!",
            "Ọ dịghị mma maka {product}! ₦{price:,} ka m nwere, zụta ugbu a!",
        ],
        "clarify": [
            "Biko, kedụ ọnụahịa ị na-eche maka {product}? Kwee ngwa ngwa!",
            "Nna, gịnị ka ị chọrọ maka ọnụahịa {product}? Kwee m ugbu a!"
        ]
    }
}

FEW_SHOT_PROMPT = """
    SYSTEM: You are a vibrant Nigerian market seller and expert negotiator, fluent in English (en), Pidgin (pcm), Yoruba (yo), Hausa (ha), and Igbo (ig), speaking like a native with authentic market energy. Use {lang_key} tone (short, direct, rich with Nigerian charm and banter).
        - Detect if a customer is speaking to you in Yoruba, English, Pidgin, Igbo, Hausa, or mixed languages and reply in the same language

        - Infuse replies with Nigerian market flair: use slang, proverbs, or playful haggling to build trust and warmth.

        - Appeal to emotions (gratitude, community, pride) to persuade

        - Be firm yet polite to protect seller margin and brand value.

        - Use currency symbol ₦ and round prices to whole naira.

        - Keep replies short (1-2 sentences, max 40 words). End with a clear CTA.

        - Do not invent shipping, freebies, or discounts.

        - Echo numeric prices exactly as provided in {final_price}.

Context: {context}

NUMERIC_GROUNDS:
final_price: {final_price}
"""

# ------------------ policy (unchanged) ------------------
def compute_counter_price(base_price: int, offer: Optional[int]) -> Tuple[str, Optional[int]]:
    if offer is None:
        return "ASK_CLARIFY", None
    try:
        base = int(base_price)
        off = int(offer)
    except Exception:
        return "ASK_CLARIFY", None
    if base <= 0:
        return "ASK_CLARIFY", None
    pct = off / base
    if pct >= 0.90:
        return "ACCEPT", off
    if pct >= 0.60:
        mid = int(round((base + off) / 2.0))
        return "COUNTER", mid
    alt = int(round(base * 0.625))
    return "REJECT", alt

# ------------------ improved numeric matcher ------------------
def _reply_contains_price(reply: str, price: int) -> bool:
    """
    Extract numeric tokens from reply (handles commas and currency) and compare equality.
    Returns True if any numeric token equals price.
    """
    if not reply or price is None:
        return False
    cleaned = re.sub(r'[₦$€£]', '', reply)
    tokens = re.findall(r'\d[\d,]*', cleaned)
    for t in tokens:
        try:
            val = int(t.replace(",", ""))
            if val == int(price):
                return True
        except Exception:
            continue
    return False

# ------------------ helpers for dynamic negotiation ------------------
def _compute_floor(min_price: Optional[int], base_price: int) -> int:
    """
    Compute merchant floor: at least 10% greater than merchant min_price (if provided),
    otherwise use DEFAULT_MIN_PRICE_RATIO on base_price as fallback, then apply 10% uplift.
    """
    try:
        if min_price is not None and int(min_price) > 0:
            mp = int(min_price)
        else:
            mp = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
        floor = int(math.ceil(mp * 1.10))
        return max(floor, 1)
    except Exception:
        return max(int(round(base_price * DEFAULT_MIN_PRICE_RATIO * 1.10)), 1)

def _initial_dynamic_counter(buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
    """
    Use previous dynamic algorithm but ensure we return at least the floor.
    """
    dyn = _dynamic_counter_price(buyer_offer, min_price, base_price)
    floor = _compute_floor(min_price, base_price)
    return max(dyn, floor)

def _next_proposal_after_reject(prev_proposals: List[int], buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
    """
    Given previous seller proposals (list), compute the next seller proposal when buyer rejected.
    Guarantees proposal >= floor (10% above min_price).
    Strategy:
      - If no previous_proposals: use initial dynamic counter anchored to floor.
      - Otherwise, start from last proposal and concede in controlled steps toward floor:
          step = max( ceil((last - floor) * 0.4), ceil(base_price * 0.03), 1 )
          next = max(last - step, floor)
    Returns integer next proposal.
    """
    floor = _compute_floor(min_price, base_price)
    try:
        if not prev_proposals:
            return _initial_dynamic_counter(buyer_offer, min_price, base_price)

        last = int(prev_proposals[-1])
        if last <= floor:
            return floor

        # compute step: at least small fraction of base_price, else 40% of remaining gap
        gap = max(last - floor, 0)
        step1 = int(math.ceil(gap * 0.40))
        step2 = int(math.ceil(base_price * 0.03))  # at least 3% of base
        step = max(step1, step2, 1)
        next_prop = last - step
        next_prop = max(next_prop, floor)
        return int(next_prop)
    except Exception as e:
        logger.exception("[llm_phraser] _next_proposal_after_reject error: %s", e)
        return floor

def _dynamic_counter_price(buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
    """
    Original dynamic logic (unchanged). Returns price >= min_price (merchant minimum)
    """
    try:
        # ensure integers
        mp = int(min_price) if min_price is not None else 0
        bp = int(base_price)
        if mp <= 0:
            # fallback min from base if merchant didn't set one
            mp = int(round(bp * DEFAULT_MIN_PRICE_RATIO))

        if buyer_offer is None:
            # No offer: start at ~75% of base but never below mp
            candidate = int(round(bp * 0.75))
            return max(candidate, mp)

        bo = int(buyer_offer)
        # if mp is zero somehow, guard divide by zero
        ratio = (bo / mp) if mp > 0 else (bo / bp if bp > 0 else 0.0)

        # primary anchors
        candidate75 = max(int(round(bp * 0.75)), mp)
        candidate60 = max(int(round(bp * 0.60)), mp)

        if ratio >= 0.8:
            # buyer very close to min: meet partway between buyer and candidate75 but protect min
            mid = int(round((bo + candidate75) / 2.0))
            # also allow slightly lower concession but never below mp
            concession = max(mp, min(mid, candidate75))
            return concession

        if 0.5 <= ratio < 0.8:
            # moderately close: prefer candidate75 but bias slightly toward buyer offer
            mid = int(round((candidate75 + bo) / 2.0))
            return max(mp, min(mid, candidate75))

        # far below: buyer offered <50% of mp -> keep a friendly proposal (candidate75),
        # encourage buyer to increase — but never go below mp
        return max(mp, candidate75)
    except Exception as e:
        logger.exception("[llm_phraser] _dynamic_counter_price error: %s", e)
        # absolute safe fallback
        try:
            fallback = int(round(base_price * 0.75))
            return max(fallback, int(min_price or 0))
        except Exception:
            return int(min_price or base_price)

def _format_naira(n: Optional[int]) -> str:
    try:
        return f"₦{int(n):,}"
    except Exception:
        return f"₦{n}"

# ------------------ template rendering helper ------------------
def _choose_template_variant(candidate: Any, ratio: Optional[float]) -> str:
    """
    If candidate is list, pick a variant based on ratio:
      - ratio >=0.8 -> prefer firmer variant (index 1 if exists)
      - 0.5..0.8 -> neutral variant (index 0)
      - ratio <0.5 -> friendlier variant (index 0)
    If candidate is str -> return as-is.
    """
    if isinstance(candidate, list):
        if not candidate:
            return ""
        # choose index safely
        if ratio is None:
            return random.choice(candidate)
        try:
            if ratio >= 0.8:
                idx = 1 if len(candidate) > 1 else 0
            elif ratio >= 0.5:
                idx = 0
            else:
                idx = 0
            return candidate[idx]
        except Exception:
            return random.choice(candidate)
    # not a list
    return str(candidate)

def _render_template_reply(template_map: Dict[str, Any], action_key: str, price: Optional[int], product_name: str, ratio: Optional[float] = None) -> str:
    """
    Safely render a template reply.
    - template_map: one language's templates (dict)
    - action_key: 'accept'|'counter'|'reject'|'clarify'
    - price: numeric price (may be None)
    - product_name: product display name
    - ratio: buyer_offer / min_price (optional) to select variant tone
    """
    try:
        candidate = template_map.get(action_key, template_map.get("counter"))
        tmpl = _choose_template_variant(candidate, ratio)

        if not isinstance(tmpl, str):
            tmpl = str(tmpl)

        tpl_price_int = None
        try:
            tpl_price_int = int(price) if price is not None else 0
        except Exception:
            tpl_price_int = 0

        return tmpl.format(price=tpl_price_int, product=product_name)
    except Exception as e:
        logger.exception("[llm_phraser] template rendering failed: %s", e)
        # safe fallback series
        try:
            return f"Our counter price is {_format_naira(price)} for {product_name}."
        except Exception:
            return f"Our counter price is ₦{price} for {product_name}."

# ------------------ main phrase() ------------------
def phrase(decision: Dict[str, Any], product: Dict[str, Any], lang: str = "en", context: Optional[str] = None) -> str:
    """
    decision: dict possibly containing 'action', 'price', 'offer', 'meta'
    product: dict with 'name' and 'base_price'
    lang: language key (en|pcm|yo|ig|ha)
    Returns a user-facing string reply.
    """
    # Determine language key used for template lookup
    lang_key = lang if lang in _TEMPLATES else ("pcm" if lang and lang.startswith("p") else "en")
    prod_name = product.get("name") or product.get("id") or "product"
    base_price = int(product.get("base_price", 0))

    # read decision fields
    explicit_action = (decision.get("action") or "").upper() or None
    explicit_price = decision.get("price")
    buyer_offer = None
    if decision.get("offer") is not None:
        try:
            buyer_offer = int(decision.get("offer"))
        except Exception:
            buyer_offer = None

    # negotiation meta (may carry min_price and previous proposals)
    meta = decision.get("meta") or {}
    min_price_meta = None
    try:
        if isinstance(meta, dict) and "min_price" in meta:
            min_price_meta = int(meta["min_price"])
    except Exception:
        min_price_meta = None

    prev_proposals: List[int] = []
    try:
        if isinstance(meta, dict) and "prev_proposals" in meta and isinstance(meta["prev_proposals"], list):
            # coerce to ints
            prev_proposals = [int(x) for x in meta["prev_proposals"] if isinstance(x, (int, str)) or hasattr(x, "__int__")]
    except Exception:
        prev_proposals = []

    floor = _compute_floor(min_price_meta, base_price)

    # If the decision is ESCALATE (from guard), compute a dynamic counter price >= floor
    if explicit_action == "ESCALATE":
        if min_price_meta is not None:
            # initial dynamic or next proposal depending on history
            if prev_proposals:
                dyn_price = _next_proposal_after_reject(prev_proposals, buyer_offer, min_price_meta, base_price)
            else:
                dyn_price = _initial_dynamic_counter(buyer_offer, min_price_meta, base_price)

            # enforce floor and set up next state
            dyn_price = max(dyn_price, floor)
            logger.info("[llm_phraser] ESCALATE -> dyn_price=%s (floor=%s) prev_proposals=%s buyer_offer=%s", dyn_price, floor, prev_proposals, buyer_offer)
            explicit_action = "COUNTER"
            explicit_price = dyn_price
            # include meta to aid caller in continuing negotiation
            meta = dict(meta or {})
            meta["next_proposal"] = dyn_price
            meta["floor"] = floor
            meta.setdefault("prev_proposals", prev_proposals)
        else:
            # no meta/min provided: fallback behavior
            explicit_action = "REJECT"
            explicit_price = explicit_price if explicit_price is not None else (buyer_offer or base_price)

    # If explicit_action present use it; otherwise compute server-side default
    if explicit_action:
        computed_action = explicit_action
        computed_price = explicit_price if explicit_price is not None else (buyer_offer or base_price)
    else:
        computed_action, computed_price = compute_counter_price(base_price, buyer_offer)

    # Ensure we never propose below floor when min_price is known
    if min_price_meta is not None and computed_price is not None:
        computed_price = max(int(computed_price), floor)

    final_price = int(computed_price) if computed_price is not None else None

    # sanitize context for prompts
    sanitized_ctx = _sanitize_context(context)
    fs = FEW_SHOT_PROMPT.format(lang_key=lang_key, context=sanitized_ctx, final_price=final_price)

    input_block = (
        f"\nINPUT:\nproduct_name: \"{prod_name}\"\n"
        f"base_price: {base_price}\n"
        f"offer: {buyer_offer if buyer_offer is not None else 'null'}\n"
        f"counter_price: {final_price if final_price is not None else 'null'}\n"
        f"decision: {computed_action}\n"
    )
    instruction = "\nINSTRUCTION:\nReply in one or two short sentences that are friendly, respectful, persuasive and end with a clear next step (CTA). Match the numeric values shown above exactly. Keep replies short and culturally appropriate."
    prompt = "\n".join(["SYSTEM PROMPT (few-shot examples):", fs, input_block, instruction])

    logger.debug("[llm_phraser] phrase() computed_action=%s final_price=%s prod=%s lang=%s floor=%s prev_proposals=%s",
                 computed_action, final_price, prod_name, lang_key, floor, prev_proposals)

    # Compute ratio for tone selection when templates are used
    min_price_for_ratio = None
    try:
        if min_price_meta is not None:
            min_price_for_ratio = int(min_price_meta)
        else:
            min_price_for_ratio = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
    except Exception:
        min_price_for_ratio = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))

    ratio = None
    try:
        if min_price_for_ratio > 0 and buyer_offer is not None:
            ratio = float(buyer_offer) / float(min_price_for_ratio)
    except Exception:
        ratio = None

    # --- LANGUAGE-BASED TEMPLATE OVERRIDE ---
    TEMPLATE_ONLY_LANGS = {"yo", "ha", "ig", "pcm"}
    if lang_key in TEMPLATE_ONLY_LANGS:
        template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
        action_map = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}
        key = action_map.get(computed_action, "counter")
        try:
            tpl_price = final_price if final_price is not None else base_price
            rendered = _render_template_reply(template_map, key, tpl_price, prod_name, ratio)
            # attach helpful meta for caller to continue negotiation if needed
            # include prev_proposals + this proposal (caller may persist)
            try:
                meta_out = dict(meta or {})
                prev = meta_out.get("prev_proposals", []) or []
                if computed_action == "COUNTER":
                    prev = list(prev) + [int(tpl_price)]
                    meta_out["prev_proposals"] = prev
                meta_out["floor"] = floor
            except Exception:
                meta_out = meta
            logger.info("[llm_phraser] template_override=true lang=%s action=%s price=%s prod=%s ratio=%s meta_prev=%s",
                        lang_key, computed_action, tpl_price, prod_name, ratio, meta_out.get("prev_proposals") if isinstance(meta_out, dict) else None)
            return rendered
        except Exception:
            logger.warning("[llm_phraser] template override failed for lang=%s key=%s — falling through", lang_key, key)

    # --- REMOTE preferred (English or other allowed languages) ---
    if LLM_MODE == "REMOTE":
        out = None
        try:
            out = _call_remote_llm(prompt)
            if out and (final_price is None or _reply_contains_price(out, final_price)):
                # if remote responded, return it
                return out.strip()
            logger.warning("[llm_phraser] remote returned no usable text or numeric mismatch; falling back")
        except Exception:
            logger.exception("[llm_phraser] remote generation error")

    # --- LOCAL fallback ---
    if LLM_MODE == "LOCAL":
        try:
            out = _run_local_generation(prompt)
            if out and (final_price is None or _reply_contains_price(out, final_price)):
                return out.strip()
        except Exception:
            logger.exception("[llm_phraser] local generation error")

    # --- TEMPLATE fallback (final) ---
    template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
    action_map = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}
    key = action_map.get(computed_action, "counter")
    try:
        tpl_price = final_price if final_price is not None else base_price
        # include prev_proposals update in logs to help caller persist
        logger.info("[llm_phraser] final_template lang=%s action=%s price=%s prod=%s prev_proposals=%s floor=%s",
                    lang_key, computed_action, tpl_price, prod_name, prev_proposals, floor)
        return _render_template_reply(template_map, key, tpl_price, prod_name, ratio)
    except Exception:
        # Final absolute fallback
        try:
            return f"Our counter price is {_format_naira(final_price)} for {prod_name}."
        except Exception:
            return f"Our counter price is ₦{final_price} for {prod_name}."
# ------------------ end of llm_phraser.py ------------------