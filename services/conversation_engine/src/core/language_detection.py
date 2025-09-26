"""
Robust language detection helper.
See docstring at top for behavior & environment.
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict
import os, re, threading, logging

logger = logging.getLogger("language_detection")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------------------------------------------------
# Config & environment
# -------------------------------------------------------------------
_FASTTEXT_PATH = os.getenv("FASTTEXT_MODEL_PATH", "").strip()
_FASTTEXT_SERVICE_URL = os.getenv("FASTTEXT_SERVICE_URL", "").strip()
_FASTTEXT_SERVICE_TIMEOUT = float(os.getenv("FASTTEXT_SERVICE_TIMEOUT", "2.0"))

# Override thresholds (tunable via env)
_CONF_OVERRIDE_THRESHOLD = float(os.getenv("LANG_DETECT_CONF_THRESHOLD", "0.95"))
_EVIDENCE_THRESHOLD = float(os.getenv("LANG_DETECT_EVIDENCE_THRESHOLD", "0.5"))
_STRONG_WEIGHT = float(os.getenv("LANG_DETECT_STRONG_WEIGHT", "1.5"))
_WEAK_WEIGHT = float(os.getenv("LANG_DETECT_WEAK_WEIGHT", "0.6"))
_STRONG_AUTO_OVERRIDE_THRESHOLD = float(os.getenv("LANG_DETECT_STRONG_AUTO_THR", "2.5"))
_OVERRIDE_CONFIDENCE = float(os.getenv("LANG_DETECT_OVERRIDE_CONF", "0.85"))

# NEW: low-confidence trust threshold for non-English fastText predictions
_LOW_CONF_TRUST = float(os.getenv("LANG_DETECT_LOW_CONF_TRUST", "0.6"))

# -------------------------------------------------------------------
# Marker dictionaries
# -------------------------------------------------------------------
STRONG_MARKERS_PCM = {"abi", "una", "wahala", "sef", "abeg", "jare", "omo", "no be", "How much for"}
WEAK_MARKERS_PCM   = {"fit", "dey", "go", "chop", "make", "come", "waka", "na", "biko"}

STRONG_MARKERS_YO  = {"ẹkọ", "ọ̀sán", "ilé", "báwo", "àwọn"}
WEAK_MARKERS_YO    = {"lọ", "jẹ", "ni", "sí"}

STRONG_MARKERS_IG  = {"nwoke", "nne", "nwanne", "ihe", "oba"}
WEAK_MARKERS_IG    = {"ga", "na", "so"}

STRONG_MARKERS_HA  = {"ina", "kai", "yau", "don"}
WEAK_MARKERS_HA    = {"ne", "da", "na"}

LANG_MARKERS = {
    "pcm": (STRONG_MARKERS_PCM, WEAK_MARKERS_PCM),
    "yo":  (STRONG_MARKERS_YO,  WEAK_MARKERS_YO),
    "ig":  (STRONG_MARKERS_IG,  WEAK_MARKERS_IG),
    "ha":  (STRONG_MARKERS_HA,  WEAK_MARKERS_HA),
}

_TOKEN_RE = re.compile(r"\b\w+'?\w*|\b\w+\b", re.UNICODE)


def _tokenize(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# -------------------------------------------------------------------
# Local FastText loader
# -------------------------------------------------------------------
_fasttext_model = None
_fasttext_lock = threading.Lock()
_use_local_fasttext = False


def _try_load_local_fasttext() -> None:
    global _fasttext_model, _use_local_fasttext
    if _fasttext_model is not None or not _FASTTEXT_PATH:
        return
    with _fasttext_lock:
        if _fasttext_model is not None:
            return
        try:
            import fasttext  # type: ignore
            if os.path.exists(_FASTTEXT_PATH):
                _fasttext_model = fasttext.load_model(_FASTTEXT_PATH)
                _use_local_fasttext = True
                logger.info(f"[language_detection] fastText model loaded from: {_FASTTEXT_PATH}")
            else:
                logger.warning(
                    f"[language_detection] FASTTEXT_MODEL_PATH set but file not found: {_FASTTEXT_PATH}"
                )
        except Exception as e:
            _use_local_fasttext = False
            _fasttext_model = None
            logger.exception(f"[language_detection] fastText failed to load: {e!r}")


# -------------------------------------------------------------------
# Remote FastText client
# -------------------------------------------------------------------
def _call_fasttext_service(text: str) -> Optional[Tuple[str, float]]:
    try:
        import requests  # type: ignore
        resp = requests.post(
            _FASTTEXT_SERVICE_URL, json={"text": text}, timeout=_FASTTEXT_SERVICE_TIMEOUT
        )
        if resp.status_code != 200:
            logger.warning(f"[language_detection] service error {resp.status_code}: {resp.text}")
            return None
        j = resp.json()
        lang = j.get("lang") or j.get("language") or "en"
        score = float(j.get("score", j.get("confidence", 0.0)))
        return lang, score
    except Exception as e:
        logger.exception(f"[language_detection] service call failed: {e!r}")
        return None


# -------------------------------------------------------------------
# Decision logic
# -------------------------------------------------------------------
def _normalize_lang_code(l: str) -> str:
    l = (l or "").strip().lower()
    if l in ("eng", "english"):
        return "en"
    if l in ("pidgin", "pcm_ng", "pcm-nigeria", "pcm"):
        return "pcm"
    return l


def decide_language_with_override(
    text: str,
    fasttext_lang: str,
    fasttext_conf: float,
    *,
    conf_override_threshold: float = _CONF_OVERRIDE_THRESHOLD,
    evidence_threshold: float = _EVIDENCE_THRESHOLD,
    strong_weight: float = _STRONG_WEIGHT,
    weak_weight: float = _WEAK_WEIGHT,
    strong_auto_override_threshold: float = _STRONG_AUTO_OVERRIDE_THRESHOLD,
) -> Dict:
    """
    Compute weighted evidence for local languages and decide whether to override.

    Returns a dict with keys:
      - final_language
      - override (bool)
      - reason (str)
      - meta (dict)
    """
    t = (text or "").strip()
    tokens = _tokenize(t)
    token_set = set(tokens)

    # compute evidence scores for each candidate local language
    scores: Dict[str, float] = {}
    per_lang_counts: Dict[str, Dict[str, int]] = {}
    for lang, (strong_markers, weak_markers) in LANG_MARKERS.items():
        # strong counts: allow substring matches for multi-word markers like "no be"
        strong_count = 0
        for m in strong_markers:
            if " " in m:
                if m in t.lower():
                    strong_count += 1
            else:
                if m in token_set:
                    strong_count += 1
        weak_count = sum(1 for m in weak_markers if m in token_set)
        per_lang_counts[lang] = {"strong": strong_count, "weak": weak_count}
        scores[lang] = strong_count * strong_weight + weak_count * weak_weight

    # default decision = fastText result
    final_lang = fasttext_lang
    override = False
    reason = "no_override"

    # best local candidate
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])

    # Auto override for very strong evidence regardless of fastText label/confidence
    if best_score >= strong_auto_override_threshold:
        final_lang = best_lang
        override = True
        reason = "strong_evidence_auto_override"
    else:
        # If fastText predicted English, use the earlier conservative logic
        if fasttext_lang in ("en", "eng"):
            if fasttext_conf < conf_override_threshold and best_score >= evidence_threshold:
                final_lang = best_lang
                override = True
                reason = f"conf_below_{conf_override_threshold}_and_evidence"
        else:
            # fastText predicted some non-English label. If its confidence is low,
            # allow evidence-based override to a local language.
            if fasttext_conf < _LOW_CONF_TRUST and best_score >= evidence_threshold:
                final_lang = best_lang
                override = True
                reason = f"low_conf_non_en_and_evidence"

    if override:
        logger.info(
            "[language_detection] override: text=%r fasttext=(%s,%.3f) -> %s (%s) evidence=%s counts=%s",
            t,
            fasttext_lang,
            fasttext_conf,
            final_lang,
            reason,
            scores,
            per_lang_counts,
        )

    return {
        "final_language": final_lang,
        "override": override,
        "reason": reason,
        "meta": {
            "tokens": len(tokens),
            "evidence_scores": scores,
            "per_lang_counts": per_lang_counts,
            "fasttext_lang": fasttext_lang,
            "fasttext_conf": fasttext_conf,
            "best_local_candidate": best_lang,
            "best_local_score": best_score,
        },
    }


# -------------------------------------------------------------------
# Public entrypoint
# -------------------------------------------------------------------
def detect_language(text: str) -> Tuple[str, float]:
    txt = (text or "").strip()
    if not txt:
        return "en", 0.5

    # helper to normalize labels
    def _norm(l: str) -> str:
        return _normalize_lang_code(l)

    # 1) Remote service
    if _FASTTEXT_SERVICE_URL:
        out = _call_fasttext_service(txt)
        if out:
            raw_lang, raw_conf = out
            try:
                lang = _norm(raw_lang)
                conf = float(raw_conf or 0.0)
            except Exception:
                lang = _norm(raw_lang)
                conf = float(raw_conf or 0.0)

            # If fastText predicts a non-English language with sufficient confidence, trust it
            if lang not in ("en", "eng") and conf >= _LOW_CONF_TRUST:
                return lang, conf

            # Otherwise (fastText says 'en' OR low-confidence non-en) decide whether to override
            decision = decide_language_with_override(txt, lang, conf)
            if decision["override"]:
                return decision["final_language"], max(conf, _OVERRIDE_CONFIDENCE)
            return lang, conf

    # 2) Local fastText
    if _FASTTEXT_PATH:
        _try_load_local_fasttext()
        if _use_local_fasttext and _fasttext_model is not None:
            try:
                labels, probs = _fasttext_model.predict(txt.replace("\n", " "), k=1)
                if labels and probs:
                    raw_label = labels[0]
                    prob = float(probs[0])
                    lang = _norm(raw_label.replace("__label__", ""))

                    # If fastText predicts non-English with enough confidence, trust it
                    if lang not in ("en", "eng") and prob >= _LOW_CONF_TRUST:
                        return lang, prob

                    # Otherwise (fastText says 'en' or non-en but low-conf) run override
                    decision = decide_language_with_override(txt, lang, prob)
                    if decision["override"]:
                        return decision["final_language"], max(prob, _OVERRIDE_CONFIDENCE)
                    return lang, prob
            except Exception as e:
                logger.exception(f"[language_detection] local predict failed: {e!r}")

    # 3) Default fallback
    logger.warning("[language_detection] fallback default -> en,0.6 for text=%r", txt)
    return "en", 0.6


# eager load if path present
if _FASTTEXT_PATH:
    try:
        _try_load_local_fasttext()
    except Exception:
        # loader logs its own exception; don't crash import
        pass
