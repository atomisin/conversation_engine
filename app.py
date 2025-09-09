# app.py
import os, sys
import logging
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.language_detection import detect_language

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fasttext-service")

app = FastAPI(title="FastText Language Detector (wrapped)")

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    """
    Report simple service health. model_loaded is True if either:
      - a remote FASTTEXT_SERVICE_URL is configured, OR
      - the local fastText loader in language_detection succeeded.
    This function checks a few likely module import names (core.language_detection
    and language_detection) and will try to trigger the module's loader if available.
    """
    # If a remote service is configured, consider model "available"
    if bool(os.getenv("FASTTEXT_SERVICE_URL")):
        return {"status": "ok", "model_loaded": True}

    import importlib

    model_loaded = False
    checked_modules = []

    # Check these candidate module names (order matters)
    candidates = ["core.language_detection", "language_detection"]

    for name in candidates:
        try:
            ld = importlib.import_module(name)
            checked_modules.append(name)
            # if module exposes the lazy loader, call it (idempotent)
            try:
                if hasattr(ld, "_try_load_local_fasttext"):
                    ld._try_load_local_fasttext()
            except Exception:
                # keep going — loader may log its own errors
                pass

            if getattr(ld, "_use_local_fasttext", False):
                model_loaded = True
                break
        except ModuleNotFoundError:
            # module not present under this name — try next
            continue
        except Exception:
            # unexpected error importing; keep going but log for debugging
            logger.exception("health: unexpected error importing %s", name)

    # If nothing found, model_loaded remains False
    return {"status": "ok", "model_loaded": bool(model_loaded)}


@app.post("/detect")
def detect_endpoint(req: TextRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        lang, confidence = detect_language(text)  # delegate to language_detection.py
        # ensure native Python types
        confidence = float(confidence or 0.0)
        return {"language": lang, "confidence": confidence}
    except HTTPException:
        # re-raise FastAPI HTTPExceptions unmodified
        raise
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"prediction error: {str(e)}")

# Generic exception handler to ensure JSON responses on uncaught exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception for request {request.url}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)}
    )
