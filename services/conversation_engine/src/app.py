import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional
from core.product_catalog import load_catalog
from core.conversation_engine import handle_message

app = FastAPI(title="YarnMarket-AI Conversation Engine")

@app.on_event("startup")
def _startup():
    # compute repo-local path to the data folder (relative to this app.py)
    base_dir = Path(__file__).resolve().parents[1]   # services/conversation-engine
    catalog_path = base_dir / "data" / "product_catalog.json"
    try:
        load_catalog(catalog_path)
    except Exception as e:
        print("Warning: failed to load product catalog:", e)

class InboundPayload(BaseModel):
    conversation_id: str
    user_id: str
    text: str
    product_id: Optional[str] = None

@app.post("/webhook")
def webhook(payload: InboundPayload):
    out = handle_message(
        conversation_id=payload.conversation_id,
        user_id=payload.user_id,
        text=payload.text,
        product_id=payload.product_id
    )
    return out.dict()
