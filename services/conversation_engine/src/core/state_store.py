from __future__ import annotations
from typing import Optional, Dict
from pydantic import BaseModel, Field
from .models import ConversationState, Turn
from datetime import datetime, timezone
timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryState:
    def __init__(self):
        self._db: Dict[str, ConversationState] = {}

    def get(self, conversation_id: str) -> ConversationState | None:
        return self._db.get(conversation_id)

    def upsert(self, state: ConversationState) -> None:
        self._db[state.conversation_id] = state

    def append_turn(self, conversation_id: str, turn: Turn) -> ConversationState:
        st = self._db.setdefault(conversation_id, ConversationState(conversation_id=conversation_id, user_id="unknown"))
        st.turns.append(turn)
        if turn.speaker == "buyer" and turn.parsed_offer:
            st.last_offer = turn.parsed_offer
        if turn.speaker == "bot" and turn.action in ("ACCEPT", "COUNTER") and turn.parsed_offer:
            st.last_counter = turn.parsed_offer
        return st

STATE = InMemoryState()
