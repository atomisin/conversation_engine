import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core.language_detection import detect_language

os.environ.setdefault("FASTTEXT_MODEL_PATH", "services/conversation_engine/models/lid.176.ftz")

examples = [
  "I fit pay 5000",
  "Abeg, how much is this?",
  "Ẹ káàbọ̀, mélòó ni?",
  "How much for this?",
  "Nwanne, biko give me price"
]
for t in examples:
    print(t, "->", detect_language(t))