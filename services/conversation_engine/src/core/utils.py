import re, unicodedata

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.strip().split())

N_NUM = re.compile(r'(?<!\d)(?:₦\s*)?(\d{1,3}(?:,\d{3})*|\d+)(?!\d)')

def contains_required_price(gen_text: str, required_price: int) -> bool:
    found = [int(m.group(1).replace(",", "")) for m in N_NUM.finditer(gen_text or "")]
    return required_price in found
