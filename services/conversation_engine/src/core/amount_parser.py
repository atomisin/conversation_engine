import re

NUM_PAT = re.compile(r'(\d{1,3}(?:,\d{3})*|\d+)', re.I)

def parse_amount(text: str) -> int | None:
    if not text:
        return None
    nums = NUM_PAT.findall(text)
    if not nums:
        return None
    try:
        val = int(nums[0].replace(",", ""))
        if 50 <= val <= 10_000_000:
            return val
    except Exception:
        return None
    return None
