import json
from pathlib import Path
from .models import Product

_CATALOG = {}

def load_catalog(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        print(f"Product catalog missing at {p}")
        return
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    global _CATALOG
    _CATALOG = {item["id"]: Product(**item) for item in data}

def get_product(product_id: str) -> Product | None:
    return _CATALOG.get(product_id)

def all_products() -> list[Product]:
    return list(_CATALOG.values())
