from .models import Product, MerchantConfig, DecisionOutput

def enforce_price_bounds(product: Product, merchant: MerchantConfig, decision: DecisionOutput) -> DecisionOutput:
    min_price = product.min_price or int(product.base_price * merchant.min_discount_ratio)
    if decision.price is not None and decision.price < min_price:
        return DecisionOutput(
            action="REJECT",
            price=None,
            confidence=decision.confidence,
            strategy_tag="below_min_price",
            meta={**decision.meta, "enforced_min_price": min_price}
        )
    return decision

def requires_approval(product: Product, merchant: MerchantConfig, decision: DecisionOutput) -> bool:
    if decision.action == "ACCEPT" and decision.price is not None:
        ratio = decision.price / max(product.base_price, 1)
        return ratio < merchant.require_approval_below_ratio
    return False
