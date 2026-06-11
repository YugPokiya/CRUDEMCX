"""Shared Telegram bot menu templates for supported markets."""

SUPPORTED_MARKETS = {
    "GOLD": {
        "label": "Gold",
        "emoji": "🥇",
        "symbol": "GOLD",
        "asset_class": "Commodity",
    },
    "SILVER": {
        "label": "Silver",
        "emoji": "🥈",
        "symbol": "SILVER",
        "asset_class": "Commodity",
    },
    "BTC": {
        "label": "Bitcoin",
        "emoji": "₿",
        "symbol": "BTC",
        "asset_class": "Crypto",
    },
    "ETH": {
        "label": "Ethereum",
        "emoji": "Ξ",
        "symbol": "ETH",
        "asset_class": "Crypto",
    },
}

MENU_ORDER = ("GOLD", "SILVER", "BTC", "ETH")


def market_button_text(market_key: str) -> str:
    """Return display text for a Telegram inline keyboard market button."""
    market = SUPPORTED_MARKETS[market_key]
    return f"{market['emoji']} {market['label']}"


def default_market_message(market_key: str) -> str:
    """Return a fallback signal card when no generated signal is available."""
    market = SUPPORTED_MARKETS[market_key]
    return (
        f"{market['emoji']} *{market['label']} ({market['symbol']})*\n"
        f"Asset class: {market['asset_class']}\n\n"
        "Signal template is ready. Connect your ML/API feed to publish live entry, "
        "target, stop-loss, and confidence values here."
    )
