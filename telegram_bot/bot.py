"""Telegram bot entrypoint for CRUDEMCX market signal templates.

The bot exposes a simple inline menu for Gold, Silver, BTC, and ETH. It can show
placeholder cards immediately and will also read the latest generated signal JSON
when the selected symbol matches that file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from templates import MENU_ORDER, SUPPORTED_MARKETS, default_market_message, market_button_text

LOGGER = logging.getLogger(__name__)
DEFAULT_SIGNAL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "smartapi_signal.json"


def build_market_keyboard() -> InlineKeyboardMarkup:
    """Build a 2x2 inline keyboard for the supported market templates."""
    rows = []
    for start in range(0, len(MENU_ORDER), 2):
        rows.append(
            [
                InlineKeyboardButton(market_button_text(key), callback_data=f"market:{key}")
                for key in MENU_ORDER[start : start + 2]
            ]
        )
    return InlineKeyboardMarkup(rows)


def load_signal(signal_path: Path, market_key: str) -> dict[str, Any] | None:
    """Load a generated signal if it matches the selected market symbol."""
    if not signal_path.exists():
        return None

    try:
        payload = json.loads(signal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Unable to read signal file %s: %s", signal_path, exc)
        return None

    symbol = str(payload.get("symbol", "")).upper()
    expected_symbol = SUPPORTED_MARKETS[market_key]["symbol"]
    if symbol != expected_symbol:
        return None
    return payload


def render_signal_message(market_key: str, signal: dict[str, Any] | None) -> str:
    """Render either a live signal card or the default market template."""
    if signal is None:
        return default_market_message(market_key)

    market = SUPPORTED_MARKETS[market_key]
    confidence = signal.get("confidence")
    confidence_text = f"{float(confidence) * 100:.1f}%" if isinstance(confidence, (int, float)) else "n/a"

    return (
        f"{market['emoji']} *{market['label']} ({market['symbol']})*\n"
        f"Signal: *{signal.get('signal', 'n/a')}*\n"
        f"Price: `{signal.get('price', 'n/a')}`\n"
        f"Target: `{signal.get('target', 'n/a')}`\n"
        f"Stop loss: `{signal.get('stop_loss', 'n/a')}`\n"
        f"Confidence: `{confidence_text}`\n"
        f"Updated: `{signal.get('timestamp', 'n/a')}`"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the initial market template menu."""
    message = (
        "Welcome to *CRUDEMCX Signal Bot*.\n\n"
        "Choose a market template to view signal details:"
    )
    await update.effective_message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_market_keyboard(),
    )


async def handle_market_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle market inline keyboard selections."""
    query = update.callback_query
    await query.answer()

    _, market_key = query.data.split(":", maxsplit=1)
    if market_key not in SUPPORTED_MARKETS:
        await query.edit_message_text("Unsupported market selected. Use /start to open the menu again.")
        return

    signal_path = Path(context.bot_data.get("signal_path", DEFAULT_SIGNAL_PATH))
    signal = load_signal(signal_path, market_key)
    await query.edit_message_text(
        render_signal_message(market_key, signal),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_market_keyboard(),
    )


def create_application(token: str, signal_path: Path = DEFAULT_SIGNAL_PATH) -> Application:
    """Create and configure the Telegram application."""
    application = Application.builder().token(token).build()
    application.bot_data["signal_path"] = signal_path
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", start))
    application.add_handler(CallbackQueryHandler(handle_market_selection, pattern=r"^market:"))
    return application


def main() -> None:
    """Run the bot with long polling."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the Telegram bot.")

    signal_path = Path(os.getenv("SIGNAL_PATH", DEFAULT_SIGNAL_PATH))
    create_application(token, signal_path).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
