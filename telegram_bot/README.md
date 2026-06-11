# CRUDEMCX Telegram Bot

Telegram bot template for CRUDEMCX market signals. The bot starts with a 2x2 inline menu for:

- Gold
- Silver
- BTC
- ETH

Each option opens a reusable signal card. If `SIGNAL_PATH` points to a JSON file whose `symbol` matches the selected market, the bot renders price, target, stop-loss, confidence, and timestamp from that file. Otherwise it shows a ready-to-connect template message.

## Setup

1. Create a bot with [BotFather](https://t.me/BotFather) and copy the token.
2. Install dependencies:
   ```bash
   cd telegram_bot
   python -m pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```bash
   cp .env.example .env
   export TELEGRAM_BOT_TOKEN=<your-token>
   export SIGNAL_PATH=../outputs/smartapi_signal.json
   ```
4. Start the bot:
   ```bash
   python bot.py
   ```

## Commands

- `/start` - open the market template menu.
- `/menu` - reopen the market template menu.

## Signal JSON shape

```json
{
  "symbol": "GOLD",
  "signal": "BUY",
  "confidence": 0.72,
  "price": 2350.5,
  "target": 2375.0,
  "stop_loss": 2330.0,
  "timestamp": "2026-06-11T12:00:00+00:00"
}
```
