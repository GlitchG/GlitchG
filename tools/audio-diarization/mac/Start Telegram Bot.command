#!/bin/bash
# Runs the Telegram bot on this Mac. The Mac is kept awake while this window is open;
# close the window to stop the bot.
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet: double-click mac/Install.command first."
  read -r -p "Press Enter to close…"
  exit 1
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
  echo "Add TELEGRAM_BOT_TOKEN=... to $(pwd)/.env first (opening it now)."
  open -e .env
  read -r -p "Press Enter to close…"
  exit 1
fi

# caffeinate -i: prevent idle sleep while the bot runs (the lid must stay open on a laptop).
exec caffeinate -i .venv/bin/python bot.py
