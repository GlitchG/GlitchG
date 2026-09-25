#!/bin/bash
# One-time setup on macOS. Double-click in Finder (or run in Terminal).
# Installs everything into tools/audio-diarization/.venv; nothing system-wide except `uv`.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "=== Who Said What: installing into $ROOT/.venv ==="

if ! command -v uv >/dev/null 2>&1; then
  echo "--> Installing uv (fast Python installer)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "--> Creating Python 3.12 environment…"
uv venv --python 3.12 --allow-existing .venv

echo "--> Installing packages (a few minutes)…"
uv pip install --python .venv/bin/python -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "--> Created .env (add ANTHROPIC_API_KEY for auto roles, TELEGRAM_BOT_TOKEN for the bot)."
fi

echo "--> Downloading the models (~2.5 GB, one time)…"
.venv/bin/python -c "from diarize_transcribe.pipeline import load_asr, load_diarizer; from diarize_transcribe.speakers import load_speaker_model; load_diarizer(); load_asr(); load_speaker_model()"

# `whosaid` command for Terminal.
ALIAS_LINE="alias whosaid='\"$ROOT/mac/whosaid\"'"
if ! grep -qF "$ALIAS_LINE" "$HOME/.zshrc" 2>/dev/null; then
  printf '\n# Who Said What (audio -> transcript with speaker roles)\n%s\n' "$ALIAS_LINE" >> "$HOME/.zshrc"
  echo "--> Added the 'whosaid' command to ~/.zshrc (open a new Terminal window to use it)."
fi

echo
echo "✅ Done! To open the app, run:"
echo "   bash \"$ROOT/mac/Who Said What.command\""
read -r -p "Press Enter to close…"
