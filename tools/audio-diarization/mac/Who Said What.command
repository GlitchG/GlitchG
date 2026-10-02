#!/bin/bash
# Opens the web app in your browser. Keep this window open while you use it; close it to stop.
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet: double-click mac/Install.command first."
  read -r -p "Press Enter to close…"
  exit 1
fi
if [ -f .env ]; then set -a; source .env; set +a; fi

echo "Starting Who Said What at http://127.0.0.1:7860 …"
echo "(Close this window to stop the app.)"
exec .venv/bin/python app.py --inbrowser
