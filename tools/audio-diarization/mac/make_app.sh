#!/bin/bash
# Builds ~/Applications/Who Said What.app and adds it to the Dock.
# Clicking the icon starts the web app in the background (no Terminal window) and opens the
# browser; clicking it again while it runs offers "Open" or "Stop it".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/Who Said What.app"
NO_DOCK="${1:-}"

echo "--> Building $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Who Said What</string>
  <key>CFBundleDisplayName</key><string>Who Said What</string>
  <key>CFBundleIdentifier</key><string>com.glitchg.whosaidwhat</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>WhoSaidWhat</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
PLIST

# The launcher. __ROOT__ is replaced with this checkout's path below.
cat > "$APP/Contents/MacOS/WhoSaidWhat" <<'LAUNCHER'
#!/bin/bash
ROOT="__ROOT__"
PORT=7860
URL="http://127.0.0.1:$PORT"
LOG="$HOME/Library/Logs/WhoSaidWhat.log"
PIDFILE="$HOME/Library/Application Support/WhoSaidWhat/server.pid"
mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"

running() { curl -s -o /dev/null --max-time 2 "$URL/"; }
notify() { osascript -e "display notification \"$1\" with title \"Who Said What\"" >/dev/null 2>&1; }

stop_server() {
  if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"; fi
  pkill -f "$ROOT/app.py" 2>/dev/null
}

if running; then
  choice=$(osascript -e 'button returned of (display dialog "Who Said What is running." buttons {"Stop it", "Open"} default button "Open" with title "Who Said What")' 2>/dev/null) || exit 0
  if [ "$choice" = "Stop it" ]; then
    stop_server
    notify "Stopped."
  else
    open "$URL"
  fi
  exit 0
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  osascript -e "display alert \"Who Said What is not installed\" message \"Run: bash '$ROOT/mac/Install.command'\"" >/dev/null
  exit 1
fi

cd "$ROOT" || exit 1
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "=== $(date) starting ===" >>"$LOG"
nohup "$ROOT/.venv/bin/python" "$ROOT/app.py" --port "$PORT" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
disown
notify "Starting… the page opens in a moment."

for _ in $(seq 1 180); do
  if running; then open "$URL"; exit 0; fi
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
  sleep 1
done
osascript -e "display alert \"Who Said What could not start\" message \"Opening the log file with the details.\"" >/dev/null
open "$LOG"
exit 1
LAUNCHER
# Escape characters that are special in a sed replacement (& | \).
ROOT_ESC=$(printf '%s' "$ROOT" | sed -e 's/[&|\\]/\\&/g')
sed -i '' -e "s|__ROOT__|$ROOT_ESC|" "$APP/Contents/MacOS/WhoSaidWhat" 2>/dev/null \
  || sed -i -e "s|__ROOT__|$ROOT_ESC|" "$APP/Contents/MacOS/WhoSaidWhat"
chmod +x "$APP/Contents/MacOS/WhoSaidWhat"

# Icon (skipped quietly if Pillow or iconutil are unavailable).
ICONSET="$(mktemp -d)/AppIcon.iconset"
if "$ROOT/.venv/bin/python" "$ROOT/mac/make_icon.py" "$ICONSET" 2>/dev/null \
  && command -v iconutil >/dev/null 2>&1; then
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns" || true
fi
touch "$APP"  # make Finder/Dock pick up the new icon

# Add to the Dock once.
if [ "$NO_DOCK" != "--no-dock" ] && command -v defaults >/dev/null 2>&1; then
  if ! defaults read com.apple.dock persistent-apps 2>/dev/null | grep -q "Who%20Said%20What.app\|Who Said What.app"; then
    defaults write com.apple.dock persistent-apps -array-add \
      "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>$APP</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>"
    killall Dock 2>/dev/null || true
    echo "--> Added 'Who Said What' to your Dock."
  fi
fi
echo "--> App ready: $APP"
