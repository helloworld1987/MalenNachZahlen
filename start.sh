#!/usr/bin/env bash
# Malen nach Zahlen Studio Launcher

set -e

PORT=8080
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/web" && pwd)"

# Check if port is in use, increment if needed
while ss -tuln | grep -q ":$PORT "; do
  PORT=$((PORT + 1))
done

URL="http://localhost:$PORT"

echo "=========================================="
echo "🎨 Malen nach Zahlen Studio wird gestartet"
echo "=========================================="
echo "Web-App URL: $URL"
echo "Drücke Strg+C zum Beenden."
echo ""

# Try opening in browser
if command -v xdg-open >/dev/null 2>&1; then
  (sleep 1 && xdg-open "$URL") >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  (sleep 1 && open "$URL") >/dev/null 2>&1 &
fi

python3 -m http.server "$PORT" --directory "$DIR"
