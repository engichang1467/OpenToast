#!/usr/bin/env bash
# Download + unzip ADE20K (ADEChallengeData2016, ~1GB): images + seg annotations,
# train + val. This is the ADE20K used by UPerNet (README downstream/ade20k).
# Usage: scripts/download_ade20k.sh [DEST_DIR]   (default: data/ade20k)
set -euo pipefail

URL="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
DEST="${1:-data/ade20k}"
ZIP="$DEST/ADEChallengeData2016.zip"

mkdir -p "$DEST"

if [ -d "$DEST/ADEChallengeData2016" ]; then
    echo "already extracted: $DEST/ADEChallengeData2016"
    exit 0
fi

echo "downloading $URL"
curl -fL --retry 3 -C - -o "$ZIP" "$URL"

echo "unzipping"
unzip -q -o "$ZIP" -d "$DEST"
rm -f "$ZIP"
echo "done: $DEST/ADEChallengeData2016"
