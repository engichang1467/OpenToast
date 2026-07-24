#!/usr/bin/env bash
# Download + unzip COCO 2017 val images (~1GB, 5000 images).
# Usage: scripts/data/download_coco_val2017.sh [DEST_DIR]   (default: data/coco)
set -euo pipefail

URL="http://images.cocodataset.org/zips/val2017.zip"
DEST="${1:-data/coco}"
ZIP="$DEST/val2017.zip"

mkdir -p "$DEST"

if [ -d "$DEST/val2017" ]; then
    echo "already extracted: $DEST/val2017"
    exit 0
fi

echo "downloading $URL"
curl -fL --retry 3 -C - -o "$ZIP" "$URL"

echo "unzipping"
unzip -q -o "$ZIP" -d "$DEST"
rm -f "$ZIP"
echo "done: $DEST/val2017"
