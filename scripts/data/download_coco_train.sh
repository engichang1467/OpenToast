#!/usr/bin/env bash
# Download + unzip COCO 2017 val images (~1GB, 5000 images).
# Usage: scripts/download_coco_train2017.sh [DEST_DIR]   (default: data/coco)
# bash scripts/data/download_coco_train.sh /data/coco2017
set -euo pipefail

URL="https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip"
DEST="${1:-data/coco}"
ZIP="$DEST/train2017.zip"
PART="$ZIP.part"


mkdir -p "$DEST"


if [ -d "$DEST/train2017" ]; then
   echo "already extracted: $DEST/train2017"
   exit 0
fi


echo "downloading $URL"
curl -fL \
   --retry 3 \
   --retry-all-errors \
   -C - \
   -o "$PART" \
   "$URL"


mv "$PART" "$ZIP"


echo "validating archive"
unzip -tq "$ZIP" >/dev/null


echo "extracting archive"
unzip -q -o "$ZIP" -d "$DEST"


rm -f "$ZIP"
echo "done: $DEST/train2017"
