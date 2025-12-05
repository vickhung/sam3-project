#!/bin/bash
# Download COCO 2017 dataset for SAM3-Small training
# 
# Options:
#   --val-only    Download only val2017 (5K images, ~1GB) - for quick tests
#   --full        Download train2017 + val2017 (~19GB total)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/coco"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "========================================"
echo "COCO 2017 Dataset Download"
echo "========================================"
echo "Target directory: $DATA_DIR"
echo ""

# Parse arguments
VAL_ONLY=true
if [[ "$1" == "--full" ]]; then
    VAL_ONLY=false
fi

# Download annotations (always needed)
if [ ! -f "annotations/instances_val2017.json" ]; then
    echo "Downloading annotations..."
    wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    echo "✓ Annotations downloaded"
else
    echo "✓ Annotations already exist"
fi

# Download val2017
if [ ! -d "val2017" ]; then
    echo "Downloading val2017 (5K images, ~1GB)..."
    wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
    echo "✓ val2017 downloaded"
else
    echo "✓ val2017 already exists"
fi

# Download train2017 if requested
if [ "$VAL_ONLY" = false ]; then
    if [ ! -d "train2017" ]; then
        echo "Downloading train2017 (118K images, ~18GB)..."
        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip
        rm train2017.zip
        echo "✓ train2017 downloaded"
    else
        echo "✓ train2017 already exists"
    fi
fi

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo ""
echo "Dataset structure:"
ls -la "$DATA_DIR"
echo ""
echo "Image counts:"
echo "  val2017:   $(ls val2017/*.jpg 2>/dev/null | wc -l) images"
if [ -d "train2017" ]; then
    echo "  train2017: $(ls train2017/*.jpg 2>/dev/null | wc -l) images"
fi
echo ""
echo "Annotation files:"
ls -la annotations/*.json


