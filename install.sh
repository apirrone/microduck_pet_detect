#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO="${PET_DETECT_REPO:-apirrone/microduck_pet_detect}"
BIN_DIR="/usr/local/bin"
MODEL_DIR="/opt/microduck"
MODEL_NAME="pet_detect.onnx"

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Microduck Pet Detect Installer       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""

ARCH=$(uname -m)
case $ARCH in
    aarch64|arm64) ;;
    *)
        echo -e "${RED}Error: unsupported architecture: $ARCH (need aarch64)${NC}"
        exit 1
        ;;
esac

# Resolve the latest release tarball URL.
echo "Fetching latest release info from $REPO..."
TARBALL_URL=$(curl -sSfL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -oE '"browser_download_url": *"[^"]*aarch64-linux\.tar\.gz"' \
    | head -n1 \
    | sed -E 's/.*"([^"]+)"$/\1/')

if [ -z "$TARBALL_URL" ]; then
    echo -e "${RED}Could not resolve a release tarball — is there a published release?${NC}"
    exit 1
fi
echo -e "  ${GREEN}→${NC} $TARBALL_URL"

TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "Downloading..."
curl -sSfL --retry 3 -o "$TMP_DIR/pet.tgz" "$TARBALL_URL"
tar xzf "$TMP_DIR/pet.tgz" -C "$TMP_DIR"

# Install binaries to /usr/local/bin
echo "Installing binaries to $BIN_DIR..."
for b in pet_detect pet_features; do
    if [ -f "$TMP_DIR/$b" ]; then
        sudo install -m 0755 "$TMP_DIR/$b" "$BIN_DIR/$b"
        echo -e "  ${GREEN}✓${NC} $BIN_DIR/$b"
    fi
done

# Install the ONNX model to /opt/microduck (shared location across microduck tools)
if [ -f "$TMP_DIR/$MODEL_NAME" ]; then
    sudo mkdir -p "$MODEL_DIR"
    sudo install -m 0644 "$TMP_DIR/$MODEL_NAME" "$MODEL_DIR/$MODEL_NAME"
    echo -e "  ${GREEN}✓${NC} $MODEL_DIR/$MODEL_NAME"
else
    echo -e "  ${YELLOW}Warning: $MODEL_NAME not in tarball — model not installed${NC}"
fi

# libonnxruntime: only install if the system doesn't already have it
# (microduck_runtime ships its own copy in /usr/local/lib).
if [ ! -f /usr/local/lib/libonnxruntime.so ] && [ ! -f /usr/lib/aarch64-linux-gnu/libonnxruntime.so ]; then
    if ls "$TMP_DIR"/libonnxruntime.so* >/dev/null 2>&1; then
        echo "Installing libonnxruntime to /usr/local/lib..."
        sudo install -m 0755 "$TMP_DIR"/libonnxruntime.so* /usr/local/lib/
        sudo ldconfig
        echo -e "  ${GREEN}✓${NC} libonnxruntime installed"
    fi
else
    echo -e "  ${GREEN}✓${NC} libonnxruntime already present, skipping"
fi

echo ""
if command -v pet_detect &>/dev/null; then
    echo -e "${GREEN}✓ Installation successful!${NC}"
    echo ""
    echo "Quick test:"
    echo "  arecord -D plughw:aic3104,0 -f S16_LE -r 16000 -c 1 -t raw \\"
    echo "    | pet_detect --model $MODEL_DIR/$MODEL_NAME"
else
    echo -e "${RED}Error: pet_detect not found in PATH after install${NC}"
    exit 1
fi
