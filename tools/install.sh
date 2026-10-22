#!/bin/sh
# Copy the mod into the Isaac mods folder.  Run from the repo root:
#   sh tools/install.sh
# Override the mods folder with ISAAC_MODS=/path/to/mods if your game is not
# in the default Steam location on Windows.
set -e
MODS="${ISAAC_MODS:-/c/Program Files (x86)/Steam/steamapps/common/The Binding of Isaac Rebirth/mods}"
DEST="$MODS/fly-brain-connectome"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r mod/* "$DEST/"
echo "installed to $DEST"
du -sh "$DEST"
