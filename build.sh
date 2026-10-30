#!/usr/bin/env bash
# Packages the module for the registry (meta.json build). The exec bits are set only in the staged
# copy: an executable entrypoint in the source tree makes the cloud builder skip this step.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf dist/pkg && mkdir -p dist/pkg/src
install -m 644 meta.json requirements.txt dist/pkg/
install -m 755 run.sh setup.sh dist/pkg/
install -m 644 src/*.py dist/pkg/src/
# COPYFILE_DISABLE: macOS tar would otherwise add AppleDouble ._* entries.
COPYFILE_DISABLE=1 tar -czf module.tar.gz -C dist/pkg .
