#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
torch_image=${TORCH_RESEARCH_IMAGE:-wzsp-demucs-e830-ready:latest}

exec docker run --rm \
  -v "$project_dir:/work" \
  -w /work \
  -e PYTHONPATH=/work/src \
  "$torch_image" \
  "$@"

