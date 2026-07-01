#!/bin/bash
set -e

export XILINX_XRT=/usr

XCLBIN=${1:-svd.xclbin}
ITER=${2:-100}
INPUT_TXT=${3:-./data/input_image.txt}
OUTPUT_TXT=${4:-}

echo "[run.sh] XCLBIN     = $XCLBIN"
echo "[run.sh] ITER       = $ITER"
echo "[run.sh] INPUT_TXT  = $INPUT_TXT"
echo "[run.sh] OUTPUT_TXT = ${OUTPUT_TXT:-<skipped>}"

if [ -n "$OUTPUT_TXT" ]; then
  ./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT" "$OUTPUT_TXT"
else
  ./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT"
fi
