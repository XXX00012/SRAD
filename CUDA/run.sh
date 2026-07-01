#!/bin/bash
set -e

export XILINX_XRT=/usr

XCLBIN=${1:-svd.xclbin}
ITER=${2:-100}
INPUT_TXT=${3:-./data/input_image.txt}
OUTPUT_TXT=${4:-}
BLOCKS=${5:-}

echo "[run.sh] XCLBIN     = $XCLBIN"
echo "[run.sh] ITER       = $ITER"
echo "[run.sh] INPUT_TXT  = $INPUT_TXT"
echo "[run.sh] OUTPUT_TXT = ${OUTPUT_TXT:-<skipped>}"
echo "[run.sh] BLOCKS     = ${BLOCKS:-<all>}"

if [ -n "$BLOCKS" ]; then
  ./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT" "${OUTPUT_TXT:-"-"}" "$BLOCKS"
elif [ -n "$OUTPUT_TXT" ]; then
  ./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT" "$OUTPUT_TXT"
else
  ./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT"
fi
