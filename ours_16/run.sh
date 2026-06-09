#!/bin/bash
set -e

export XILINX_XRT=/usr

XCLBIN=${1:-svd.xclbin}
ITER=${2:-3}
INPUT_TXT=${3:-./data/plio_ours_j.txt}
OUTPUT_TXT=${4:-./data/aie_j_next.txt}

if [ ! -f "$XCLBIN" ] && [ -f "a.xclbin" ]; then
  XCLBIN="a.xclbin"
fi

echo "[run.sh] XCLBIN     = $XCLBIN"
echo "[run.sh] ITER       = $ITER"
echo "[run.sh] INPUT_TXT  = $INPUT_TXT"
echo "[run.sh] OUTPUT_TXT = $OUTPUT_TXT"

./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT" "$OUTPUT_TXT"
