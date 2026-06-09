#!/bin/bash
set -e

export XILINX_XRT=/usr

XCLBIN=${1:-svd.xclbin}
ITER=${2:-100}
INPUT_TXT=${3:-./data/plio_ours_j.txt}

if [ ! -f "$XCLBIN" ] && [ -f "a.xclbin" ]; then
  XCLBIN="a.xclbin"
fi

echo "[run.sh] XCLBIN     = $XCLBIN"
echo "[run.sh] ITER       = $ITER"
echo "[run.sh] INPUT_TXT  = $INPUT_TXT"

./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT"
