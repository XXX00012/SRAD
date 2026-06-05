#!/bin/bash
set -e

export XILINX_XRT=/usr

XCLBIN=${1:-svd.xclbin}
ITER=${2:-100}
INPUT_TXT=${3:-}

if [ ! -f "$XCLBIN" ] && [ -f "a.xclbin" ]; then
  XCLBIN="a.xclbin"
fi

echo "[run.sh] XCLBIN     = $XCLBIN"
echo "[run.sh] ITER       = $ITER"
if [ -n "$INPUT_TXT" ]; then
  echo "[run.sh] INPUT_TXT  = $INPUT_TXT"
else
  echo "[run.sh] INPUT_TXT  = generated"
fi

if [ -n "$INPUT_TXT" ]; then
  ./host.exe "$XCLBIN" "$ITER" "$INPUT_TXT"
else
  ./host.exe "$XCLBIN" "$ITER"
fi
