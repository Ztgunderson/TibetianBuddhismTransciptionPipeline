#!/usr/bin/env bash
# Engine A — build whisper.cpp with CUDA on the Jetson Orin and produce the FP16/Q8_0/Q4_0
# ladder for whisper-large-v3 and whisper-small.
#
# RUN THIS INSIDE A JETSON CONTAINER (the host Python torch is CPU-only; the GPU is only
# reachable from a CUDA-enabled container). Example:
#
#   cd /home/jetson/jetson-containers
#   ./run.sh --volume $PWD/../Documents/TibetianBuddhismTransciptionPipeline:/work \
#            $(./autotag cuda) bash -c "cd /work && bash models/build.sh"
#
# Produces:
#   models/ggml/large-v3-{fp16,q8_0,q4_0}.bin
#   models/ggml/small-{fp16,q8_0,q4_0}.bin
#   models/whisper.cpp/                (the built repo, incl. whisper-cli + quantize)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WCPP="$ROOT/models/whisper.cpp"
OUT="$ROOT/models/ggml"
mkdir -p "$OUT"

# 1. Fetch + build whisper.cpp with the CUDA backend.
if [ ! -d "$WCPP" ]; then
  git clone https://github.com/ggml-org/whisper.cpp "$WCPP"
fi
cmake -S "$WCPP" -B "$WCPP/build" -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release
cmake --build "$WCPP/build" -j --config Release

CLI="$WCPP/build/bin/whisper-cli"
QUANT="$WCPP/build/bin/quantize"

# 2. Download the FP16 ggml base models (multilingual — required for Tibetan/Sanskrit).
#    q5_0 intentionally omitted: the study fixes a clean three-level ladder (see design_spec).
for m in large-v3 small; do
  bash "$WCPP/models/download-ggml-model.sh" "$m"
  cp "$WCPP/models/ggml-$m.bin" "$OUT/$m-fp16.bin"
done

# 3. Quantize each model to Q8_0 and Q4_0.
for m in large-v3 small; do
  "$QUANT" "$OUT/$m-fp16.bin" "$OUT/$m-q8_0.bin" q8_0
  "$QUANT" "$OUT/$m-fp16.bin" "$OUT/$m-q4_0.bin" q4_0
done

echo "built variants:"
ls -lh "$OUT"
echo
echo "Set WHISPER_CLI=$CLI when running bench/runner.py, or pass bin_path in engine_cfg."
