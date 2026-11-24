#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_commit=d88040c704c8855c497d46c20675a3d492f70f94
task_source="${LLAMA_CPP_DIR:-$task_root/vendor/llama.cpp}"
if [[ ! -d "$task_source/.git" ]]; then
  mkdir -p "$(dirname -- "$task_source")"
  git clone https://github.com/ggml-org/llama.cpp.git "$task_source"
fi
if [[ "$(git -C "$task_source" rev-parse HEAD)" != "$task_commit" ]]; then
  git -C "$task_source" fetch origin "$task_commit"
  git -C "$task_source" checkout --detach "$task_commit"
fi
cmake -S "$task_source" -B "$task_source/build-cpu" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$task_source/build-cpu" --config Release -j "${BUILD_JOBS:-4}" \
  --target llama-completion llama-bench llama-quantize
