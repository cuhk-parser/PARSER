export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"

python -m sglang_router.launch_server \
    --served-model-name Qwen3.5-4B \
    --model-path ../Qwen3.5-4B \
    --reasoning-parser qwen3 \
    --context-length 5120 \
    --dtype bfloat16 \
    --log-level INFO \
    --router-log-level info \
    --mem-fraction-static 0.85 \
    --dp-size 2 \
    --attention-backend flashinfer \
    --grpc-mode \
    --router-policy cache_aware \
    --router-health-check-interval-secs 60 \
    --router-prometheus-port 10002 \
    --port 8006
