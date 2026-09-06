export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"

python -m sglang_router.launch_server \
    --served-model-name Qwen3.5-4B \
    --model-path Qwen3.5-4B \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --log-level WARNING \
    --router-log-level warn \
    --mem-fraction-static 0.9 \
    --dp-size 8 \
    --prefill-attention-backend fa3 \
    --decode-attention-backend flashinfer \
    --grpc-mode \
    --router-policy cache_aware \
    --router-health-check-interval-secs 60 \
    --router-prometheus-port 10001 \
    --host 0.0.0.0 \
    --port 8012

# The OpenAI-compatible endpoint is available at http://<host>:8012/v1.
