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

# Scale SGLang workers without restarting the router.
# List registered workers:
# curl http://127.0.0.1:8012/workers
# Register a new SGLang worker (the worker must already be serving with --grpc-mode):
# curl -X POST http://127.0.0.1:8012/workers -H "Content-Type: application/json" -d '{"url": "grpc://192.168.0.2:8009"}'
# Remove a worker by the id returned from GET /workers:
# curl -X DELETE http://127.0.0.1:8012/workers/2fcaad6f-ebbc-4423-8426-7c0b8e02840f
