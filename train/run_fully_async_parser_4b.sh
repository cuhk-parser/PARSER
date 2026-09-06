set -x
export WANDB_MODE=offline
export CUDA_HOME="/usr/local/cuda"

ROOT_DIR=${ROOT_DIR:-$(pwd)}
DATA_ROOT=${DATA_ROOT:-$(dirname "$ROOT_DIR")/data/train}

train_file=${TRAIN_FILE:-$DATA_ROOT/hotpotqa_train_process_emb-select_llm-unable.parquet}
model_path=${MODEL_PATH:-$(dirname "$ROOT_DIR")/Qwen3.5-4B}

project_name=${PROJECT_NAME:-parser_verl}
experiment_name=${EXPERIMENT_NAME:-parser_4B_$(date +'%m%d-%H%M')}
default_local_dir=${DEFAULT_LOCAL_DIR:-$ROOT_DIR/checkpoint/$experiment_name}

export LONGMAS_SGLANG_UPSTREAMS="${LONGMAS_SGLANG_UPSTREAMS:-127.0.0.1:8012}"
export LONGMAS_NGINX_KEEPALIVE="${LONGMAS_NGINX_KEEPALIVE:-6400}"

max_prompt_length=${MAX_PROMPT_LENGTH:-840}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}

chunk_max_length=${CHUNK_MAX_LENGTH:-512}
max_chunk_num=${MAX_CHUNK_NUM:-null}

TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}
infer_tp=${INFER_TP:-1}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}

actor_lr=${ACTOR_LR:-1e-6}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.2}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
n_gpus_rollout=${N_GPUS_ROLLOUT:-4}
n_gpus_training=${N_GPUS_TRAINING:-$((NGPUS_PER_NODE - n_gpus_rollout))}

ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-5}
staleness_threshold=${STALENESS_THRESHOLD:-2}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-4}
require_batches=${REQUIRE_BATCHES:-1}
partial_rollout=${PARTIAL_ROLLOUT:-True}

total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-9999999}
total_epochs=${TOTAL_EPOCHS:-25}

agent_loop_config_path=${AGENT_LOOP_CONFIG_PATH:-scripts/parser_agent_loop.yaml}

DATA=(
    data.train_files="['$train_file']"
    data.val_files="['$train_file']"
    data.train_max_samples=32768
    data.train_batch_size=0
    data.gen_batch_size=1
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=False
    data.truncation=error
    data.custom_cls.path=scripts/parser_dataset.py
    data.custom_cls.name=ParserRLHFDataset
    +data.chunk_max_length=$chunk_max_length
    +data.max_chunk_num=$max_chunk_num
)

MODEL=(
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.path=$model_path
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high
    actor_rollout_ref.actor.optim.lr=$actor_lr
    actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=true
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu
    actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.optim.lr_warmup_steps=70
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    rollout.total_rollout_steps=$total_rollout_steps
    trainer.total_epochs=$total_epochs
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.n=$n_resp_per_prompt
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.agent.default_agent_loop=parser_lead_agent
    actor_rollout_ref.rollout.agent.num_workers=$n_gpus_rollout
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path
    actor_rollout_ref.rollout.multi_turn.enable=True
    rollout.nnodes=$NNODES
    rollout.n_gpus_per_node=$n_gpus_rollout
    actor_rollout_ref.hybrid_engine=False
    async_training.staleness_threshold=$staleness_threshold
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step
    async_training.require_batches=$require_batches
    async_training.partial_rollout=$partial_rollout
    async_training.dynamic_sampling.enable=True
    async_training.use_trainer_do_validate=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.rollout_correction.bypass_mode=False
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

TRAINER=(
    trainer.logger="['console', 'wandb']"
    trainer.project_name=$project_name
    trainer.experiment_name=$experiment_name
    trainer.val_before_train=False
    trainer.log_val_generations=0
    trainer.save_freq=10
    trainer.test_freq=-1
    trainer.default_local_dir=$default_local_dir
    trainer.nnodes=$NNODES
    ray_kwargs.ray_init.num_cpus=$((12 * NGPUS_PER_NODE))
    trainer.n_gpus_per_node=$n_gpus_training
    trainer.max_actor_ckpt_to_keep=4
    trainer.default_hdfs_dir=null
)

split_csv() {
    local raw="$1"
    local -n out_ref="$2"
    local item
    IFS=',' read -ra out_ref <<< "$raw"
    for i in "${!out_ref[@]}"; do
        item="${out_ref[$i]}"
        item="${item#"${item%%[![:space:]]*}"}"
        item="${item%"${item##*[![:space:]]}"}"
        out_ref[$i]="$item"
    done
}

mkdir -p "$default_local_dir"

split_csv "$LONGMAS_SGLANG_UPSTREAMS" LONGMAS_SGLANG_UPSTREAM_LIST
if [[ -n "${LONGMAS_NGINX_SOCKET:-}" ]]; then
    split_csv "$LONGMAS_NGINX_SOCKET" LONGMAS_NGINX_SOCKET_LIST
else
    LONGMAS_NGINX_SOCKET_LIST=()
    for i in "${!LONGMAS_SGLANG_UPSTREAM_LIST[@]}"; do
        LONGMAS_NGINX_SOCKET_LIST+=("/tmp/longmas_nginx_${SLURM_JOB_ID:-$$}_${i}.sock")
    done
fi

if [[ "${#LONGMAS_NGINX_SOCKET_LIST[@]}" -ne "${#LONGMAS_SGLANG_UPSTREAM_LIST[@]}" ]]; then
    echo "LONGMAS_NGINX_SOCKET count (${#LONGMAS_NGINX_SOCKET_LIST[@]}) must match LONGMAS_SGLANG_UPSTREAMS count (${#LONGMAS_SGLANG_UPSTREAM_LIST[@]})." >&2
    exit 1
fi

LONGMAS_NGINX_SOCKET="$(IFS=,; echo "${LONGMAS_NGINX_SOCKET_LIST[*]}")"
export LONGMAS_NGINX_SOCKET

stop_nginx_proxies() {
    local i
    for i in "${!LONGMAS_NGINX_SOCKET_LIST[@]}"; do
        LONGMAS_NGINX_SOCKET="${LONGMAS_NGINX_SOCKET_LIST[$i]}" \
        LONGMAS_SGLANG_UPSTREAMS="${LONGMAS_SGLANG_UPSTREAM_LIST[$i]}" \
        LONGMAS_NGINX_CONF="$default_local_dir/longmas_sglang_nginx_${i}.conf" \
        LONGMAS_NGINX_PID="$default_local_dir/longmas_sglang_nginx_${i}.pid" \
        LONGMAS_NGINX_ERROR_LOG="$default_local_dir/longmas_sglang_nginx_${i}_error.log" \
            bash scripts/start_sglang_nginx_proxy.sh stop || true
    done
}

start_nginx_proxies() {
    local i
    for i in "${!LONGMAS_SGLANG_UPSTREAM_LIST[@]}"; do
        if ! LONGMAS_NGINX_SOCKET="${LONGMAS_NGINX_SOCKET_LIST[$i]}" \
            LONGMAS_SGLANG_UPSTREAMS="${LONGMAS_SGLANG_UPSTREAM_LIST[$i]}" \
            LONGMAS_NGINX_CONF="$default_local_dir/longmas_sglang_nginx_${i}.conf" \
            LONGMAS_NGINX_PID="$default_local_dir/longmas_sglang_nginx_${i}.pid" \
            LONGMAS_NGINX_ERROR_LOG="$default_local_dir/longmas_sglang_nginx_${i}_error.log" \
                bash scripts/start_sglang_nginx_proxy.sh restart; then
            echo "Failed to start local nginx proxy for upstream ${LONGMAS_SGLANG_UPSTREAM_LIST[$i]}." >&2
            return 1
        fi
    done
}

if ! start_nginx_proxies; then
    echo "Failed to start local nginx proxies. Check nginx availability, socket path length, and upstream config." >&2
    stop_nginx_proxies
    exit 1
fi
trap 'stop_nginx_proxies' EXIT

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"