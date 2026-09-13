# PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents

This repository is the official implementation of
[**PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents**](https://arxiv.org/abs/2609.06702).

[[Project Page](https://cuhk-parser.github.io/)]
[[Paper](https://arxiv.org/abs/2609.06702)]
[[Model](https://huggingface.co/inNexus/parser-qwen35-4b)]
[[Dataset](https://huggingface.co/datasets/inNexus/parser_dataset)]

![Overview of PARSER](workflow.png)

## Released model

We release the trained 4B-parameter PARSER lead agent on Hugging Face:
[parser-4b](https://huggingface.co/inNexus/parser-qwen35-4b).

```bash
hf download inNexus/parser-qwen35-4b --local-dir parser-4b
```

The downloaded checkpoint can be supplied as `MODEL_PATH` during evaluation.

## Repository layout

```text
.
├── chat_template.jinja   # Qwen3.5 lead-agent chat template
├── data/                 # Training and evaluation data (download separately)
├── eval/                 # HotpotQA and 2WikiMultiHopQA evaluation
└── train/                # Training code and launch scripts
```

## RL training framework

PARSER trains the lead policy with a fully asynchronous RL pipeline while
keeping the subagents frozen:

```text
SGLang Router (frozen subagents on multiple SGLang servers)
                         ↕ continuous queries and observations
Lead-agent rollout workers (continuous trajectory generation)
                         ↓ trajectories
                         ↑ updated parameters every 4 steps
Megatron Trainer (policy updates)
```

- **Fully asynchronous training.** Rollout generation and policy optimization
  run continuously on separate compute pools. While Megatron updates the lead
  policy, the subagent cluster continues serving rollout requests, minimizing
  subagent idle time.
- **Cache-aware subagent routing.** SGLang Router and Radix Cache route the
  same document chunk to the same SGLang worker across reasoning rounds. This
  maximizes KV-cache reuse and reduces repeated prefill cost.

## Model preparation

Both the lead agent and the subagents use Qwen3.5. The original Qwen3.5 chat
template omits thinking content from earlier assistant turns. PARSER requires
that content to remain visible to the lead agent, so copy the template supplied
in this repository into the lead-agent checkpoint:

```bash
cp chat_template.jinja /path/to/Qwen3.5-lead-agent-checkpoint/chat_template.jinja
```

The subagent checkpoint can use the original Qwen3.5 chat template.

## Installation

We recommend creating a dedicated Conda environment named `mcore` and using
**Python 3.12** with **CUDA 12.8/12.9**.

```bash
conda create -n mcore python=3.12 -y
conda activate mcore
```

If you use a different environment name, change `conda activate mcore` in
`train/train_4b.sh` accordingly.

Before installing the dependencies, edit the first two lines of
`train/env_setup.sh` so that `CUDA_HOME` points to your CUDA installation:

```bash
cd train
# Edit CUDA_HOME in env_setup.sh first.
bash env_setup.sh
```

### SGLang Router compatibility patch

The installed `sglang_router` requires a one-line import update to work with
`sglang==0.5.10.post1`. Locate
`<CONDA_PREFIX>/lib/python3.12/site-packages/sglang_router/launch_server.py`
and make the following change:

```diff
-from sglang.srt.utils import is_port_available
+from sglang.srt.utils.network import is_port_available
```

You can print the expected file path with:

```bash
python -c "import site; print(site.getsitepackages()[0] + '/sglang_router/launch_server.py')"
```

### NGINX

PARSER uses NGINX as a local proxy to efficiently reuse connections to SGLang
services across nodes. Install it on the training node, for example:

```bash
sudo apt-get update
sudo apt-get install -y nginx
```

If NGINX is installed at a nonstandard location, set
`LONGMAS_NGINX_BIN=/absolute/path/to/nginx`. The training launcher starts and
stops the required local NGINX proxies automatically.

## Data

Download the complete
[parser_dataset](https://huggingface.co/datasets/inNexus/parser_dataset)
repository into `data/` at the repository root:

```bash
cd /path/to/PARSER
hf download inNexus/parser_dataset \
  --repo-type dataset \
  --local-dir data
```

The resulting layout should be:

```text
data/
├── train/
│   └── hotpotqa_train_process_emb-select_llm-unable.parquet
├── hqa_val/
└── 2wiki_val/
```

## Training

### 1. Start the subagent service

Edit `train/set_sglang_router.sh` and configure:

- `--served-model-name`: the subagent model name;
- `--model-path`: the subagent checkpoint path;
- `--dp-size`: the number of data-parallel workers;
- `--host` and `--port`: the address exposed by the router.

Start the SGLang Router in a separate terminal:

```bash
cd train
bash set_sglang_router.sh
```

Record its address as `HOST:PORT`. Multiple routers may be launched on
different nodes or ports to provide additional subagent capacity.

SGLang Router can add or remove SGLang worker nodes at any time without
restarting the router. Example commands for listing, registering, and deleting
workers are provided at the end of `train/set_sglang_router.sh`.

### 2. Configure training

Set the lead-agent Qwen3.5 checkpoint in
`train/run_fully_async_parser_4b.sh` (`model_path`) and list all SGLang Router
addresses in `train/train_4b.sh`:

```bash
export LONGMAS_SGLANG_UPSTREAMS=host1:port1,host2:port2
```

The value must contain router addresses without the `/v1` suffix.

### 3. Launch training

```bash
cd train
bash train_4b.sh
```

The default configuration trains a Qwen3.5-4B lead agent with fully
asynchronous GRPO. Hardware and optimization settings can be overridden with
the environment variables defined in `train/run_fully_async_parser_4b.sh`.

## Evaluation

Evaluation reuses the SGLang Router service used by the subagents. Apart from
the GPUs serving the subagents, evaluating the lead agent requires one GPU.

For HotpotQA, edit `eval/eval_hqa.sh`:

1. Set `CUDA_VISIBLE_DEVICES` to the GPU used by the lead agent.
2. Set `URL` to the router API endpoint, including `/v1`.
3. Set `SUBAGENT_MODEL` to the router's served model name.
4. Set `TOKENIZER_PATH` to the original Qwen3.5-4B checkpoint.
5. Add the trained lead-agent checkpoint path to `MODELS`.

Then run:

```bash
cd eval
bash eval_hqa.sh
```

Use the equivalent script for 2WikiMultiHopQA:

```bash
cd eval
bash eval_2wiki.sh
```

The scripts preprocess each evaluation split when necessary and write the
predictions under their respective evaluation output directories.

## Citation

If you find PARSER useful, please cite:

```bibtex
@misc{li2026parserreadparallelreason,
      title={PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents},
      author={Kun Li and Zexuan Qiu and Tianhua Zhang and Irwin King and Helen Meng},
      year={2026},
      eprint={2609.06702},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.06702},
}
```

## Acknowledgements

This repository builds on
[Qwen](https://github.com/QwenLM/Qwen3),
[SGLang](https://github.com/sgl-project/sglang),
[VERL](https://github.com/volcengine/verl), and
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

## License

This project is released under the [MIT License](LICENSE).
