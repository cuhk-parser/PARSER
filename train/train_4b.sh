#!/bin/bash

source ~/miniconda3/bin/activate # Enabled conda
eval "$(conda shell.bash hook)"
conda activate mcore      # Activate the Conda Environment
export LONGMAS_SGLANG_UPSTREAMS=ip1:port1,ip2:port2
bash run_fully_async_parser_4b.sh