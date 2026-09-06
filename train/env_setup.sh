export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install verl[mcore]==0.8.0
pip install qwen_vl_utils
pip uninstall mbridge -y
pip install sglang[all]==0.5.10.post1 --no-build-isolation --no-cache-dir
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir --no-binary flash-attn
pip install causal-conv1d==1.6.1 --no-build-isolation --no-cache-dir
pip install --no-build-isolation flash-linear-attention==0.5.0 --no-cache-dir
pip install cupy-cuda12x==14.0.1 --no-build-isolation --no-cache-dir
pip install git+https://github.com/ISEEKYAN/mbridge.git --no-build-isolation
pip install tilelang==0.1.9 apache-tvm-ffi==0.1.11 --no-build-isolation --no-cache-dir
# git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e . --no-build-isolation
pip install transformers==5.7.0
pip install nvidia-cutlass-dsl==4.4.2 --no-build-isolation
pip install nvidia-cutlass-dsl-libs-base==4.4.2 --no-build-isolation
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export CUDNN_PATH="$SP/nvidia/cudnn"
export CUDNN_HOME="$CUDNN_PATH"
export NCCL_HOME="$SP/nvidia/nccl"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:$CUDNN_PATH/include:$NCCL_HOME/include"
export CPATH="$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$CUDNN_PATH/lib:$NCCL_HOME/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$NCCL_HOME/lib:$LD_LIBRARY_PATH"

pip install --no-build-isolation transformer-engine[pytorch,core_cu12]==2.14.1 --no-cache-dir
pip install json_repair
pip install cachetools
pip install sglang_router