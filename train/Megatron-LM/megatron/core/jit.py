# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os

import torch

from megatron.core.utils import is_torch_min_version


def _env_flag_enabled(var: str) -> bool:
    return os.environ.get(var, "").strip().lower() in ("1", "true", "yes", "on")

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2


def noop_decorator(func):
    '''No-op decorator'''
    return func


def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global jit_fuser
    # torch.compile eagerly pulls Torch Inductor at import-time; some torch/triton combos
    # crash with AssertionError duplicate template name. Set MEGATRON_DISABLE_JIT_FUSER=1 to skip.
    if _env_flag_enabled("MEGATRON_DISABLE_JIT_FUSER"):
        jit_fuser = noop_decorator
        return
    try:
        if is_torch_min_version("2.2.0a0"):
            jit_fuser = torch.compile
    except ImportError:

        jit_fuser = noop_decorator


def disable_jit_fuser():
    '''Disable the JIT fuser'''
    global jit_fuser
    jit_fuser = noop_decorator


enable_jit_fuser()
