# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

from .base import BaseEngine, EngineRegistry, note_megatron_engine_import_failed
from .fsdp import DiffusersFSDPEngine, FSDPEngine, FSDPEngineWithLMHead

logger = logging.getLogger(__name__)

__all__ = [
    "BaseEngine",
    "EngineRegistry",
    "FSDPEngine",
    "FSDPEngineWithLMHead",
]

if DiffusersFSDPEngine is not None:
    __all__.append("DiffusersFSDPEngine")

try:
    from .torchtitan import TorchTitanEngine, TorchTitanEngineWithLMHead

    __all__ += ["TorchTitanEngine", "TorchTitanEngineWithLMHead"]
except ImportError:
    TorchTitanEngine = None
    TorchTitanEngineWithLMHead = None

try:
    from .veomni import VeOmniEngine, VeOmniEngineWithLMHead

    __all__ += ["VeOmniEngine", "VeOmniEngineWithLMHead"]
except ImportError:
    VeOmniEngine = None
    VeOmniEngineWithLMHead = None

try:
    from .automodel import AutomodelEngine, AutomodelEngineWithLMHead

    __all__ += ["AutomodelEngine", "AutomodelEngineWithLMHead"]
except ImportError:
    AutomodelEngine = None
    AutomodelEngineWithLMHead = None

# Mindspeed must be imported before Megatron to ensure the related monkey patches take effect as expected
try:
    from .mindspeed import MindspeedEngineWithLMHead, MindSpeedLLMEngineWithLMHead

    __all__ += ["MindspeedEngineWithLMHead", "MindSpeedLLMEngineWithLMHead"]
except ImportError:
    MindspeedEngineWithLMHead = None
    MindSpeedLLMEngineWithLMHead = None

try:
    from .megatron import MegatronEngine, MegatronEngineWithLMHead

    __all__ += ["MegatronEngine", "MegatronEngineWithLMHead"]
except ImportError as e:
    note_megatron_engine_import_failed(e)
    MegatronEngine = None
    MegatronEngineWithLMHead = None
    logger.error(
        "Megatron training engine failed to import; strategy=megatron will not work until fixed. (%s: %s)",
        type(e).__name__,
        e,
    )
