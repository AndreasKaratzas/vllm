# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.
"""

import sys
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

from .quant_config import DeepseekV4FP8Config

# Pick the per-platform implementation. The NVIDIA branch is the static
# default that mypy sees; the ROCm branch overrides it at runtime and is
# kept type-compatible via ``# type: ignore[assignment]``.
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia import model as _model_module
    from .nvidia import mtp as _mtp_module
else:
    from .amd import model as _model_module
    from .amd import mtp as _mtp_module

DeepseekV4ForCausalLM = _model_module.DeepseekV4ForCausalLM
DeepSeekV4MTP = _mtp_module.DeepSeekV4MTP

# The AMD files are currently symlinks to the NVIDIA implementation. Ensure
# direct imports through either package path reuse the already-loaded module
# instead of executing the same source twice and re-registering custom ops.
if current_platform.is_rocm():
    sys.modules.setdefault(__name__ + ".nvidia.model", _model_module)
    sys.modules.setdefault(__name__ + ".nvidia.mtp", _mtp_module)

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
