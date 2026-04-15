#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import importlib.util
import logging

logger = logging.getLogger(__name__)

# vLLM V1 removes `vllm.worker.cache_engine`. Keep this patch for legacy
# layouts only, and skip it gracefully on V1 to avoid hard import failures.
if importlib.util.find_spec("vllm.worker.cache_engine") is not None:
    import vllm_ascend.worker.cache_engine  # noqa: F401
else:
    logger.info(
        "Skip vllm_ascend.worker.cache_engine patch: "
        "vllm.worker.cache_engine not found (likely vLLM V1 layout).")
