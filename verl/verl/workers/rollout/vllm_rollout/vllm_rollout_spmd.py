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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import copy
import getpass
import glob
import inspect
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import cloudpickle as pickle
import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, LoRAConfig
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput

try:
    # https://github.com/vllm-project/vllm/commit/96b9aa5aa076e64c68765232aec343e4d0006e2a
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.model import get_lora_rank_from_adapter
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _should_hijack_vllm_lora() -> bool:
    # The hijack path imports vLLM LoRA internals, which can pull in CUDA/FP8-only
    # modules. Only enable it when LoRA is actually requested.
    return bool(os.getenv("VERL_ENABLE_VLLM_LORA_HIJACK", "0") == "1")


if is_version_ge(pkg="vllm", minver="0.7.3") and _should_hijack_vllm_lora():
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code

        lora_adapter_path = getattr(model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            lora_rank = get_lora_rank_from_adapter(lora_adapter_path)
        else:
            lora_rank = model_config.lora_rank

        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(lora_rank)}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        # # create nvfp4 model than sleep it
        # import multiprocessing as mp
        # mp.set_start_method('spawn', force=True)
        # self.inference_engine_quantization = LLM(
        #     model="/input0/Llama/Llama-3.1-8B-Instruct-wfp4afp4-nvfp4",
        #     enable_sleep_mode=True,
        #     tensor_parallel_size=1,
        #     # distributed_executor_backend="external_launcher",
        #     gpu_memory_utilization=0.4,
        #     trust_remote_code=True,
        #     seed=config.get("seed", 0),
        # )
        # print(f"inference_engine_quantization created!")
        # self.inference_engine_quantization.sleep(level=2)
        # print(f"inference_engine_quantization sleep 2!")

        print(f"enable_sleep_mode: {config.free_cache_engine}")
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @staticmethod
    def _unique_ids_in_order(id_array: np.ndarray) -> list[str]:
        seen: set[str] = set()
        ordered_ids: list[str] = []
        for item in id_array.tolist():
            item_str = str(item)
            if item_str in seen:
                continue
            seen.add(item_str)
            ordered_ids.append(item_str)
        return ordered_ids

    @staticmethod
    def _slice_non_tensor_batch(non_tensor_batch: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
        if len(indices) == 0:
            return {key: np.asarray(val)[:0] for key, val in non_tensor_batch.items()}
        index_array = np.asarray(indices, dtype=np.int64)
        return {key: np.asarray(val)[index_array] for key, val in non_tensor_batch.items()}

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _online_migration_enabled(self) -> bool:
        return self._env_flag("VERL_REDUNDANT_ONLINE_MIGRATION_ENABLED", False)

    @staticmethod
    def _build_synthetic_request_output(
        *,
        request_id: str,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
    ) -> RequestOutput:
        return RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=list(prompt_token_ids),
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=list(response_token_ids),
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="length" if response_token_ids else "stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    def _generate_with_online_migration(
        self,
        *,
        vllm_inputs: list[dict[str, Any]],
        lora_requests: list[LoRARequest] | None,
        migration_run_tag: str,
    ) -> tuple[list[Any], dict[str, Any]]:
        if lora_requests is not None:
            raise NotImplementedError("Online migration path does not support LoRA requests yet.")
        if self.config.calculate_log_probs:
            raise NotImplementedError("Online migration path does not support rollout logprobs.")

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        trigger_response_len = 8192
        start_time_s = time.perf_counter()

        active_request_meta: dict[str, dict[str, Any]] = {}
        pending_request_ids: set[str] = set()
        latest_local_response_suffix: dict[str, list[int]] = {}
        completed_records_local: list[dict[str, Any]] = []
        local_origin_request_count = len(vllm_inputs)
        migration_done = False
        migration_disabled_for_small_tail = False
        small_tail_collective_exit = False
        migration_count = 0
        migration_reason = "none"
        trigger_flag_dir = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
        trigger_flag_prefix = os.path.join(
            trigger_flag_dir,
            f"verl_tail_migration_{migration_run_tag}",
        )

        def add_active_request(
            *,
            request_id: str,
            prompt: dict[str, Any],
            origin_rank: int,
            origin_request_idx: int,
            original_prompt_token_ids: list[int],
            response_prefix_token_ids: list[int],
            max_tokens: int,
        ) -> None:
            params = self.sampling_params if max_tokens == self.config.response_length else copy.deepcopy(self.sampling_params)
            params.max_tokens = max_tokens
            active_request_meta[request_id] = {
                "origin_rank": origin_rank,
                "origin_request_idx": origin_request_idx,
                "original_prompt_token_ids": list(original_prompt_token_ids),
                "original_prompt": {
                    key: (list(value) if key == "prompt_token_ids" else value)
                    for key, value in prompt.items()
                },
                "response_prefix_token_ids": list(response_prefix_token_ids),
            }
            pending_request_ids.add(request_id)
            self.inference_engine.add_request_with_id(
                request_id=request_id,
                prompt=prompt,
                params=params,
                lora_request=None,
            )

        for request_idx, prompt in enumerate(vllm_inputs):
            request_id = f"tailmig-r{rank}-req-{request_idx}"
            prompt_token_ids = list(prompt["prompt_token_ids"])
            add_active_request(
                request_id=request_id,
                prompt=prompt,
                origin_rank=rank,
                origin_request_idx=request_idx,
                original_prompt_token_ids=prompt_token_ids,
                response_prefix_token_ids=[],
                max_tokens=self.config.response_length,
            )

        for stale_trigger_path in glob.glob(f"{trigger_flag_prefix}_m*.flag"):
            try:
                os.remove(stale_trigger_path)
            except FileNotFoundError:
                pass

        try:
            while True:
                local_has_unfinished = bool(pending_request_ids)
                if small_tail_collective_exit and torch.distributed.is_initialized():
                    flag_device = "cuda" if torch.cuda.is_available() else "cpu"
                    unfinished_flag = torch.tensor(
                        [1 if local_has_unfinished else 0],
                        device=flag_device,
                        dtype=torch.int64,
                    )
                    torch.distributed.all_reduce(unfinished_flag, op=torch.distributed.ReduceOp.MAX)
                    if unfinished_flag.item() == 0:
                        break
                step_outputs = self.inference_engine.step_engine() if local_has_unfinished else []
                local_trigger_reason = "none"
                current_trigger_path = f"{trigger_flag_prefix}_m{migration_count + 1}.flag"

                for output in step_outputs:
                    request_id = output.request_id
                    request_state = active_request_meta.get(request_id)
                    if request_state is None:
                        continue
                    first_output = output.outputs[0] if output.outputs else None
                    if first_output is None:
                        continue

                    response_suffix_token_ids = list(first_output.token_ids)
                    if output.finished:
                        pending_request_ids.discard(request_id)
                        latest_local_response_suffix.pop(request_id, None)
                        if first_output.finish_reason == "abort":
                            active_request_meta.pop(request_id, None)
                            continue
                        completed_records_local.append(
                            {
                                "origin_rank": request_state["origin_rank"],
                                "origin_request_idx": request_state["origin_request_idx"],
                                "request_id": request_id,
                                "prompt_token_ids": list(request_state["original_prompt_token_ids"]),
                                "response_token_ids": list(request_state["response_prefix_token_ids"])
                                + response_suffix_token_ids,
                            }
                        )
                        active_request_meta.pop(request_id, None)
                    else:
                        latest_local_response_suffix[request_id] = response_suffix_token_ids

                local_max_response_len = 0
                for request_id in pending_request_ids:
                    request_state = active_request_meta[request_id]
                    response_len = len(request_state["response_prefix_token_ids"]) + len(
                        latest_local_response_suffix.get(request_id, [])
                    )
                    local_max_response_len = max(local_max_response_len, response_len)
                if not migration_disabled_for_small_tail:
                    if not pending_request_ids:
                        local_trigger_reason = "idle"
                    elif migration_count == 0 and local_max_response_len >= trigger_response_len:
                        local_trigger_reason = "len8k"

                if local_trigger_reason != "none" and not os.path.exists(current_trigger_path):
                    try:
                        with open(current_trigger_path, "w", encoding="ascii") as f:
                            f.write(local_trigger_reason)
                    except OSError:
                        pass

                global_trigger = local_trigger_reason != "none" or os.path.exists(current_trigger_path)
                if global_trigger:
                    local_unfinished_records: list[dict[str, Any]] = []
                    for request_id in sorted(
                        pending_request_ids,
                        key=lambda rid: (
                            active_request_meta[rid]["origin_rank"],
                            active_request_meta[rid]["origin_request_idx"],
                        ),
                    ):
                        request_state = active_request_meta[request_id]
                        response_prefix_token_ids = list(request_state["response_prefix_token_ids"]) + list(
                            latest_local_response_suffix.get(request_id, [])
                        )
                        remaining_tokens = max(0, self.config.response_length - len(response_prefix_token_ids))
                        if remaining_tokens <= 0:
                            completed_records_local.append(
                                {
                                    "origin_rank": request_state["origin_rank"],
                                    "origin_request_idx": request_state["origin_request_idx"],
                                    "request_id": request_id,
                                    "prompt_token_ids": list(request_state["original_prompt_token_ids"]),
                                    "response_token_ids": response_prefix_token_ids,
                                }
                            )
                            continue

                        migrated_prompt = {
                            key: (list(value) if key == "prompt_token_ids" else value)
                            for key, value in request_state["original_prompt"].items()
                        }
                        migrated_prompt["prompt_token_ids"] = list(
                            request_state["original_prompt_token_ids"]
                        ) + list(response_prefix_token_ids)
                        local_unfinished_records.append(
                            {
                                "origin_rank": request_state["origin_rank"],
                                "origin_request_idx": request_state["origin_request_idx"],
                                "prompt": migrated_prompt,
                                "original_prompt_token_ids": list(request_state["original_prompt_token_ids"]),
                                "response_prefix_token_ids": response_prefix_token_ids,
                                "remaining_tokens": remaining_tokens,
                            }
                        )

                    gathered_trigger_reasons = (
                        [None for _ in range(world_size)] if torch.distributed.is_initialized() else None
                    )
                    gathered_unfinished_records = (
                        [None for _ in range(world_size)] if torch.distributed.is_initialized() else None
                    )
                    if torch.distributed.is_initialized():
                        torch.distributed.all_gather_object(
                            gathered_trigger_reasons,
                            local_trigger_reason if local_trigger_reason != "none" else "none",
                        )
                        torch.distributed.all_gather_object(
                            gathered_unfinished_records,
                            local_unfinished_records,
                        )
                        flat_unfinished_records = [
                            record
                            for rank_records in gathered_unfinished_records
                            for record in rank_records
                        ]
                        pre_counts = [len(rank_records) for rank_records in gathered_unfinished_records]
                    else:
                        gathered_trigger_reasons = [local_trigger_reason]
                        flat_unfinished_records = local_unfinished_records
                        pre_counts = [len(local_unfinished_records)]

                    try:
                        os.remove(current_trigger_path)
                    except FileNotFoundError:
                        pass

                    total_unfinished = len(flat_unfinished_records)
                    if total_unfinished == 0:
                        break
                    if total_unfinished <= world_size:
                        migration_disabled_for_small_tail = True
                        small_tail_collective_exit = True
                        if rank == 0:
                            logger.warning(
                                "[VERL online-migration] skip migration: total_unfinished=%s <= dp_world_size=%s",
                                total_unfinished,
                                world_size,
                            )
                        continue

                    if pending_request_ids:
                        self.inference_engine.abort_requests(sorted(pending_request_ids))
                    pending_request_ids.clear()
                    active_request_meta.clear()
                    latest_local_response_suffix.clear()

                    migration_done = True
                    migration_count += 1
                    migration_reason = next(
                        (reason for reason in gathered_trigger_reasons if reason != "none"),
                        "unknown",
                    )

                    base_chunk = total_unfinished // world_size
                    remainder = total_unfinished % world_size
                    chunk_start = rank * base_chunk + min(rank, remainder)
                    chunk_end = chunk_start + base_chunk + (1 if rank < remainder else 0)
                    reassigned_records = flat_unfinished_records[chunk_start:chunk_end]
                    post_counts = [
                        (base_chunk + (1 if rank_idx < remainder else 0))
                        for rank_idx in range(world_size)
                    ]

                    logger.warning(
                        "[VERL online-migration] rank=%s trigger=%s total_unfinished=%s reassigned_local=%s",
                        rank,
                        migration_reason,
                        total_unfinished,
                        len(reassigned_records),
                    )
                    if rank == 0:
                        logger.warning(
                            "[VERL online-migration] trigger=%s migration_idx=%s pre_counts=%s post_counts=%s",
                            migration_reason,
                            migration_count,
                            pre_counts,
                            post_counts,
                        )

                    for reassigned in reassigned_records:
                        new_request_id = (
                            f"tailmig-m{migration_count}-o{reassigned['origin_rank']}-i{reassigned['origin_request_idx']}"
                        )
                        add_active_request(
                            request_id=new_request_id,
                            prompt=reassigned["prompt"],
                            origin_rank=int(reassigned["origin_rank"]),
                            origin_request_idx=int(reassigned["origin_request_idx"]),
                            original_prompt_token_ids=list(reassigned["original_prompt_token_ids"]),
                            response_prefix_token_ids=list(reassigned["response_prefix_token_ids"]),
                            max_tokens=int(reassigned["remaining_tokens"]),
                        )
                    continue

                if not pending_request_ids:
                    if small_tail_collective_exit:
                        continue
                    # This only waits for another rank to publish the next migration trigger.
                    time.sleep(0.001)
        finally:
            for stale_trigger_path in glob.glob(f"{trigger_flag_prefix}_m*.flag"):
                try:
                    os.remove(stale_trigger_path)
                except FileNotFoundError:
                    pass

        if migration_done and torch.distributed.is_initialized():
            gathered_completed_records = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_completed_records, completed_records_local)
            visible_completed_records = [
                record
                for rank_records in gathered_completed_records
                for record in rank_records
                if int(record["origin_rank"]) == rank
            ]
        else:
            visible_completed_records = completed_records_local

        visible_completed_records.sort(key=lambda record: int(record["origin_request_idx"]))
        if len(visible_completed_records) != local_origin_request_count:
            raise RuntimeError(
                "Online migration returned mismatched local request count: "
                f"expected={local_origin_request_count}, got={len(visible_completed_records)}, rank={rank}."
            )

        outputs = [
            self._build_synthetic_request_output(
                request_id=str(record["request_id"]),
                prompt_token_ids=list(record["prompt_token_ids"]),
                response_token_ids=list(record["response_token_ids"]),
            )
            for record in visible_completed_records
        ]
        round_stats = {
            "input_request_count": local_origin_request_count,
            "finished_request_count": len(outputs),
            "accepted_request_count": len(outputs),
            "accepted_request_indices": list(range(local_origin_request_count)),
            "accepted_prompt_uids": [],
            "completed_prompt_uids": [],
            "aborted_request_count": 0,
            "exec_s": time.perf_counter() - start_time_s,
            "online_migration_enabled": True,
            "online_migration_count": migration_count,
            "online_migration_reason": migration_reason,
        }
        return outputs, round_stats

    def _generate_with_redundant_early_stop(
        self,
        *,
        vllm_inputs: list[dict[str, Any]],
        lora_requests: list[LoRARequest] | None,
        prompt_uids: np.ndarray,
        target_prompt_count: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        repeat_per_prompt = int(self.config.n)
        partial_accept_ratio = float(getattr(self.config, "redundant_partial_prompt_accept_ratio", 1.0))
        partial_accept_ratio = min(max(partial_accept_ratio, 0.0), 1.0)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        start_time_s = time.perf_counter()

        request_meta: dict[str, dict[str, Any]] = {}
        pending_request_ids: set[str] = set()
        local_prompt_order: list[str] = []
        local_prompt_seen: set[str] = set()
        for request_idx, (prompt, prompt_uid) in enumerate(zip(vllm_inputs, prompt_uids, strict=True)):
            request_id = f"redundant-r{rank}-req-{request_idx}"
            prompt_uid_str = str(prompt_uid)
            request_meta[request_id] = {
                "request_idx": request_idx,
                "uid": prompt_uid_str,
            }
            if prompt_uid_str not in local_prompt_seen:
                local_prompt_seen.add(prompt_uid_str)
                local_prompt_order.append(prompt_uid_str)
            pending_request_ids.add(request_id)
            lora_request = None if lora_requests is None else lora_requests[request_idx]
            self.inference_engine.add_request_with_id(
                request_id=request_id,
                prompt=prompt,
                params=self.sampling_params,
                lora_request=lora_request,
            )

        completed_request_counts: dict[str, int] = defaultdict(int)
        completed_prompt_uids: list[str] = []
        completed_prompt_uid_set: set[str] = set()
        finished_outputs: dict[str, Any] = {}

        while True:
            local_has_unfinished = self.inference_engine.has_unfinished_requests()
            if torch.distributed.is_initialized():
                flag_device = "cuda" if torch.cuda.is_available() else "cpu"
                unfinished_flag = torch.tensor(
                    [1 if local_has_unfinished else 0],
                    device=flag_device,
                    dtype=torch.int64,
                )
                torch.distributed.all_reduce(unfinished_flag, op=torch.distributed.ReduceOp.MAX)
                global_has_unfinished = bool(unfinished_flag.item())
            else:
                global_has_unfinished = local_has_unfinished

            if not global_has_unfinished:
                break

            step_outputs = self.inference_engine.step_engine() if local_has_unfinished else []
            local_finished_uids: list[str] = []
            for output in step_outputs:
                if not output.finished:
                    continue

                request_id = output.request_id
                pending_request_ids.discard(request_id)
                first_output = output.outputs[0] if output.outputs else None
                if first_output is not None and first_output.finish_reason == "abort":
                    continue

                finished_outputs[request_id] = output
                local_finished_uids.append(request_meta[request_id]["uid"])

            if torch.distributed.is_initialized():
                gathered_finished_uids = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(gathered_finished_uids, local_finished_uids)
            else:
                gathered_finished_uids = [local_finished_uids]

            for rank_finished_uids in gathered_finished_uids:
                for prompt_uid in rank_finished_uids:
                    completed_request_counts[prompt_uid] += 1
                    if (
                        completed_request_counts[prompt_uid] == repeat_per_prompt
                        and prompt_uid not in completed_prompt_uid_set
                    ):
                        completed_prompt_uid_set.add(prompt_uid)
                        completed_prompt_uids.append(prompt_uid)

            if len(completed_prompt_uids) >= target_prompt_count:
                if pending_request_ids:
                    self.inference_engine.abort_requests(sorted(pending_request_ids))
                break

        if torch.distributed.is_initialized():
            gathered_prompt_orders = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_prompt_orders, local_prompt_order)
            prompt_order = [uid for rank_prompt_order in gathered_prompt_orders for uid in rank_prompt_order]
        else:
            prompt_order = local_prompt_order

        base_accepted_prompt_uids = completed_prompt_uids[:target_prompt_count]
        accepted_prompt_uids: list[str] = list(base_accepted_prompt_uids)
        accepted_prompt_uid_set = set(accepted_prompt_uids)
        partially_accepted_prompt_uids: list[str] = []
        partial_completion_counts: dict[str, int] = {}
        if partial_accept_ratio < 1.0:
            for prompt_uid in prompt_order:
                if prompt_uid in accepted_prompt_uid_set:
                    continue
                completed_count = completed_request_counts.get(prompt_uid, 0)
                if repeat_per_prompt > 0 and (completed_count / repeat_per_prompt) >= partial_accept_ratio:
                    accepted_prompt_uids.append(prompt_uid)
                    accepted_prompt_uid_set.add(prompt_uid)
                    if completed_count < repeat_per_prompt:
                        partially_accepted_prompt_uids.append(prompt_uid)
                        partial_completion_counts[prompt_uid] = completed_count

        accepted_request_entries: list[tuple[int, Any, bool, str]] = []
        for prompt_uid in prompt_order:
            if prompt_uid not in accepted_prompt_uid_set:
                continue

            prompt_finished_entries = sorted(
                [
                    (request_meta[request_id]["request_idx"], output)
                    for request_id, output in finished_outputs.items()
                    if request_meta[request_id]["uid"] == prompt_uid
                ],
                key=lambda item: item[0],
            )
            prompt_request_indices = [
                request_meta[request_id]["request_idx"]
                for request_id in sorted(
                    [
                        request_id
                        for request_id, meta in request_meta.items()
                        if meta["uid"] == prompt_uid
                    ],
                    key=lambda request_id: request_meta[request_id]["request_idx"],
                )
            ]
            finished_request_idx_set = {request_idx for request_idx, _ in prompt_finished_entries}

            for request_idx, output in prompt_finished_entries:
                accepted_request_entries.append((request_idx, output, False, prompt_uid))

            for request_idx in prompt_request_indices:
                if request_idx in finished_request_idx_set:
                    continue
                accepted_request_entries.append((request_idx, None, True, prompt_uid))

        accepted_request_entries.sort(key=lambda item: item[0])
        accepted_outputs = [output for _, output, _, _ in accepted_request_entries]
        accepted_request_indices = [request_idx for request_idx, _, _, _ in accepted_request_entries]
        dummy_request_indices = [request_idx for request_idx, _, is_dummy, _ in accepted_request_entries if is_dummy]
        accepted_grpo_uids = [
            (
                f"{prompt_uid}__partial_dummy__req{request_idx}"
                if is_dummy
                else prompt_uid
            )
            for request_idx, _, is_dummy, prompt_uid in accepted_request_entries
        ]
        round_stats = {
            "input_request_count": len(vllm_inputs),
            "finished_request_count": len(finished_outputs),
            "accepted_request_count": len(accepted_outputs),
            "accepted_request_indices": accepted_request_indices,
            "accepted_prompt_uids": accepted_prompt_uids,
            "completed_prompt_uids": completed_prompt_uids,
            "partially_accepted_prompt_uids": partially_accepted_prompt_uids,
            "partial_completion_counts": partial_completion_counts,
            "dummy_request_indices": dummy_request_indices,
            "dummy_request_count": len(dummy_request_indices),
            "accepted_grpo_uids": accepted_grpo_uids,
            "partial_accept_ratio": partial_accept_ratio,
            "aborted_request_count": max(0, len(vllm_inputs) - len(finished_outputs)),
            "exec_s": time.perf_counter() - start_time_s,
        }
        return accepted_outputs, round_stats

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            redundant_round_mode = prompts.meta_info.get("redundant_round_mode", "disabled")
            selected_request_indices = list(range(batch_size))
            round_stats = None
            if redundant_round_mode != "disabled":
                logger.warning(
                    "[VERL tail-batch] rollout rank=%s mode=%s batch_size=%s non_tensor_keys=%s has_uid=%s",
                    torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    redundant_round_mode,
                    batch_size,
                    sorted(non_tensor_batch.keys()),
                    "uid" in non_tensor_batch,
                )
            if redundant_round_mode == "redundant_short":
                target_prompt_count = int(prompts.meta_info["redundant_target_prompt_count"])
                if "uid" not in non_tensor_batch:
                    raise ValueError(
                        "Redundant rollout requires uid in non_tensor_batch. "
                        f"available_keys={sorted(non_tensor_batch.keys())}"
                    )
                outputs, round_stats = self._generate_with_redundant_early_stop(
                    vllm_inputs=vllm_inputs,
                    lora_requests=lora_requests,
                    prompt_uids=np.asarray(non_tensor_batch["uid"], dtype=object),
                    target_prompt_count=target_prompt_count,
                )
                selected_request_indices = round_stats["accepted_request_indices"]
            elif redundant_round_mode == "tail_replay" and self._online_migration_enabled():
                outputs, round_stats = self._generate_with_online_migration(
                    vllm_inputs=vllm_inputs,
                    lora_requests=lora_requests,
                    migration_run_tag=(
                        f"port{os.getenv('MASTER_PORT', '0')}_"
                        f"step{prompts.meta_info.get('global_steps', 'na')}_"
                        f"rankset{torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1}"
                    ),
                )
            else:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
                round_stats = {
                    "input_request_count": batch_size,
                    "finished_request_count": len(outputs),
                    "accepted_request_count": len(outputs),
                    "accepted_request_indices": selected_request_indices,
                    "accepted_prompt_uids": self._unique_ids_in_order(np.asarray(non_tensor_batch["uid"], dtype=object))
                    if "uid" in non_tensor_batch
                    else [],
                    "completed_prompt_uids": [],
                    "partially_accepted_prompt_uids": [],
                    "partial_completion_counts": {},
                    "dummy_request_indices": [],
                    "dummy_request_count": 0,
                    "accepted_grpo_uids": (
                        np.asarray(non_tensor_batch["uid"], dtype=object).astype(str).tolist()
                        if "uid" in non_tensor_batch
                        else []
                    ),
                    "partial_accept_ratio": 1.0,
                    "aborted_request_count": 0,
                    "exec_s": 0.0,
                }

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        if len(selected_request_indices) != len(outputs):
            raise RuntimeError(
                "Mismatch between selected requests and generation outputs: "
                f"{len(selected_request_indices)} vs {len(outputs)}."
            )

        if len(selected_request_indices) != batch_size:
            idx = idx[selected_request_indices]
            attention_mask = attention_mask[selected_request_indices]
            position_ids = position_ids[selected_request_indices]
            non_tensor_batch = self._slice_non_tensor_batch(non_tensor_batch, selected_request_indices)
            batch_size = len(selected_request_indices)

        # print(f"inference_engine sleep 1!")
        # self.inference_engine.sleep(level=1)
        # time.sleep(5)
        # print(f"inference_engine_quantization wake up!")
        # self.inference_engine_quantization.wake_up(tags=["weights"])
        # self.inference_engine_quantization.wake_up(tags=["kv_cache"])
        # print(f"inference_engine_quantization second sleep 2!")
        # self.inference_engine_quantization.sleep(level=2)
        # time.sleep(5)
        # print(f"inference_engine wake up!")
        # self.inference_engine.wake_up()

        selected_attention_mask = attention_mask
        response = []
        rollout_log_probs = []
        response_token_lengths = []
        prompt_token_lengths = selected_attention_mask.sum(dim=-1).detach().cpu().tolist()
        dummy_request_mask = []
        grpo_uids = round_stats.get("accepted_grpo_uids", []) if round_stats is not None else []
        for output in outputs:
            if output is None:
                response.append([])
                response_token_lengths.append(0)
                dummy_request_mask.append(True)
                if self.config.calculate_log_probs:
                    rollout_log_probs.append([])
                continue

            first_output = output.outputs[0] if output.outputs else None
            response_ids = [] if first_output is None else first_output.token_ids
            response.append(response_ids)
            response_token_lengths.append(len(response_ids))
            dummy_request_mask.append(False)
            if self.config.calculate_log_probs:
                curr_log_prob = []
                if first_output is not None:
                    for i, logprob in enumerate(first_output.logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                rollout_log_probs.append(curr_log_prob)

        # Redundant short rounds may leave some DP ranks with no accepted requests after
        # the global early-stop selection. Keep returning an empty local batch instead of
        # crashing in padding helpers that assume at least one response.
        if response:
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)
        else:
            response = torch.empty((batch_size, self.config.response_length), dtype=idx.dtype, device=idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs = torch.empty(
                    (batch_size, self.config.response_length),
                    dtype=torch.float32,
                    device=idx.device,
                )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        if dummy_request_mask:
            dummy_request_mask_tensor = torch.tensor(dummy_request_mask, device=attention_mask.device, dtype=torch.bool)
            response_attention_mask[dummy_request_mask_tensor] = 0
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_attention_mask,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        non_tensor_batch["prompt_token_len"] = np.asarray(prompt_token_lengths, dtype=np.int32)
        non_tensor_batch["response_token_len"] = np.asarray(response_token_lengths, dtype=np.int32)
        non_tensor_batch["total_token_len"] = np.asarray(
            [prompt_len + response_len for prompt_len, response_len in zip(prompt_token_lengths, response_token_lengths)],
            dtype=np.int32,
        )
        non_tensor_batch["redundant_partial_dummy_mask"] = np.asarray(dummy_request_mask, dtype=bool)
        if grpo_uids:
            non_tensor_batch["grpo_uid"] = np.asarray(grpo_uids, dtype=object)

        if redundant_round_mode != "disabled":
            logger.warning(
                "[VERL tail-batch] rollout rank=%s mode=%s accepted_prompts=%s partial_prompts=%s "
                "accepted_requests=%s dummy_requests=%s finished_requests=%s aborted_requests=%s exec_s=%.3f",
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                redundant_round_mode,
                len(round_stats["accepted_prompt_uids"]) if round_stats is not None else 0,
                len(round_stats.get("partially_accepted_prompt_uids", [])) if round_stats is not None else 0,
                len(selected_request_indices),
                round_stats.get("dummy_request_count", 0) if round_stats is not None else 0,
                round_stats["finished_request_count"] if round_stats is not None else len(outputs),
                round_stats["aborted_request_count"] if round_stats is not None else 0,
                round_stats["exec_s"] if round_stats is not None else 0.0,
            )
            if round_stats is not None and round_stats.get("online_migration_enabled", False):
                logger.warning(
                    "[VERL online-migration] rank=%s mode=%s migrations=%s reason=%s",
                    torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    redundant_round_mode,
                    round_stats.get("online_migration_count", 0),
                    round_stats.get("online_migration_reason", "none"),
                )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        # On Ascend, camem extension may be unavailable; in that case vLLM
        # disables sleep mode at runtime and calling sleep() would assert.
        try:
            self.inference_engine.sleep(level=self.sleep_level)
        except AssertionError as e:
            if "Sleep mode is not enabled in the model config" in str(e):
                logger.warning("Skip vLLM sleep() because sleep mode is disabled at runtime.")
                return
            raise

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
