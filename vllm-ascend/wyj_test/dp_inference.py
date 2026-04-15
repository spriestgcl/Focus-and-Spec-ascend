#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
8-GPU data-parallel offline inference for DeepMath-103K with a local Qwen3-8B
checkpoint.

This script follows the same DP pattern as
`examples/offline_inference/data_parallel.py`, but reads prompts from a local
parquet dataset and writes merged JSONL outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Array, Barrier, Manager, Process
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import pandas as pd

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def parse_env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment variable {name}={raw_value!r}")


def parse_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="8-GPU data-parallel inference for Qwen3-8B on DeepMath-103K."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/models/Qwen/Qwen3-8B",
        help="Local model path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/data/deepmath-103k/train.parquet",
        help="Input parquet dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="vllm/wyj_test/outputs/qwen3_8b_deepmath_train_dp8.jsonl",
        help="Merged JSONL output path.",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column containing chat-format prompts.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=8,
        help="Data parallel size. For 8-card DP, keep this as 8.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size per DP rank.",
    )
    parser.add_argument(
        "--node-size",
        type=int,
        default=1,
        help="Total number of nodes.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Rank of the current node.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="",
        help="Master node IP address for multi-node DP.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="Master node port for multi-node DP.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N samples. 0 means the full dataset.",
    )
    parser.add_argument(
        "--dataset-indices",
        type=str,
        default="",
        help="Comma-separated dataset row indices to select and preserve as dataset_index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Base number of raw prompts processed per round across all DP ranks.",
    )
    parser.add_argument(
        "--redundancy-p",
        type=float,
        default=0.0,
        help="Redundant prompt fraction p. Redundant rounds use ceil(batch_size * (1 + p)) prompts.",
    )
    parser.add_argument(
        "--repeat-per-prompt",
        type=int,
        default=1,
        help="Repeat each prompt this many times within one rank-local batch.",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode.",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode.",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Defaults depend on thinking mode.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Sampling top-p. Defaults depend on thinking mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sampling top-k. Defaults depend on thinking mode.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Sampling min-p. Defaults depend on thinking mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM can allocate.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences to process in one iteration.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enable eager execution.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model/tokenizer.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Seconds to wait before killing a stuck worker process.",
    )
    return parser.parse_args()


def resolve_sampling_config(
    enable_thinking: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
) -> tuple[float, float, int, float]:
    if enable_thinking:
        default_temperature = 0.6
        default_top_p = 0.95
        default_top_k = 20
        default_min_p = 0.0
    else:
        default_temperature = 0.7
        default_top_p = 0.8
        default_top_k = 20
        default_min_p = 0.0

    return (
        default_temperature if temperature is None else temperature,
        default_top_p if top_p is None else top_p,
        default_top_k if top_k is None else top_k,
        default_min_p if min_p is None else min_p,
    )


def even_shard_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    floor = total // world_size
    remainder = total % world_size

    def start(idx: int) -> int:
        return idx * floor + min(idx, remainder)

    return start(rank), start(rank + 1)


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"Expected a list of chat messages, got {type(value)!r}")
    return value


def load_records(
    dataset_path: str,
    prompt_column: str,
    limit: int,
    dataset_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    dataframe = pd.read_parquet(dataset_path)
    if dataset_indices:
        dataframe = dataframe.iloc[dataset_indices]
    if limit > 0:
        dataframe = dataframe.head(limit)

    records: list[dict[str, Any]] = []
    source_indices = list(dataframe.index)
    for row_idx, row in enumerate(dataframe.itertuples(index=False)):
        row_dict = row._asdict()
        records.append(
            {
                "dataset_index": int(source_indices[row_idx]) if dataset_indices else row_idx,
                "data_source": row_dict.get("data_source"),
                "ability": row_dict.get("ability"),
                "reward_model": row_dict.get("reward_model"),
                "extra_info": row_dict.get("extra_info"),
                "messages": normalize_messages(row_dict[prompt_column]),
            }
        )
    return records


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_redundant_prompt_count(batch_size: int, redundancy_p: float) -> int:
    return max(batch_size, math.ceil(batch_size * (1.0 + redundancy_p)))


def expand_prompt_records(
    prompt_records: list[dict[str, Any]],
    repeat_per_prompt: int,
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "repeat_idx": repeat_idx,
        }
        for record in prompt_records
        for repeat_idx in range(repeat_per_prompt)
    ]


def build_output_row(
    *,
    record: dict[str, Any],
    output: Any,
    enable_thinking: bool,
    round_id: int,
    round_mode: str,
) -> dict[str, Any]:
    first_output = output.outputs[0]
    return {
        "dataset_index": record["dataset_index"],
        "data_source": record["data_source"],
        "ability": record["ability"],
        "reward_model": record["reward_model"],
        "extra_info": record["extra_info"],
        "prompt": record["messages"],
        "response": first_output.text,
        "finish_reason": first_output.finish_reason,
        "num_prompt_tokens": len(output.prompt_token_ids or []),
        "num_output_tokens": len(first_output.token_ids or []),
        "enable_thinking": enable_thinking,
        "repeat_idx": record["repeat_idx"],
        "round_id": round_id,
        "round_mode": round_mode,
    }


def split_completed_and_tail_prompts(
    *,
    prompt_records: list[dict[str, Any]],
    finished_rows: list[dict[str, Any]],
    repeat_per_prompt: int,
    round_id: int,
    round_mode: str,
    partial_completion_enabled: bool,
    partial_completion_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    for row in finished_rows:
        rows_by_index.setdefault(row["dataset_index"], []).append(row)

    completed_rows: list[dict[str, Any]] = []
    fully_completed_rows: list[dict[str, Any]] = []
    tail_prompt_records: list[dict[str, Any]] = []
    partial_completion_min_count = math.ceil(partial_completion_ratio * repeat_per_prompt)
    fully_completed_indices = {
        record["dataset_index"]
        for record in prompt_records
        if len(rows_by_index.get(record["dataset_index"], [])) >= repeat_per_prompt
    }
    completed_indices = {
        record["dataset_index"]
        for record in prompt_records
        if (
            record["dataset_index"] in fully_completed_indices
            or (
                partial_completion_enabled
                and len(rows_by_index.get(record["dataset_index"], []))
                >= partial_completion_min_count
            )
        )
    }

    for row in finished_rows:
        if row["dataset_index"] in completed_indices:
            completed_rows.append(row)
        if row["dataset_index"] in fully_completed_indices:
            fully_completed_rows.append(row)

    for record in prompt_records:
        if record["dataset_index"] not in completed_indices:
            tail_prompt_records.append(
                {
                    **record,
                    "tail_round_id": round_id,
                    "tail_round_mode": round_mode,
                }
            )

    return (
        completed_rows,
        fully_completed_rows,
        tail_prompt_records,
        len(completed_indices),
        len(fully_completed_indices),
    )


def serialize_tail_prompts_for_output(
    prompt_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_index": record["dataset_index"],
            "data_source": record["data_source"],
            "ability": record["ability"],
            "extra_info": record["extra_info"],
            "prompt": record["messages"],
            "tail_round_id": record.get("tail_round_id"),
            "tail_round_mode": record.get("tail_round_mode"),
        }
        for record in prompt_records
    ]


def execute_round(
    *,
    llm: Any,
    sampling_params: Any,
    local_prompt_records: list[dict[str, Any]],
    repeat_per_prompt: int,
    enable_thinking: bool,
    round_id: int,
    round_mode: str,
    global_dp_rank: int,
    target_local_completed_prompts: int | None,
    round_label: str,
    partial_completion_enabled: bool,
    partial_completion_ratio: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    list[dict[str, Any]],
    int,
    int,
    list[str],
]:
    expanded_records = expand_prompt_records(local_prompt_records, repeat_per_prompt)
    if not expanded_records:
        return [], [], 0, 0, [], 0, 0, []

    prompts = llm.preprocess_chat(
        messages=[record["messages"] for record in expanded_records],
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    request_meta: dict[str, dict[str, Any]] = {}
    pending_request_ids: set[str] = set()
    for request_idx, (record, prompt) in enumerate(zip(expanded_records, prompts, strict=True)):
        request_id = f"round-{round_id}-rank-{global_dp_rank}-req-{request_idx}"
        request_meta[request_id] = record
        pending_request_ids.add(request_id)
        llm.llm_engine.add_request(request_id, prompt, sampling_params)

    finished_rows: list[dict[str, Any]] = []
    finished_request_count = 0
    finished_prompt_repeat_counts: dict[int, int] = {}
    completed_prompt_indices: set[int] = set()
    abort_issued = False

    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()
        for output in step_outputs:
            request_id = output.request_id

            if not output.finished:
                continue

            pending_request_ids.discard(request_id)

            first_output = output.outputs[0] if output.outputs else None
            finish_reason = first_output.finish_reason if first_output else "abort"
            if finish_reason == "abort":
                continue

            row = build_output_row(
                record=request_meta[request_id],
                output=output,
                enable_thinking=enable_thinking,
                round_id=round_id,
                round_mode=round_mode,
            )
            finished_rows.append(row)
            finished_request_count += 1
            dataset_index = row["dataset_index"]
            finished_prompt_repeat_counts[dataset_index] = (
                finished_prompt_repeat_counts.get(dataset_index, 0) + 1
            )
            if finished_prompt_repeat_counts[dataset_index] == repeat_per_prompt:
                completed_prompt_indices.add(dataset_index)

        if (
            round_mode == "redundant"
            and not abort_issued
            and target_local_completed_prompts is not None
            and len(completed_prompt_indices) >= target_local_completed_prompts
        ):
            if pending_request_ids:
                llm.llm_engine.abort_request(sorted(pending_request_ids))
            abort_issued = True

    if round_mode == "vanilla":
        completed_rows = finished_rows
        fully_completed_rows = finished_rows
        tail_prompt_records = []
        accepted_prompt_count = len(local_prompt_records)
        fully_completed_prompt_count = len(local_prompt_records)
    else:
        (
            completed_rows,
            fully_completed_rows,
            tail_prompt_records,
            accepted_prompt_count,
            fully_completed_prompt_count,
        ) = split_completed_and_tail_prompts(
            prompt_records=local_prompt_records,
            finished_rows=finished_rows,
            repeat_per_prompt=repeat_per_prompt,
            round_id=round_id,
            round_mode=round_mode,
            partial_completion_enabled=partial_completion_enabled,
            partial_completion_ratio=partial_completion_ratio,
        )
    total_tokens = sum(
        row["num_prompt_tokens"] + row["num_output_tokens"] for row in completed_rows
    )
    completed_tokens = sum(
        row["num_prompt_tokens"] + row["num_output_tokens"] for row in fully_completed_rows
    )
    req_log_lines = [
        (
            f"DP rank {global_dp_rank}: {round_label} "
            f"req dataset_index={row['dataset_index']} "
            f"repeat_idx={row['repeat_idx']} "
            f"prompt_tokens={row['num_prompt_tokens']} "
            f"response_tokens={row['num_output_tokens']} "
            f"total_tokens={row['num_prompt_tokens'] + row['num_output_tokens']}"
        )
        for row in sorted(
            finished_rows,
            key=lambda row: (row["dataset_index"], row["repeat_idx"]),
        )
    ]
    return (
        finished_rows,
        completed_rows,
        accepted_prompt_count,
        fully_completed_prompt_count,
        tail_prompt_records,
        total_tokens,
        completed_tokens,
        req_log_lines,
    )


def run_worker(
    *,
    model: str,
    records: list[dict[str, Any]],
    output_path: str,
    tail_output_path: str,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    tp_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
    enforce_eager: bool,
    trust_remote_code: bool,
    max_num_seqs: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    batch_size: int,
    redundancy_p: float,
    repeat_per_prompt: int,
    redundant_inference_enabled: bool,
    partial_completion_enabled: bool,
    partial_completion_ratio: float,
    round_barrier: Barrier,
    shared_control: Any,
    tail_prompt_queue: Any,
    batch_start_times: Any,
    batch_end_times: Any,
    batch_req_logs: Any,
    batch_total_tokens: Any,
    batch_completed_tokens: Any,
    batch_tail_prompts: Any,
    batch_finished_prompt_counts: Any,
    batch_fully_completed_prompt_counts: Any,
    batch_finished_request_counts: Any,
) -> None:
    from vllm import LLM, SamplingParams

    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    shard_output_path = Path(f"{output_path}.rank{global_dp_rank:02d}.jsonl")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if redundancy_p < 0:
        raise ValueError("redundancy_p must be non-negative")
    if repeat_per_prompt <= 0:
        raise ValueError("repeat_per_prompt must be positive")

    if shard_output_path.exists():
        shard_output_path.unlink()
    write_jsonl(shard_output_path, [])
    if local_dp_rank == 0:
        tail_output = Path(tail_output_path)
        if tail_output.exists():
            tail_output.unlink()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
    )
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )
    total = len(records)
    dataset_cursor = 0
    round_id = 0

    while True:
        if local_dp_rank == 0:
            for idx in range(len(batch_req_logs)):
                batch_req_logs[idx] = ""
                batch_total_tokens[idx] = 0
                batch_completed_tokens[idx] = 0
                batch_tail_prompts[idx] = []
                batch_finished_prompt_counts[idx] = 0
                batch_fully_completed_prompt_counts[idx] = 0
                batch_finished_request_counts[idx] = 0
                batch_start_times[idx] = 0.0
                batch_end_times[idx] = 0.0

            if not redundant_inference_enabled:
                if dataset_cursor < total:
                    round_prompt_count = min(total - dataset_cursor, batch_size)
                    selected_round_prompts = records[
                        dataset_cursor : dataset_cursor + round_prompt_count
                    ]
                    shared_control["round_mode"] = "vanilla"
                    shared_control["effective_base_prompt_count"] = len(selected_round_prompts)
                    shared_control["round_start"] = dataset_cursor
                    shared_control["round_end"] = dataset_cursor + round_prompt_count
                    dataset_cursor += round_prompt_count
                else:
                    selected_round_prompts = []
                    shared_control["round_mode"] = "done"
                    shared_control["effective_base_prompt_count"] = 0
                    shared_control["round_start"] = -1
                    shared_control["round_end"] = -1
            else:
                if len(tail_prompt_queue) >= batch_size:
                    selected_round_prompts = [
                        tail_prompt_queue.pop(0) for _ in range(batch_size)
                    ]
                    shared_control["round_mode"] = "tail_replay"
                    shared_control["effective_base_prompt_count"] = len(selected_round_prompts)
                    shared_control["round_start"] = -1
                    shared_control["round_end"] = -1
                elif dataset_cursor < total:
                    round_prompt_count = min(
                        total - dataset_cursor,
                        compute_redundant_prompt_count(batch_size, redundancy_p),
                    )
                    selected_round_prompts = records[
                        dataset_cursor : dataset_cursor + round_prompt_count
                    ]
                    shared_control["round_mode"] = "redundant"
                    shared_control["effective_base_prompt_count"] = min(
                        batch_size,
                        len(selected_round_prompts),
                    )
                    shared_control["round_start"] = dataset_cursor
                    shared_control["round_end"] = dataset_cursor + round_prompt_count
                    dataset_cursor += round_prompt_count
                elif len(tail_prompt_queue) > 0:
                    selected_round_prompts = [
                        tail_prompt_queue.pop(0) for _ in range(len(tail_prompt_queue))
                    ]
                    shared_control["round_mode"] = "tail_replay"
                    shared_control["effective_base_prompt_count"] = len(selected_round_prompts)
                    shared_control["round_start"] = -1
                    shared_control["round_end"] = -1
                else:
                    selected_round_prompts = []
                    shared_control["round_mode"] = "done"
                    shared_control["effective_base_prompt_count"] = 0
                    shared_control["round_start"] = -1
                    shared_control["round_end"] = -1

            shared_control["round_id"] = round_id
            shared_control["round_prompts"] = selected_round_prompts
            round_id += 1

        round_barrier.wait()

        current_round_id = int(shared_control["round_id"])
        round_mode = str(shared_control["round_mode"])
        round_prompts = list(shared_control["round_prompts"])
        effective_base_prompt_count = int(shared_control["effective_base_prompt_count"])
        round_start = int(shared_control["round_start"])
        round_end = int(shared_control["round_end"])

        if round_mode == "done":
            break

        local_start, local_end = even_shard_bounds(
            len(round_prompts),
            global_dp_rank,
            dp_size,
        )
        local_prompt_records = round_prompts[local_start:local_end]

        if round_mode == "redundant":
            base_start, base_end = even_shard_bounds(
                effective_base_prompt_count,
                global_dp_rank,
                dp_size,
            )
            target_local_completed_prompts = base_end - base_start
            round_label = (
                f"round_id={current_round_id} mode=redundant "
                f"dataset_slice=[{round_start}, {round_end})"
            )
        elif round_mode == "tail_replay":
            target_local_completed_prompts = None
            round_label = (
                f"round_id={current_round_id} mode=tail_replay "
                f"tail_prompts={len(round_prompts)}"
            )
        else:
            target_local_completed_prompts = None
            round_label = (
                f"round_id={current_round_id} mode=vanilla "
                f"dataset_slice=[{round_start}, {round_end})"
            )

        expanded_request_count = len(local_prompt_records) * repeat_per_prompt
        print(
            f"DP rank {global_dp_rank}: {round_label} "
            f"local_prompts={len(local_prompt_records)} "
            f"expanded_requests={expanded_request_count}"
        )

        batch_start_times[local_dp_rank] = perf_counter()
        (
            finished_rows,
            completed_rows,
            accepted_prompt_count,
            fully_completed_prompt_count,
            tail_prompt_records,
            local_total_tokens,
            local_completed_tokens,
            batch_log_lines,
        ) = execute_round(
            llm=llm,
            sampling_params=sampling_params,
            local_prompt_records=local_prompt_records,
            repeat_per_prompt=repeat_per_prompt,
            enable_thinking=enable_thinking,
            round_id=current_round_id,
            round_mode=round_mode,
            global_dp_rank=global_dp_rank,
            target_local_completed_prompts=target_local_completed_prompts,
            round_label=round_label,
            partial_completion_enabled=partial_completion_enabled,
            partial_completion_ratio=partial_completion_ratio,
        )
        batch_end_times[local_dp_rank] = perf_counter()

        if completed_rows:
            append_jsonl(shard_output_path, completed_rows)

        batch_req_logs[local_dp_rank] = "\n".join(batch_log_lines)
        batch_total_tokens[local_dp_rank] = local_total_tokens
        batch_completed_tokens[local_dp_rank] = local_completed_tokens
        batch_tail_prompts[local_dp_rank] = tail_prompt_records
        batch_finished_prompt_counts[local_dp_rank] = accepted_prompt_count
        batch_fully_completed_prompt_counts[local_dp_rank] = fully_completed_prompt_count
        batch_finished_request_counts[local_dp_rank] = len(finished_rows)

        round_barrier.wait()
        if local_dp_rank == 0:
            collected_tail_prompts: list[dict[str, Any]] = []
            for rank_tail_records in batch_tail_prompts[:]:
                collected_tail_prompts.extend(rank_tail_records)
            if collected_tail_prompts:
                tail_prompt_queue.extend(collected_tail_prompts)
                append_jsonl(
                    Path(tail_output_path),
                    serialize_tail_prompts_for_output(collected_tail_prompts),
                )

            for rank_log in batch_req_logs[:]:
                if rank_log:
                    print(rank_log)

            batch_elapsed_s = max(batch_end_times[:]) - min(batch_start_times[:])
            total_tokens = int(sum(batch_total_tokens[:]))
            completed_tokens = int(sum(batch_completed_tokens[:]))
            throughput_tokens_per_s = (
                total_tokens / batch_elapsed_s if batch_elapsed_s > 0 else 0.0
            )
            completed_throughput_tokens_per_s = (
                completed_tokens / batch_elapsed_s if batch_elapsed_s > 0 else 0.0
            )
            accepted_prompt_count = int(sum(batch_finished_prompt_counts[:]))
            fully_completed_prompt_count = int(sum(batch_fully_completed_prompt_counts[:]))
            finished_request_count = int(sum(batch_finished_request_counts[:]))
            print(
                f"{round_label} completed_prompts={fully_completed_prompt_count} "
                f"accepted_prompts={accepted_prompt_count} "
                f"finished_requests={finished_request_count} "
                f"collected_tail_prompts={len(collected_tail_prompts)} "
                f"tail_queue_size={len(tail_prompt_queue)} "
                f"total_tokens={total_tokens} "
                f"completed_tokens={completed_tokens} "
                f"elapsed_s={batch_elapsed_s:.3f} "
                f"throughput_tokens_per_s={throughput_tokens_per_s:.3f}"
                f" completed_throughput_tokens_per_s={completed_throughput_tokens_per_s:.3f}"
            )
        round_barrier.wait()

    sleep(1)


def merge_outputs(output_path: Path, dp_size: int) -> int:
    merged_rows: list[dict[str, Any]] = []
    for rank in range(dp_size):
        shard_path = Path(f"{output_path}.rank{rank:02d}.jsonl")
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard output: {shard_path}")
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                merged_rows.append(json.loads(line))

    merged_rows.sort(
        key=lambda row: (
            row["dataset_index"],
            row.get("round_id", -1),
            row.get("repeat_idx", -1),
        )
    )
    write_jsonl(output_path, merged_rows)
    return len(merged_rows)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path)
    dataset_indices = (
        [int(item.strip()) for item in args.dataset_indices.split(",") if item.strip()]
        if args.dataset_indices
        else None
    )
    redundant_inference_enabled = parse_env_flag(
        "REDUNDANT_INFERENCE_ENABLED",
        default=True,
    )
    partial_completion_enabled = parse_env_flag(
        "TAIL_PARTIAL_COMPLETE_ENABLED",
        default=False,
    )
    partial_completion_ratio = parse_env_float(
        "TAIL_PARTIAL_COMPLETE_RATIO",
        default=0.6,
    )

    if args.node_size == 1:
        from vllm.utils import get_open_port

        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    if args.dp_size % args.node_size != 0:
        raise ValueError("dp-size must be divisible by node-size")

    if args.dp_size <= 0 or args.tp_size <= 0:
        raise ValueError("dp-size and tp-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.redundancy_p < 0:
        raise ValueError("redundancy-p must be non-negative")
    if args.repeat_per_prompt <= 0:
        raise ValueError("repeat-per-prompt must be positive")
    if not 0.0 <= partial_completion_ratio <= 1.0:
        raise ValueError("TAIL_PARTIAL_COMPLETE_RATIO must be in [0, 1]")
    if args.node_size != 1:
        raise ValueError(
            "Sequential global batching currently supports single-node DP only."
        )

    temperature, top_p, top_k, min_p = resolve_sampling_config(
        args.enable_thinking,
        args.temperature,
        args.top_p,
        args.top_k,
        args.min_p,
    )
    dp_per_node = args.dp_size // args.node_size
    records = load_records(
        args.dataset,
        args.prompt_column,
        args.limit,
        dataset_indices=dataset_indices,
    )
    if not records:
        raise ValueError("No records loaded from dataset.")

    mode = "thinking" if args.enable_thinking else "nothinking"
    print(
        f"Qwen3 mode={mode}, temperature={temperature}, top_p={top_p}, "
        f"top_k={top_k}, min_p={min_p}"
    )
    print(
        f"Sequential batching enabled: base_batch_size={args.batch_size}, "
        f"redundancy_p={args.redundancy_p}, "
        f"repeat_per_prompt={args.repeat_per_prompt}."
    )
    print(f"Redundant inference enabled: {redundant_inference_enabled}")
    print(
        "Tail partial completion: "
        f"enabled={partial_completion_enabled}, "
        f"ratio={partial_completion_ratio}"
    )

    round_barrier = Barrier(dp_per_node)
    batch_start_times = Array("d", dp_per_node)
    batch_end_times = Array("d", dp_per_node)
    batch_total_tokens = Array("q", dp_per_node)
    batch_completed_tokens = Array("q", dp_per_node)
    manager = Manager()
    tail_output_path = Path(f"{output_path}.tail_prompts.jsonl")
    if output_path.exists():
        output_path.unlink()
    if tail_output_path.exists():
        tail_output_path.unlink()
    shared_control = manager.dict(
        {
            "round_id": 0,
            "round_mode": "done",
            "round_prompts": [],
            "effective_base_prompt_count": 0,
            "round_start": -1,
            "round_end": -1,
        }
    )
    tail_prompt_queue = manager.list()
    batch_req_logs = manager.list([""] * dp_per_node)
    batch_tail_prompts = manager.list([[] for _ in range(dp_per_node)])
    batch_finished_prompt_counts = Array("i", dp_per_node)
    batch_fully_completed_prompt_counts = Array("i", dp_per_node)
    batch_finished_request_counts = Array("i", dp_per_node)
    procs: list[Process] = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(args.node_rank * dp_per_node, (args.node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=run_worker,
            kwargs={
                "model": args.model,
                "records": records,
                "output_path": str(output_path),
                "tail_output_path": str(tail_output_path),
                "dp_size": args.dp_size,
                "local_dp_rank": local_dp_rank,
                "global_dp_rank": global_dp_rank,
                "dp_master_ip": dp_master_ip,
                "dp_master_port": dp_master_port,
                "tp_size": args.tp_size,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "enable_thinking": args.enable_thinking,
                "enforce_eager": args.enforce_eager,
                "trust_remote_code": args.trust_remote_code,
                "max_num_seqs": args.max_num_seqs,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "batch_size": args.batch_size,
                "redundancy_p": args.redundancy_p,
                "repeat_per_prompt": args.repeat_per_prompt,
                "redundant_inference_enabled": redundant_inference_enabled,
                "partial_completion_enabled": partial_completion_enabled,
                "partial_completion_ratio": partial_completion_ratio,
                "round_barrier": round_barrier,
                "shared_control": shared_control,
                "tail_prompt_queue": tail_prompt_queue,
                "batch_start_times": batch_start_times,
                "batch_end_times": batch_end_times,
                "batch_req_logs": batch_req_logs,
                "batch_total_tokens": batch_total_tokens,
                "batch_completed_tokens": batch_completed_tokens,
                "batch_tail_prompts": batch_tail_prompts,
                "batch_finished_prompt_counts": batch_finished_prompt_counts,
                "batch_fully_completed_prompt_counts": batch_fully_completed_prompt_counts,
                "batch_finished_request_counts": batch_finished_request_counts,
            },
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that exceeded timeout.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    if exit_code == 0 and args.node_size == 1:
        merged = merge_outputs(output_path, args.dp_size)
        print(f"Merged {merged} results into {output_path}")
    elif exit_code == 0:
        print(
            "All local DP workers finished. Multi-node output merging is not "
            "handled automatically by this script."
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
