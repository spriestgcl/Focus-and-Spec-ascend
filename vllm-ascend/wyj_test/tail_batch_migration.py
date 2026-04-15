#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Array, Barrier, Lock, Manager, Process
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from dp_inference import (append_jsonl, ensure_parent_dir, even_shard_bounds,
                          load_records, parse_env_flag, resolve_sampling_config,
                          write_jsonl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tail-batch DP inference with script-level request migration."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-indices", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--prompt-column", type=str, default="prompt")
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--enable-thinking", dest="enable_thinking",
                        action="store_true")
    parser.add_argument("--disable-thinking", dest="enable_thinking",
                        action="store_false")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--repeat-per-prompt", type=int, default=8)
    parser.add_argument("--migration-min-total-tokens", type=int, default=2000)
    parser.add_argument(
        "--migration-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable script-level request migration. Use --no-migration-enabled to disable.",
    )
    parser.add_argument(
        "--tail-concentrate-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When migration is disabled, optionally concentrate the longest requests onto specific ranks to simulate stragglers.",
    )
    parser.add_argument(
        "--tail-concentrate-top-k",
        type=int,
        default=0,
        help="Number of longest requests (by prompt length) to concentrate.",
    )
    parser.add_argument(
        "--tail-concentrate-ranks",
        type=str,
        default="0,1",
        help="Comma-separated DP ranks that will receive the concentrated long requests (e.g. '0,1').",
    )
    parser.add_argument(
        "--tail-concentrate-request-keys",
        type=str,
        default="",
        help="Comma-separated request keys (dataset_index-repeat_idx) to concentrate onto target ranks.",
    )
    parser.add_argument(
        "--sync-every-steps",
        type=int,
        default=64,
        help="All ranks synchronize every N engine.step() calls (or earlier if a rank runs out of requests).",
    )
    parser.add_argument(
        "--report-every-decode-steps",
        type=int,
        default=1024,
        help="Rank-0 prints a progress line every N engine.step() decode iterations.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def serialize_request_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_key": state["request_key"],
        "dataset_index": state["dataset_index"],
        "repeat_idx": state["repeat_idx"],
        "data_source": state.get("data_source"),
        "ability": state.get("ability"),
        "reward_model": state.get("reward_model"),
        "extra_info": state.get("extra_info"),
        "messages": state["messages"],
        "base_prompt_token_ids": list(state["base_prompt_token_ids"]),
        "generated_token_ids": list(state["generated_token_ids"]),
        "migration_count": int(state["migration_count"]),
    }


def deserialize_request_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "base_prompt_token_ids": list(state["base_prompt_token_ids"]),
        "generated_token_ids": list(state["generated_token_ids"]),
        "migration_count": int(state["migration_count"]),
    }


def build_initial_request_states(
    records: list[dict[str, Any]],
    repeat_per_prompt: int,
    prompt_token_ids_list: list[list[int]],
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for record, prompt_token_ids in zip(records, prompt_token_ids_list, strict=True):
        for repeat_idx in range(repeat_per_prompt):
            states.append(
                {
                    "request_key": f"{record['dataset_index']}-{repeat_idx}",
                    "dataset_index": record["dataset_index"],
                    "repeat_idx": repeat_idx,
                    "data_source": record.get("data_source"),
                    "ability": record.get("ability"),
                    "reward_model": record.get("reward_model"),
                    "extra_info": record.get("extra_info"),
                    "messages": record["messages"],
                    "base_prompt_token_ids": list(prompt_token_ids),
                    "generated_token_ids": [],
                    "migration_count": 0,
                }
            )
    return states


def rebalance_request_states(
    unfinished_states: list[dict[str, Any]],
    dp_size: int,
) -> list[dict[str, Any]]:
    sorted_states = sorted(
        unfinished_states,
        key=lambda state: (
            -len(state["generated_token_ids"]),
            state["dataset_index"],
            state["repeat_idx"],
        ),
    )
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(dp_size)]
    bucket_loads = [0] * dp_size
    for state in sorted_states:
        target_rank = min(range(dp_size), key=lambda idx: bucket_loads[idx])
        buckets[target_rank].append(state)
        bucket_loads[target_rank] += 1
    flat_states: list[dict[str, Any]] = []
    for bucket in buckets:
        flat_states.extend(bucket)
    return flat_states


def parse_rank_list(value: str) -> list[int]:
    items = [item.strip() for item in (value or "").split(",") if item.strip()]
    return [int(item) for item in items]


def parse_request_key_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def concentrate_long_requests_to_ranks(
    *,
    stage_requests: list[dict[str, Any]],
    dp_size: int,
    target_ranks: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not stage_requests:
        return stage_requests
    target_ranks = [r for r in target_ranks if 0 <= r < dp_size]
    if not target_ranks:
        return stage_requests

    desired_sizes = [
        even_shard_bounds(len(stage_requests), rank, dp_size)[1]
        - even_shard_bounds(len(stage_requests), rank, dp_size)[0]
        for rank in range(dp_size)
    ]

    # Pick the longest requests by prompt length (base prompt tokens).
    sorted_states = sorted(
        stage_requests,
        key=lambda s: (len(s.get("base_prompt_token_ids", [])), s.get("dataset_index", 0), s.get("repeat_idx", 0)),
        reverse=True,
    )
    selected = sorted_states[: min(top_k, len(sorted_states))]
    selected_keys = {s["request_key"] for s in selected}
    remaining = [s for s in stage_requests if s.get("request_key") not in selected_keys]

    buckets: list[list[dict[str, Any]]] = [[] for _ in range(dp_size)]
    # Distribute selected long requests across the target ranks.
    for i, state in enumerate(selected):
        buckets[target_ranks[i % len(target_ranks)]].append(state)

    # Fill each rank's bucket up to its desired shard size.
    cursor = 0
    for rank in range(dp_size):
        need = max(desired_sizes[rank] - len(buckets[rank]), 0)
        if need == 0:
            continue
        buckets[rank].extend(remaining[cursor : cursor + need])
        cursor += need

    # Any leftovers (shouldn't happen) go to the last rank.
    if cursor < len(remaining):
        buckets[-1].extend(remaining[cursor:])

    flattened: list[dict[str, Any]] = []
    for rank in range(dp_size):
        flattened.extend(buckets[rank])
    return flattened


def concentrate_specific_requests_to_ranks(
    *,
    stage_requests: list[dict[str, Any]],
    dp_size: int,
    target_ranks: list[int],
    request_keys: list[str],
) -> list[dict[str, Any]]:
    if not stage_requests or not request_keys:
        return stage_requests
    target_ranks = [r for r in target_ranks if 0 <= r < dp_size]
    if not target_ranks:
        return stage_requests

    desired_sizes = [
        even_shard_bounds(len(stage_requests), rank, dp_size)[1]
        - even_shard_bounds(len(stage_requests), rank, dp_size)[0]
        for rank in range(dp_size)
    ]

    key_set = set(request_keys)
    selected = [s for s in stage_requests if s.get("request_key") in key_set]
    selected_keys = {s["request_key"] for s in selected}
    remaining = [s for s in stage_requests if s.get("request_key") not in selected_keys]

    # Keep deterministic order for selected requests.
    selected.sort(key=lambda s: (s.get("dataset_index", 0), s.get("repeat_idx", 0)))

    buckets: list[list[dict[str, Any]]] = [[] for _ in range(dp_size)]
    for i, state in enumerate(selected):
        buckets[target_ranks[i % len(target_ranks)]].append(state)

    cursor = 0
    for rank in range(dp_size):
        need = max(desired_sizes[rank] - len(buckets[rank]), 0)
        if need == 0:
            continue
        buckets[rank].extend(remaining[cursor : cursor + need])
        cursor += need

    if cursor < len(remaining):
        buckets[-1].extend(remaining[cursor:])

    flattened: list[dict[str, Any]] = []
    for rank in range(dp_size):
        flattened.extend(buckets[rank])
    return flattened


def build_output_row(
    *,
    state: dict[str, Any],
    output_token_ids: list[int],
    tokenizer: Any,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "dataset_index": state["dataset_index"],
        "data_source": state.get("data_source"),
        "ability": state.get("ability"),
        "reward_model": state.get("reward_model"),
        "extra_info": state.get("extra_info"),
        "prompt": state["messages"],
        "response": tokenizer.decode(output_token_ids, skip_special_tokens=True),
        "finish_reason": "length_or_stop",
        "num_prompt_tokens": len(state["base_prompt_token_ids"]),
        "num_output_tokens": len(output_token_ids),
        "enable_thinking": enable_thinking,
        "repeat_idx": state["repeat_idx"],
        "migration_count": state["migration_count"],
    }


def should_trigger_migration(
    *,
    total_unfinished: int,
    unfinished_counts: list[int],
    max_total_tokens: int,
    migration_min_total_tokens: int,
) -> bool:
    _ = unfinished_counts
    if total_unfinished <= 0:
        return False
    return max_total_tokens >= migration_min_total_tokens


def execute_stage(
    *,
    llm: Any,
    tokenizer: Any,
    local_states: list[dict[str, Any]],
    global_dp_rank: int,
    stage_id: int,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    max_tokens: int,
    dp_size: int,
    migration_min_total_tokens: int,
    migration_enabled: bool,
    sync_every_steps: int,
    report_every_decode_steps: int,
    stage_barrier: Barrier,
    control_lock: Lock,
    shared_control: Any,
    shared_unfinished_counts: Any,
    shared_rank_max_total_tokens: Any,
    shared_rank_window_steps: Any,
    shared_rank_step_totals: Any,
    shared_rank_decode_elapsed_ms: Any,
    shared_rank_barrier1_wait_ms: Any,
    shared_rank_barrier2_wait_ms: Any,
    shared_rank_snapshot_elapsed_ms: Any,
    shared_rank_barrier3_wait_ms: Any,
    shared_rank_barrier4_wait_ms: Any,
    shared_snapshot_ready: Any,
    shared_unfinished_states: Any,
    shared_stage_has_work: Any,
    shared_stage_first_token_ms: Any,
) -> tuple[list[dict[str, Any]], bool]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    def make_sampling_params(remaining_max_tokens: int) -> Any:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max(remaining_max_tokens, 1),
            output_kind=RequestOutputKind.CUMULATIVE,
        )

    request_states: dict[str, dict[str, Any]] = {}
    pending_request_ids: set[str] = set()
    final_rows: list[dict[str, Any]] = []
    stage_start_time = perf_counter()
    local_first_token_ms = -1.0

    for local_idx, state in enumerate(local_states):
        prompt_token_ids = state["base_prompt_token_ids"] + state["generated_token_ids"]
        remaining_max_tokens = max_tokens - len(state["generated_token_ids"])
        if remaining_max_tokens <= 0:
            final_rows.append(
                build_output_row(
                    state=state,
                    output_token_ids=list(state["generated_token_ids"]),
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                )
            )
            continue
        request_id = f"stage-{stage_id}-rank-{global_dp_rank}-req-{local_idx}"
        request_states[request_id] = {
            **deserialize_request_state(state),
            "_stage_token_ids_ref": [],
            "_stage_token_len": 0,
        }
        pending_request_ids.add(request_id)
        llm.llm_engine.add_request(
            request_id,
            {"prompt_token_ids": prompt_token_ids},
            make_sampling_params(remaining_max_tokens),
        )

    local_stage_has_work = 1 if pending_request_ids else 0
    shared_stage_has_work[global_dp_rank] = local_stage_has_work
    shared_stage_first_token_ms[global_dp_rank] = local_first_token_ms

    if sync_every_steps <= 0:
        sync_every_steps = 64
    if report_every_decode_steps <= 0:
        report_every_decode_steps = 1024
    poll_sleep_s = 0.001
    local_step_total = 0
    local_window_steps = 0
    local_window_decode_ms = 0.0
    local_snapshot_sent = False

    def compute_local_stats() -> tuple[int, int]:
        local_unfinished_count = len(pending_request_ids)
        local_rank_max_total_tokens = 0
        if pending_request_ids:
            total_tokens_list = []
            for request_id in pending_request_ids:
                state = request_states[request_id]
                current_decode_pos = (
                    len(state["generated_token_ids"])
                    + int(state.get("_stage_token_len", 0))
                )
                total_tokens_list.append(
                    len(state["base_prompt_token_ids"]) + current_decode_pos
                )
            local_rank_max_total_tokens = max(total_tokens_list)
        return local_unfinished_count, local_rank_max_total_tokens

    def publish_status() -> tuple[int, int]:
        nonlocal local_window_steps, local_window_decode_ms
        local_unfinished_count, local_rank_max_total_tokens = compute_local_stats()
        shared_unfinished_counts[global_dp_rank] = local_unfinished_count
        shared_rank_max_total_tokens[global_dp_rank] = int(local_rank_max_total_tokens)
        shared_rank_window_steps[global_dp_rank] = int(local_window_steps)
        shared_rank_step_totals[global_dp_rank] = int(local_step_total)
        shared_rank_decode_elapsed_ms[global_dp_rank] = float(local_window_decode_ms)
        shared_rank_barrier1_wait_ms[global_dp_rank] = 0.0
        shared_rank_barrier2_wait_ms[global_dp_rank] = 0.0
        shared_rank_barrier3_wait_ms[global_dp_rank] = 0.0
        shared_rank_barrier4_wait_ms[global_dp_rank] = 0.0
        shared_stage_has_work[global_dp_rank] = local_stage_has_work
        shared_stage_first_token_ms[global_dp_rank] = local_first_token_ms
        return local_unfinished_count, local_rank_max_total_tokens

    def maybe_log_prefill_ttft() -> None:
        if global_dp_rank != 0 or stage_id <= 0:
            return
        if bool(shared_control.get("stage_prefill_logged", False)):
            return
        active_ranks = [idx for idx in range(dp_size) if int(shared_stage_has_work[idx]) > 0]
        if not active_ranks:
            return
        if not all(float(shared_stage_first_token_ms[idx]) >= 0.0 for idx in active_ranks):
            return
        active_rank_set = set(active_ranks)
        per_rank_ttft_parts = []
        active_ttfts = []
        for idx in range(dp_size):
            if idx in active_rank_set:
                ttft_ms = float(shared_stage_first_token_ms[idx])
                active_ttfts.append(ttft_ms)
                per_rank_ttft_parts.append(f"{ttft_ms:.3f}")
            else:
                per_rank_ttft_parts.append("na")
        print(
            f"stage_id={stage_id} migration_prefill_ttft_ms={max(active_ttfts):.3f} "
            f"per_rank_ttft_ms=[{','.join(per_rank_ttft_parts)}]",
            flush=True,
        )
        shared_control["stage_prefill_logged"] = True

    def maybe_trigger_collect_from_any_rank(
        local_unfinished_count: int,
        local_rank_max_total_tokens: int,
    ) -> None:
        if not migration_enabled or stage_id != 0:
            return
        if str(shared_control.get("command", "run")) != "run":
            return
        trigger_idle = (local_unfinished_count == 0)
        trigger_threshold = (local_rank_max_total_tokens >= migration_min_total_tokens)
        if not (trigger_idle or trigger_threshold):
            return
        with control_lock:
            if (
                int(shared_control.get("stage_id", stage_id)) != stage_id
                or str(shared_control.get("command", "run")) != "run"
            ):
                return
            unfinished_counts = list(shared_unfinished_counts[:])
            total_unfinished = int(sum(unfinished_counts))
            if total_unfinished <= 0:
                shared_control["command"] = "done"
                shared_control["all_done"] = True
                return
            max_total_tokens = int(max(shared_rank_max_total_tokens[:])) if total_unfinished > 0 else 0
            shared_control["command"] = "collect"
            shared_control["before_counts"] = unfinished_counts
            shared_control["trigger_max_total_tokens"] = max_total_tokens
            shared_control["trigger_reason"] = (
                "idle_rank_and_threshold"
                if trigger_idle and trigger_threshold
                else ("idle_rank" if trigger_idle else "threshold")
            )
            for idx in range(dp_size):
                shared_snapshot_ready[idx] = 0
            print(
                f"stage_id={stage_id} trigger_collect "
                f"reason={shared_control['trigger_reason']} "
                f"trigger_rank={global_dp_rank} "
                f"unfinished_counts={unfinished_counts} "
                f"max_total_tokens={max_total_tokens}",
                flush=True,
            )

    def maybe_update_progress_and_trigger() -> None:
        if global_dp_rank != 0:
            return
        maybe_log_prefill_ttft()
        total_unfinished = int(sum(shared_unfinished_counts[:]))
        stage_decode_base = int(shared_control.get("stage_decode_base", 0))
        global_decode_pos = stage_decode_base + int(max(shared_rank_step_totals[:], default=0))
        sample_prev_total_steps = int(
            shared_control.get("sample_prev_total_steps", stage_decode_base)
        )
        sample_prev_time = float(shared_control.get("sample_prev_time", perf_counter()))
        sample_now_time = perf_counter()
        delta_steps = max(global_decode_pos - sample_prev_total_steps, 0)
        delta_time_s = max(sample_now_time - sample_prev_time, 0.0)
        progress_last_pos = int(shared_control.get("progress_last_pos", 0))
        progress_accum_steps = float(shared_control.get("progress_accum_steps", 0.0))
        progress_accum_elapsed_s = float(
            shared_control.get("progress_accum_elapsed_s", 0.0)
        )
        if delta_steps > 0:
            progress_accum_steps += float(delta_steps)
            progress_accum_elapsed_s += delta_time_s
            while progress_accum_steps >= float(report_every_decode_steps):
                consume_steps = float(report_every_decode_steps)
                consume_ratio = consume_steps / progress_accum_steps
                consume_elapsed_s = progress_accum_elapsed_s * consume_ratio
                step_ms = (consume_elapsed_s * 1000.0) / consume_steps
                progress_last_pos += report_every_decode_steps
                print(
                    "decode_progress "
                    f"decode_step={progress_last_pos} "
                    f"remaining_reqs={total_unfinished} "
                    f"last_{report_every_decode_steps}_decode_ms={step_ms:.3f}",
                    flush=True,
                )
                progress_accum_steps -= consume_steps
                progress_accum_elapsed_s -= consume_elapsed_s
        shared_control["progress_last_pos"] = progress_last_pos
        shared_control["progress_accum_steps"] = progress_accum_steps
        shared_control["progress_accum_elapsed_s"] = progress_accum_elapsed_s
        shared_control["sample_prev_total_steps"] = global_decode_pos
        shared_control["sample_prev_time"] = sample_now_time
        if total_unfinished == 0:
            with control_lock:
                if (
                    int(shared_control.get("stage_id", stage_id)) == stage_id
                    and str(shared_control.get("command", "run")) == "run"
                ):
                    shared_control["command"] = "done"
                    shared_control["all_done"] = True

    while True:
        command = str(shared_control.get("command", "run"))
        if int(shared_control.get("stage_id", stage_id)) != stage_id:
            return final_rows, True

        if command == "run":
            if llm.llm_engine.has_unfinished_requests():
                step_start = perf_counter()
                step_outputs = llm.llm_engine.step()
                local_window_decode_ms += (perf_counter() - step_start) * 1000.0
                local_window_steps += 1
                local_step_total += 1
                for output in step_outputs:
                    request_id = output.request_id
                    state = request_states.get(request_id)
                    if state is None:
                        continue
                    first_output = output.outputs[0] if output.outputs else None
                    if first_output is not None:
                        token_ids = first_output.token_ids or []
                        token_len = len(token_ids)
                        state["_stage_token_ids_ref"] = token_ids
                        state["_stage_token_len"] = token_len
                        if token_len > 0 and local_first_token_ms < 0.0:
                            local_first_token_ms = (perf_counter() - stage_start_time) * 1000.0
                            shared_stage_first_token_ms[global_dp_rank] = local_first_token_ms
                    if not output.finished:
                        continue
                    pending_request_ids.discard(request_id)
                    finish_reason = first_output.finish_reason if first_output else "abort"
                    if finish_reason == "abort":
                        request_states.pop(request_id, None)
                        continue
                    full_output_token_ids = (
                        list(state["generated_token_ids"])
                        + list(state["_stage_token_ids_ref"] or [])
                    )
                    final_rows.append(
                        build_output_row(
                            state=state,
                            output_token_ids=full_output_token_ids,
                            tokenizer=tokenizer,
                            enable_thinking=enable_thinking,
                        )
                    )
                    request_states.pop(request_id, None)
                if local_window_steps >= sync_every_steps or not pending_request_ids:
                    local_unfinished_count, local_rank_max_total_tokens = publish_status()
                    maybe_trigger_collect_from_any_rank(
                        local_unfinished_count,
                        local_rank_max_total_tokens,
                    )
                    maybe_update_progress_and_trigger()
                    if global_dp_rank == 0:
                        print(
                            "async_profile "
                            f"stage_id={stage_id} "
                            f"rank={global_dp_rank} "
                            f"window_steps={local_window_steps} "
                            f"step_total={local_step_total} "
                            f"unfinished_local={len(pending_request_ids)} "
                            f"decode_ms={local_window_decode_ms:.3f}",
                            flush=True,
                        )
                    local_window_steps = 0
                    local_window_decode_ms = 0.0
            else:
                local_unfinished_count, local_rank_max_total_tokens = publish_status()
                maybe_trigger_collect_from_any_rank(
                    local_unfinished_count,
                    local_rank_max_total_tokens,
                )
                maybe_update_progress_and_trigger()
                sleep(poll_sleep_s)
        elif command == "collect":
            local_unfinished_count, local_rank_max_total_tokens = publish_status()
            maybe_trigger_collect_from_any_rank(
                local_unfinished_count,
                local_rank_max_total_tokens,
            )
            maybe_update_progress_and_trigger()
            if not local_snapshot_sent:
                snapshot_start = perf_counter()
                unfinished_snapshot: list[dict[str, Any]] = []
                for request_id in sorted(pending_request_ids):
                    state = request_states[request_id]
                    unfinished_snapshot.append(
                        serialize_request_state(
                            {
                                **state,
                                "generated_token_ids": (
                                    list(state["generated_token_ids"])
                                    + list(state["_stage_token_ids_ref"] or [])
                                ),
                            }
                        )
                    )
                shared_unfinished_states[global_dp_rank] = unfinished_snapshot
                shared_rank_snapshot_elapsed_ms[global_dp_rank] = (
                    perf_counter() - snapshot_start
                ) * 1000.0
                shared_snapshot_ready[global_dp_rank] = 1
                local_snapshot_sent = True
            if global_dp_rank == 0:
                ready_flags = list(shared_snapshot_ready[:])
                if all(flag == 1 for flag in ready_flags):
                    rebalance_start = perf_counter()
                    all_unfinished_states: list[dict[str, Any]] = []
                    for rank_states in shared_unfinished_states[:]:
                        all_unfinished_states.extend(rank_states)
                    before_counts = list(shared_unfinished_counts[:])
                    if all_unfinished_states:
                        bumped_states: list[dict[str, Any]] = []
                        for state in all_unfinished_states:
                            bumped_states.append(
                                {
                                    **state,
                                    "migration_count": int(state.get("migration_count", 0)) + 1,
                                }
                            )
                        next_stage = rebalance_request_states(
                            bumped_states,
                            dp_size=dp_size,
                        )
                        after_counts = [
                            even_shard_bounds(len(next_stage), rank, dp_size)[1]
                            - even_shard_bounds(len(next_stage), rank, dp_size)[0]
                            for rank in range(dp_size)
                        ]
                        shared_control["next_stage_requests"] = next_stage
                        shared_control["migrate"] = True
                        shared_control["all_done"] = False
                        shared_control["command"] = "migrate"
                        reset_time = perf_counter()
                        shared_control["progress_accum_steps"] = 0.0
                        shared_control["progress_accum_elapsed_s"] = 0.0
                        shared_control["sample_prev_time"] = reset_time
                        shared_control["sample_prev_total_steps"] = int(
                            shared_control.get("stage_decode_base", 0)
                        ) + int(max(shared_rank_step_totals[:], default=0))
                        shared_control["migration_message"] = (
                            f"stage_id={stage_id} migrate unfinished_reqs={len(all_unfinished_states)} "
                            f"before_counts={before_counts} "
                            f"after_counts={after_counts} "
                            f"trigger_max_total_tokens={int(shared_control.get('trigger_max_total_tokens', 0))} "
                            f"threshold={migration_min_total_tokens} "
                            f"reason={str(shared_control.get('trigger_reason', 'unknown'))} "
                            f"rebalance_ms={(perf_counter() - rebalance_start) * 1000.0:.3f}"
                        )
                        print(str(shared_control["migration_message"]), flush=True)
                    else:
                        shared_control["command"] = "done"
                        shared_control["all_done"] = True
            next_command = str(shared_control.get("command", "collect"))
            if next_command == "migrate":
                if pending_request_ids:
                    llm.llm_engine.abort_request(sorted(pending_request_ids))
                while llm.llm_engine.has_unfinished_requests():
                    llm.llm_engine.step()
                return final_rows, True
            if next_command == "done":
                break
            sleep(poll_sleep_s)
        elif command == "migrate":
            if pending_request_ids:
                llm.llm_engine.abort_request(sorted(pending_request_ids))
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()
            return final_rows, True
        elif command == "done" or bool(shared_control.get("all_done", False)):
            break
        else:
            sleep(poll_sleep_s)

    return final_rows, False


def run_worker(
    *,
    model: str,
    records: list[dict[str, Any]],
    output_path: str,
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
    repeat_per_prompt: int,
    migration_min_total_tokens: int,
    migration_enabled: bool,
    tail_concentrate_enabled: bool,
    tail_concentrate_top_k: int,
    tail_concentrate_ranks: str,
    tail_concentrate_request_keys: str,
    sync_every_steps: int,
    report_every_decode_steps: int,
    stage_barrier: Barrier,
    control_lock: Lock,
    shared_control: Any,
    shared_unfinished_counts: Any,
    shared_rank_max_total_tokens: Any,
    shared_rank_window_steps: Any,
    shared_rank_step_totals: Any,
    shared_rank_decode_elapsed_ms: Any,
    shared_rank_barrier1_wait_ms: Any,
    shared_rank_barrier2_wait_ms: Any,
    shared_rank_snapshot_elapsed_ms: Any,
    shared_rank_barrier3_wait_ms: Any,
    shared_rank_barrier4_wait_ms: Any,
    shared_snapshot_ready: Any,
    shared_unfinished_states: Any,
    shared_stage_has_work: Any,
    shared_stage_first_token_ms: Any,
    batch_elapsed_times: Any,
) -> None:
    from vllm import LLM

    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    shard_output_path = Path(f"{output_path}.rank{global_dp_rank:02d}.jsonl")
    if shard_output_path.exists():
        shard_output_path.unlink()
    write_jsonl(shard_output_path, [])

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
    tokenizer = llm.get_tokenizer()

    if global_dp_rank == 0:
        initial_prompts = llm.preprocess_chat(
            messages=[record["messages"] for record in records],
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        prompt_token_ids_list = [
            list(prompt["prompt_token_ids"]) for prompt in initial_prompts
        ]
        shared_control["stage_requests"] = build_initial_request_states(
            records,
            repeat_per_prompt=repeat_per_prompt,
            prompt_token_ids_list=prompt_token_ids_list,
        )
        if (not migration_enabled) and tail_concentrate_enabled and tail_concentrate_top_k > 0:
            target_ranks = parse_rank_list(tail_concentrate_ranks)
            explicit_keys = parse_request_key_list(tail_concentrate_request_keys)
            if explicit_keys:
                shared_control["stage_requests"] = concentrate_specific_requests_to_ranks(
                    stage_requests=list(shared_control["stage_requests"]),
                    dp_size=dp_size,
                    target_ranks=target_ranks,
                    request_keys=explicit_keys,
                )
            else:
                shared_control["stage_requests"] = concentrate_long_requests_to_ranks(
                    stage_requests=list(shared_control["stage_requests"]),
                    dp_size=dp_size,
                    target_ranks=target_ranks,
                    top_k=tail_concentrate_top_k,
                )
            per_rank_counts = [
                even_shard_bounds(len(shared_control["stage_requests"]), rank, dp_size)[1]
                - even_shard_bounds(len(shared_control["stage_requests"]), rank, dp_size)[0]
                for rank in range(dp_size)
            ]
            print(
                "tail_concentrate "
                f"top_k={tail_concentrate_top_k} "
                f"target_ranks={target_ranks} "
                f"explicit_keys={len(explicit_keys)} "
                f"per_rank_counts={per_rank_counts}",
                flush=True,
            )
        shared_control["stage_id"] = 0
        shared_control["done"] = False
        shared_control["command"] = "run"
        shared_control["migration_message"] = ""
        # Initialize decode-step progress logger state once per run.
        shared_control.setdefault("progress_last_pos", 0)
        shared_control.setdefault("progress_accum_steps", 0.0)
        shared_control.setdefault("progress_accum_elapsed_s", 0.0)
        shared_control.setdefault("decode_step_total", 0)
        shared_control.setdefault("stage_decode_base", 0)
        shared_control.setdefault("sample_prev_total_steps", 0)
        shared_control.setdefault("sample_prev_time", perf_counter())
        shared_control.setdefault("stage_prefill_logged", False)
        for idx in range(dp_size):
            shared_snapshot_ready[idx] = 0
            shared_rank_step_totals[idx] = 0
    else:
        while "stage_requests" not in shared_control:
            sleep(0.01)
    batch_start = perf_counter()

    while True:
        stage_requests = [deserialize_request_state(state) for state in shared_control["stage_requests"]]
        if not stage_requests:
            break
        stage_id = int(shared_control["stage_id"])
        local_start, local_end = even_shard_bounds(len(stage_requests), global_dp_rank, dp_size)
        local_states = stage_requests[local_start:local_end]

        final_rows, migrated = execute_stage(
            llm=llm,
            tokenizer=tokenizer,
            local_states=local_states,
            global_dp_rank=global_dp_rank,
            stage_id=stage_id,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            dp_size=dp_size,
            migration_min_total_tokens=migration_min_total_tokens,
            migration_enabled=migration_enabled,
            sync_every_steps=sync_every_steps,
            report_every_decode_steps=report_every_decode_steps,
            stage_barrier=stage_barrier,
            control_lock=control_lock,
            shared_control=shared_control,
            shared_unfinished_counts=shared_unfinished_counts,
            shared_rank_max_total_tokens=shared_rank_max_total_tokens,
            shared_rank_window_steps=shared_rank_window_steps,
            shared_rank_step_totals=shared_rank_step_totals,
            shared_rank_decode_elapsed_ms=shared_rank_decode_elapsed_ms,
            shared_rank_barrier1_wait_ms=shared_rank_barrier1_wait_ms,
            shared_rank_barrier2_wait_ms=shared_rank_barrier2_wait_ms,
            shared_rank_snapshot_elapsed_ms=shared_rank_snapshot_elapsed_ms,
            shared_rank_barrier3_wait_ms=shared_rank_barrier3_wait_ms,
            shared_rank_barrier4_wait_ms=shared_rank_barrier4_wait_ms,
            shared_snapshot_ready=shared_snapshot_ready,
            shared_unfinished_states=shared_unfinished_states,
            shared_stage_has_work=shared_stage_has_work,
            shared_stage_first_token_ms=shared_stage_first_token_ms,
        )
        if final_rows:
            append_jsonl(shard_output_path, final_rows)

        if global_dp_rank == 0:
            if migrated:
                shared_control["decode_step_total"] = int(
                    shared_control.get("stage_decode_base", 0)
                ) + int(max(shared_rank_step_totals[:], default=0))
                shared_control["stage_requests"] = list(shared_control["next_stage_requests"])
                shared_control["stage_id"] = stage_id + 1
                shared_control["stage_decode_base"] = int(shared_control["decode_step_total"])
                shared_control["sample_prev_total_steps"] = int(shared_control["decode_step_total"])
                shared_control["sample_prev_time"] = perf_counter()
                shared_control["command"] = "run"
                shared_control["migrate"] = False
                shared_control["all_done"] = False
                shared_control["stage_prefill_logged"] = False
                for idx in range(dp_size):
                    shared_snapshot_ready[idx] = 0
                    shared_rank_step_totals[idx] = 0
            else:
                shared_control["stage_requests"] = []
                shared_control["command"] = "done"

        if migrated and global_dp_rank != 0:
            while int(shared_control.get("stage_id", stage_id)) == stage_id:
                sleep(0.001)

        if not migrated:
            break

    batch_elapsed_times[global_dp_rank] = perf_counter() - batch_start
    sleep(1)


def merge_outputs(output_path: Path, dp_size: int) -> int:
    merged_rows: list[dict[str, Any]] = []
    for rank in range(dp_size):
        shard_path = Path(f"{output_path}.rank{rank:02d}.jsonl")
        if not shard_path.exists():
            continue
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                merged_rows.append(json.loads(line))
    merged_rows.sort(
        key=lambda row: (
            row["dataset_index"],
            row.get("repeat_idx", -1),
        )
    )
    write_jsonl(output_path, merged_rows)
    return len(merged_rows)


def print_request_token_logs(output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))

    rows.sort(
        key=lambda row: (
            row["dataset_index"],
            row.get("repeat_idx", -1),
        )
    )
    for row in rows:
        prompt_tokens = int(row.get("num_prompt_tokens", 0))
        response_tokens = int(row.get("num_output_tokens", 0))
        total_tokens = prompt_tokens + response_tokens
        print(
            "req "
            f"dataset_index={row['dataset_index']} "
            f"repeat_idx={row.get('repeat_idx', -1)} "
            f"migration_count={row.get('migration_count', 0)} "
            f"prompt_tokens={prompt_tokens} "
            f"response_tokens={response_tokens} "
            f"total_tokens={total_tokens}"
        )


def main() -> int:
    args = parse_args()
    dataset_indices = [int(item.strip()) for item in args.dataset_indices.split(",") if item.strip()]
    temperature, top_p, top_k, min_p = resolve_sampling_config(
        args.enable_thinking,
        args.temperature,
        args.top_p,
        args.top_k,
        args.min_p,
    )
    records = load_records(
        args.dataset,
        args.prompt_column,
        limit=0,
        dataset_indices=dataset_indices,
    )
    if not records:
        raise ValueError("No records loaded.")

    from vllm.utils import get_open_port

    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    output_path = Path(args.output_path)
    ensure_parent_dir(output_path)
    if output_path.exists():
        output_path.unlink()

    print(
        f"Tail batch migration enabled: requests={len(records) * args.repeat_per_prompt}, "
        f"migration_min_total_tokens={args.migration_min_total_tokens} "
        f"migration_enabled={args.migration_enabled} "
        f"tail_concentrate_enabled={args.tail_concentrate_enabled} "
        f"tail_concentrate_top_k={args.tail_concentrate_top_k} "
        f"tail_concentrate_ranks={args.tail_concentrate_ranks} "
        f"tail_concentrate_request_keys={len(parse_request_key_list(args.tail_concentrate_request_keys))} "
        f"sync_every_steps={args.sync_every_steps} "
        f"report_every_decode_steps={args.report_every_decode_steps}"
    )

    stage_barrier = Barrier(args.dp_size)
    control_lock = Lock()
    manager = Manager()
    shared_control = manager.dict()
    shared_unfinished_states = manager.list([[] for _ in range(args.dp_size)])
    shared_unfinished_counts = Array("i", args.dp_size)
    shared_rank_max_total_tokens = Array("q", args.dp_size)
    shared_rank_window_steps = Array("q", args.dp_size)
    shared_rank_step_totals = Array("q", args.dp_size)
    shared_rank_decode_elapsed_ms = Array("d", args.dp_size)
    shared_rank_barrier1_wait_ms = Array("d", args.dp_size)
    shared_rank_barrier2_wait_ms = Array("d", args.dp_size)
    shared_rank_snapshot_elapsed_ms = Array("d", args.dp_size)
    shared_rank_barrier3_wait_ms = Array("d", args.dp_size)
    shared_rank_barrier4_wait_ms = Array("d", args.dp_size)
    shared_snapshot_ready = Array("i", args.dp_size)
    shared_stage_has_work = Array("i", args.dp_size)
    shared_stage_first_token_ms = Array("d", args.dp_size)
    batch_elapsed_times = Array("d", args.dp_size)

    procs: list[Process] = []
    for local_dp_rank, global_dp_rank in enumerate(range(args.dp_size)):
        proc = Process(
            target=run_worker,
            kwargs={
                "model": args.model,
                "records": records,
                "output_path": str(output_path),
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
                "repeat_per_prompt": args.repeat_per_prompt,
                "migration_min_total_tokens": args.migration_min_total_tokens,
                "migration_enabled": args.migration_enabled,
                "tail_concentrate_enabled": args.tail_concentrate_enabled,
                "tail_concentrate_top_k": args.tail_concentrate_top_k,
                "tail_concentrate_ranks": args.tail_concentrate_ranks,
                "tail_concentrate_request_keys": args.tail_concentrate_request_keys,
                "sync_every_steps": args.sync_every_steps,
                "report_every_decode_steps": args.report_every_decode_steps,
                "stage_barrier": stage_barrier,
                "control_lock": control_lock,
                "shared_control": shared_control,
                "shared_unfinished_counts": shared_unfinished_counts,
                "shared_rank_max_total_tokens": shared_rank_max_total_tokens,
                "shared_rank_window_steps": shared_rank_window_steps,
                "shared_rank_step_totals": shared_rank_step_totals,
                "shared_rank_decode_elapsed_ms": shared_rank_decode_elapsed_ms,
                "shared_rank_barrier1_wait_ms": shared_rank_barrier1_wait_ms,
                "shared_rank_barrier2_wait_ms": shared_rank_barrier2_wait_ms,
                "shared_rank_snapshot_elapsed_ms": shared_rank_snapshot_elapsed_ms,
                "shared_rank_barrier3_wait_ms": shared_rank_barrier3_wait_ms,
                "shared_rank_barrier4_wait_ms": shared_rank_barrier4_wait_ms,
                "shared_snapshot_ready": shared_snapshot_ready,
                "shared_unfinished_states": shared_unfinished_states,
                "shared_stage_has_work": shared_stage_has_work,
                "shared_stage_first_token_ms": shared_stage_first_token_ms,
                "batch_elapsed_times": batch_elapsed_times,
            },
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=args.timeout)
        if proc.exitcode not in (0, None):
            exit_code = proc.exitcode
        if proc.is_alive():
            proc.kill()
            exit_code = 1

    if exit_code != 0:
        return exit_code

    merged_count = merge_outputs(output_path, args.dp_size)
    print_request_token_logs(output_path)
    batch_elapsed_s = max(batch_elapsed_times[:])
    print(
        f"Tail batch migration finished rows={merged_count} "
        f"elapsed_s={batch_elapsed_s:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
