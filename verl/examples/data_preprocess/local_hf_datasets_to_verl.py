import argparse
import ast
import copy
import json
import os
from pathlib import Path

import datasets


MATH_INSTRUCTION = "Let's think step by step and put the final answer within \\boxed{}."
CODE_INSTRUCTION = "Write executable Python 3 code only. Do not include any explanation outside the code."
DEFAULT_TEST_SIZE = 500
RANDOM_SEED = 42

NUMINA_SOURCE_MAP = {
    "aops_forum": "numina_aops_forum",
    "amc_aime": "numina_amc_aime",
    "cn_k12": "numina_cn_k12",
    "olympiads": "numina_olympiads",
    "synthetic_amc": "numina_synthetic_amc",
    "synthetic_math": "numina_synthetic_math",
}
FALLBACK_MATH_DATA_SOURCE = "numina_synthetic_math"
COMMON_KODCODE_IMPORTS = {
    "collections": "import collections",
    "itertools": "import itertools",
    "json": "import json",
    "math": "import math",
    "np": "import numpy as np",
    "os": "import os",
    "pd": "import pandas as pd",
    "random": "import random",
    "re": "import re",
    "sys": "import sys",
    "tempfile": "import tempfile",
}


def last_boxed_only_string(text):
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    right_brace_idx = None
    num_left_braces_open = 0
    i = idx
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def remove_boxed(text):
    if text.startswith("\\boxed "):
        return text[len("\\boxed ") :]

    prefix = "\\boxed{"
    if not text.startswith(prefix) or not text.endswith("}"):
        raise ValueError(f"Invalid boxed answer: {text}")
    return text[len(prefix) : -1]


def extract_boxed_answer(solution):
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed).strip()
    except Exception:
        return None


def is_fixture_function(node):
    if not isinstance(node, ast.FunctionDef):
        return False
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "fixture":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "fixture":
            return True
    return False


def normalize_helper_node(node):
    node_copy = copy.deepcopy(node)
    if is_fixture_function(node_copy):
        node_copy.decorator_list = []
    return node_copy


def has_import_for_alias(test_script, alias):
    if alias in {"np", "pd"}:
        return f"import {'numpy' if alias == 'np' else 'pandas'} as {alias}" in test_script
    if alias == "collections":
        return "import collections" in test_script or "from collections import" in test_script
    if alias == "itertools":
        return "import itertools" in test_script or "from itertools import" in test_script
    return f"import {alias}" in test_script or f"from {alias} import" in test_script


def infer_missing_imports(test_script):
    missing_imports = []
    for alias, import_stmt in COMMON_KODCODE_IMPORTS.items():
        if f"{alias}." in test_script and not has_import_for_alias(test_script, alias):
            missing_imports.append(import_stmt)
    return missing_imports


def build_kodcode_assert_cases(test_script):
    try:
        tree = ast.parse(test_script)
    except SyntaxError:
        return None

    fixture_nodes = {}
    helper_nodes = []
    test_nodes = []

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            continue
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_nodes.append(node)
            continue

        normalized = normalize_helper_node(node)
        helper_nodes.append(normalized)
        if isinstance(normalized, ast.FunctionDef) and is_fixture_function(node):
            fixture_nodes[normalized.name] = normalized

    if not test_nodes:
        return None

    helper_sections = []
    missing_imports = infer_missing_imports(test_script)
    if missing_imports:
        helper_sections.append("\n".join(missing_imports))
    helper_code = "\n\n".join(ast.unparse(node) for node in helper_nodes).strip()
    if helper_code:
        helper_sections.append(helper_code)
    helper_code = "\n\n".join(helper_sections).strip()
    assert_cases = []

    for test_node in test_nodes:
        if test_node.decorator_list:
            return None

        params = [arg.arg for arg in test_node.args.args]
        if any(param not in fixture_nodes for param in params):
            return None

        setup_lines = []
        for param in params:
            fixture_node = fixture_nodes[param]
            if fixture_node.args.args or fixture_node.args.posonlyargs or fixture_node.args.kwonlyargs:
                return None
            if fixture_node.args.vararg or fixture_node.args.kwarg:
                return None
            setup_lines.append(f"{param} = {param}()")

        test_body_code = "\n".join(ast.unparse(stmt) for stmt in test_node.body).strip()
        if not test_body_code:
            return None

        case_parts = []
        if helper_code:
            case_parts.append(helper_code)
        if setup_lines:
            case_parts.append("\n".join(setup_lines))
        case_parts.append(test_body_code)
        assert_cases.append("\n\n".join(case_parts))

    return assert_cases


def attach_split_info(dataset, split_name):
    def process_fn(example, idx):
        extra_info = dict(example["extra_info"])
        extra_info["split"] = split_name
        extra_info["index"] = idx
        return {"extra_info": extra_info}

    return dataset.map(process_fn, with_indices=True)


def save_train_test(dataset_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict["train"].to_parquet(os.path.join(output_dir, "train.parquet"))
    dataset_dict["test"].to_parquet(os.path.join(output_dir, "test.parquet"))


def process_numina(root_dir, test_size):
    dataset_dir = Path(root_dir) / "NuminaMath-CoT"
    local_dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "train": sorted(str(path) for path in (dataset_dir / "data").glob("train-*.parquet")),
            "test": sorted(str(path) for path in (dataset_dir / "data").glob("test-*.parquet")),
        },
    )

    def make_map_fn(split_name):
        def process_fn(example):
            ground_truth = extract_boxed_answer(example["solution"])
            keep = ground_truth is not None and ground_truth != ""
            data_source = NUMINA_SOURCE_MAP.get(example["source"], FALLBACK_MATH_DATA_SOURCE)
            return {
                "keep": keep,
                "data_source": data_source,
                "prompt": [{"role": "user", "content": example["problem"] + " " + MATH_INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "dataset_name": "AI-MO/NuminaMath-CoT",
                    "original_source": example["source"],
                    "split": split_name,
                },
            }

        return process_fn

    train_dataset = local_dataset["train"].map(
        make_map_fn("train"),
        remove_columns=local_dataset["train"].column_names,
    )
    test_dataset = local_dataset["test"].map(
        make_map_fn("test"),
        remove_columns=local_dataset["test"].column_names,
    )

    train_dataset = train_dataset.filter(lambda example: example["keep"]).remove_columns(["keep"])
    test_dataset = test_dataset.filter(lambda example: example["keep"]).remove_columns(["keep"])

    train_dataset = attach_split_info(train_dataset, "train")
    test_dataset = attach_split_info(test_dataset, "test")

    save_train_test(datasets.DatasetDict({"train": train_dataset, "test": test_dataset}), dataset_dir)

    print(
        f"Processed NuminaMath-CoT: train={len(train_dataset)} test={len(test_dataset)} "
        f"saved_to={dataset_dir}"
    )


def process_deepscaler(root_dir, test_size):
    dataset_dir = Path(root_dir) / "DeepScaleR-Preview-Dataset"
    local_dataset = datasets.load_dataset(
        "json",
        data_files={"train": str(dataset_dir / "deepscaler.json")},
    )["train"]

    def process_fn(example):
        answer = (example.get("answer") or "").strip()
        keep = answer != ""
        return {
            "keep": keep,
            "data_source": FALLBACK_MATH_DATA_SOURCE,
            "prompt": [{"role": "user", "content": example["problem"] + " " + MATH_INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "dataset_name": "agentica-org/DeepScaleR-Preview-Dataset",
                "raw_answer": answer,
            },
        }

    processed = local_dataset.map(process_fn, remove_columns=local_dataset.column_names)
    processed = processed.filter(lambda example: example["keep"]).remove_columns(["keep"])

    split_size = min(test_size, max(100, len(processed) // 20))
    split_dataset = processed.train_test_split(test_size=split_size, seed=RANDOM_SEED, shuffle=True)
    split_dataset["train"] = attach_split_info(split_dataset["train"], "train")
    split_dataset["test"] = attach_split_info(split_dataset["test"], "test")

    save_train_test(split_dataset, dataset_dir)

    print(
        f"Processed DeepScaleR-Preview-Dataset: train={len(split_dataset['train'])} "
        f"test={len(split_dataset['test'])} saved_to={dataset_dir}"
    )


def process_kodcode(root_dir, test_size):
    dataset_dir = Path(root_dir) / "KodCode-Light-RL-10K"
    local_dataset = datasets.load_dataset(
        "parquet",
        data_files={"train": str(dataset_dir / "data" / "train-00000-of-00001.parquet")},
    )["train"]

    def process_fn(example):
        assert_cases = build_kodcode_assert_cases(example["test"])
        keep = assert_cases is not None and len(assert_cases) > 0
        ground_truth = json.dumps({"assert_case": assert_cases}, ensure_ascii=False) if keep else None
        return {
            "keep": keep,
            "data_source": "apps",
            "prompt": [{"role": "user", "content": example["question"] + "\n\n" + CODE_INSTRUCTION}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "dataset_name": "KodCode/KodCode-Light-RL-10K",
                "question_id": example["question_id"],
                "subset": example["subset"],
                "style": example["style"],
                "function_name": example["test_info"][0]["function_name"] if example["test_info"] else None,
            },
        }

    processed = local_dataset.map(process_fn, remove_columns=local_dataset.column_names)
    processed = processed.filter(lambda example: example["keep"]).remove_columns(["keep"])

    split_size = min(test_size, max(100, len(processed) // 20))
    split_dataset = processed.train_test_split(test_size=split_size, seed=RANDOM_SEED, shuffle=True)
    split_dataset["train"] = attach_split_info(split_dataset["train"], "train")
    split_dataset["test"] = attach_split_info(split_dataset["test"], "test")

    save_train_test(split_dataset, dataset_dir)

    print(
        f"Processed KodCode-Light-RL-10K: train={len(split_dataset['train'])} "
        f"test={len(split_dataset['test'])} saved_to={dataset_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="~/data", help="Root directory containing downloaded raw datasets.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "numina", "kodcode", "deepscaler"],
        help="Which dataset to preprocess.",
    )
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()

    root_dir = os.path.expanduser(args.root_dir)

    if args.dataset in {"all", "numina"}:
        process_numina(root_dir, args.test_size)
    if args.dataset in {"all", "kodcode"}:
        process_kodcode(root_dir, args.test_size)
    if args.dataset in {"all", "deepscaler"}:
        process_deepscaler(root_dir, args.test_size)


if __name__ == "__main__":
    main()
