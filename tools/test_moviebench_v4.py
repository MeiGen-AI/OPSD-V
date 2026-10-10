#!/usr/bin/env python3
"""Smoke-test a MovieBench v4 annotation file without models or GPUs.

Usage: python tools/test_moviebench_v4.py /path/to/moviebench120_vbench_full_info_v4.json

Checks the prompt eligibility used by the 2026-09-18 strict evaluation.
This validates annotations only; it does not run video metrics. Flickering
keeps all 120 eligible prompts: common static windows depend on model outputs.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {
    "subject_consistency": 106,
    "background_consistency": 114,
    "aesthetic_quality": 120,
    "imaging_quality": 120,
    "motion_smoothness": 120,
    "dynamic_degree": 120,
    "temporal_flickering": 120,
    "overall_consistency": 120,
    "object_class": 111,
    "multiple_objects": 55,
    "color": 47,
    "spatial_relationship": 5,
    "scene": 96,
    "human_action": 11,
    "appearance_style": 80,
    "temporal_style": 4,
}
AUXILIARY_DIMENSIONS = {
    "object_class", "multiple_objects", "color", "spatial_relationship",
    "scene", "human_action", "appearance_style", "temporal_style",
}
STRICT_MULTIPLE_OBJECT_IDS = {
    0, 4, 5, 7, 8, 10, 14, 17, 18, 20, 21, 22, 24, 26, 27, 28,
    31, 32, 35, 36, 37, 38, 39, 40, 45, 48, 49, 50, 53, 54, 55,
    59, 61, 63, 64, 65, 66, 68, 70, 71, 72, 74, 81, 88, 90, 93,
    94, 96, 97, 101, 102, 107, 110, 111, 115,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_nonempty(value, location):
    require(value is not None and value != "" and value != [] and value != {},
            f"Empty auxiliary value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            check_nonempty(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_nonempty(child, f"{location}[{index}]")


def validate(records):
    require(isinstance(records, list) and len(records) == 120,
            "Expected a JSON list containing 120 prompts")
    counts = Counter()
    ids = []
    multiple_ids = set()
    spatial_ids = set()
    for row in records:
        require(isinstance(row, dict), "Each prompt must be an object")
        prompt_id = row.get("prompt_id")
        require(type(prompt_id) is int, "prompt_id must be an integer")
        ids.append(prompt_id)
        prompt = row.get("prompt_en")
        require(isinstance(prompt, str) and bool(prompt.strip()),
                f"Prompt {prompt_id}: missing prompt_en")
        require(row.get("annotation_version") == "moviebench_v4_strict_scope_20260918",
                f"Prompt {prompt_id}: unexpected annotation version")
        dimensions = row.get("dimension")
        require(isinstance(dimensions, list) and all(isinstance(x, str) for x in dimensions),
                f"Prompt {prompt_id}: dimension must be a list of strings")
        require(len(dimensions) == len(set(dimensions)),
                f"Prompt {prompt_id}: repeated dimensions")
        auxiliary = row.get("auxiliary_info")
        require(isinstance(auxiliary, dict),
                f"Prompt {prompt_id}: auxiliary_info must be an object")
        require(set(auxiliary) == set(dimensions) & AUXILIARY_DIMENSIONS,
                f"Prompt {prompt_id}: auxiliary_info does not match active dimensions")
        check_nonempty(auxiliary, f"prompt[{prompt_id}].auxiliary_info")
        counts.update(dimensions)
        if "multiple_objects" in dimensions:
            multiple_ids.add(prompt_id)
        if "spatial_relationship" in dimensions:
            spatial_ids.add(prompt_id)
    require(sorted(ids) == list(range(120)), "Expected unique prompt IDs 0 through 119")
    require(dict(counts) == EXPECTED_COUNTS, f"Unexpected dimension counts: {dict(counts)}")
    require(multiple_ids == STRICT_MULTIPLE_OBJECT_IDS,
            "Multiple-object IDs do not match the strict 55-prompt scope")
    require(spatial_ids == {4, 10, 32, 49, 70}, "Unexpected spatial-relationship scope")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    try:
        records = json.loads(args.annotations.read_text(encoding="utf-8"))
        counts = validate(records)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print("PASS: 120 prompts; all 16 dimension counts; strict multiple_objects=55")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
