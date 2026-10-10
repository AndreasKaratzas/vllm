# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render AMD workers after runtime selection; never upload or publish anything."""

import argparse
import copy
import json
from pathlib import Path

import regex as re
import yaml
from runtime_worker import validate_handoff

BUILDER_KEYS = {
    "image-build",
    "image-build-amd",
    "refresh-rocm-base-amd",
    "ensure-ci-base-amd",
    "verify-native-ci-base-amd",
    "promote-stable-rocm-images-amd",
}
MUTATIONS = re.compile(
    r"\b(?:uv\s+pip|pip[0-9.]*|python[0-9.]*\s+-m\s+pip)\s+"
    r"(?:install|uninstall|wheel|download)\b|"
    r"\b(?:apt(?:-get)?|conda|mamba)\s+(?:install|update)\b|"
    r"\buv\s+(?:sync|build)\b|\bgit\s+clone\b|python_only_compile\.sh|"
    r"\b(?:export\s+)?(?:PYTHONPATH|PYTHONHOME|VIRTUAL_ENV)\s*="
)
OLD_ENV = {
    "DOCKER_IMAGE_NAME",
    "DOCKER_BUILDKIT",
    "VLLM_CI_BASE_IMAGE",
    "VLLM_CI_FALLBACK_IMAGE",
    "VLLM_CI_USE_ARTIFACTS",
    "AMD_CI_RUNTIME",
    "NATIVE_CI",
    "VLLM_CI_DOCKER_DISABLED",
    "VLLM_CI_ARTIFACT_GLOB",
    "VLLM_CI_ARTIFACT_STEP",
    "VLLM_CI_ARTIFACT_CHECKSUM_GLOB",
    "VLLM_CI_REQUIRE_WORKSPACE_MOUNT",
}


def _commands(step: dict) -> str:
    value = (step.get("env") or {}).get("VLLM_TEST_COMMANDS")
    if value:
        return value
    commands = step.get("commands", step.get("command", []))
    return commands if isinstance(commands, str) else " && ".join(commands)


def render(
    selected: dict,
    handoff: dict,
    artifact_root: Path,
    cache_root: Path,
    only_keys: set[str] | None = None,
    external_dependencies: set[str] | None = None,
) -> tuple[dict, dict]:
    """Transform selected AMD jobs without repeating producer or builder keys."""
    validate_handoff(handoff)
    for path in (artifact_root, cache_root):
        if not path.is_absolute() or ":" in str(path) or "," in str(path):
            raise ValueError("artifact/cache roots must be absolute local paths")
    commands, blocks = [], {}
    for group in selected.get("steps", []):
        for step in group.get("steps", [group]):
            if "block" in step:
                blocks[step["key"]] = step
            elif step.get("key") not in BUILDER_KEYS and str(
                (step.get("agents") or {}).get("queue", "")
            ).startswith("amd_"):
                commands.append(step)
    found = {step["key"] for step in commands}
    if only_keys and only_keys - found:
        raise ValueError(f"unknown AMD worker keys: {sorted(only_keys - found)}")
    workers = [step for step in commands if not only_keys or step["key"] in only_keys]
    if not workers:
        raise ValueError("selected pipeline does not contain AMD deployment workers")
    report = {
        "supported": [],
        "unsupported": [],
        "runtime_image": handoff["runtime_image"],
        "producer_step": handoff["producer_step"],
        "uploads_performed": False,
    }
    for step in workers:
        command = _commands(step)
        reason = None
        if MUTATIONS.search(command):
            reason = "job-time package/source mutation; move it to the locked producer"
        elif "run-amd-test.sh" in command or not command:
            reason = "missing underlying test commands"
        if reason:
            report["unsupported"].append({"key": step["key"], "reason": reason})
        else:
            report["supported"].append(step["key"])
    if report["unsupported"]:
        return {"steps": []}, report
    needed_blocks = set()
    for step in workers:
        depends = step.get("depends_on") or []
        if isinstance(depends, str):
            depends = [depends]
        needed_blocks.update(key for key in depends if key in blocks)
    emitted = {step["key"] for step in workers} | needed_blocks
    external = set(external_dependencies or ()) | {handoff["producer_step"]}

    def dependencies(step):
        result = [handoff["producer_step"]]
        requested = step.get("depends_on") or []
        for key in [requested] if isinstance(requested, str) else requested:
            if key in BUILDER_KEYS:
                continue
            if key not in emitted | external:
                raise ValueError(f"worker references an unavailable dependency: {key}")
            if key not in result:
                result.append(key)
        return result

    output = []
    for key in sorted(needed_blocks):
        block = copy.deepcopy(blocks[key])
        block["depends_on"] = dependencies(block)
        output.append(block)
    for original in workers:
        worker = copy.deepcopy(original)
        worker["depends_on"] = dependencies(worker)
        env = {
            key: value
            for key, value in (worker.get("env") or {}).items()
            if key not in OLD_ENV
        }
        native = any("kubernetes" in plugin for plugin in worker.get("plugins") or [])
        env.update(
            {
                "ROCM_CI_EXECUTION": "native" if native else "dind",
                "ROCM_CI_HANDOFF": json.dumps(
                    handoff, sort_keys=True, separators=(",", ":")
                ),
                "ROCM_RUNTIME_IMAGE": handoff["runtime_image"],
                "ROCM_ARTIFACT_ROOT": str(artifact_root),
                "ROCM_CACHE_ROOT": str(cache_root),
                "VLLM_CI_WORKSPACE": "/vllm-workspace",
                "VLLM_TEST_COMMANDS": _commands(original),
                "PIP_NO_INDEX": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        worker["commands"] = [
            "python3 .buildkite/scripts/rocm/runtime_worker.py --launch"
        ]
        worker.pop("command", None)
        if native:
            for plugin in worker["plugins"]:
                if "kubernetes" not in plugin:
                    continue
                patch = plugin["kubernetes"]["podSpecPatch"]
                container = next(
                    item
                    for item in patch["containers"]
                    if item["name"] == "container-0"
                )
                container["image"] = handoff["runtime_image"]
                container["imagePullPolicy"] = "IfNotPresent"
                container["env"] = [
                    item
                    for item in container.get("env", [])
                    if item["name"] not in OLD_ENV
                ]
                for name, path, readonly in (
                    ("runtime-artifacts", artifact_root, True),
                    ("runtime-cache", cache_root, False),
                ):
                    patch.setdefault("volumes", []).append(
                        {
                            "name": name,
                            "hostPath": {
                                "path": str(path),
                                "type": "Directory"
                                if readonly
                                else "DirectoryOrCreate",
                            },
                        }
                    )
                    container.setdefault("volumeMounts", []).append(
                        {"name": name, "mountPath": str(path), "readOnly": readonly}
                    )
        else:
            env["DOCKER_IMAGE_NAME"] = handoff["runtime_image"]
        names = [name for name in env if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)]
        if len(names) != len(env):
            raise ValueError("invalid job environment variable name")
        env["ROCM_CI_ENV_NAMES"] = " ".join(names)
        worker["env"] = env
        output.append(worker)
    return {"steps": [{"group": "AMD deployment tests", "steps": output}]}, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-pipeline", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only-step-key", action="append")
    parser.add_argument("--external-dependency", action="append", default=[])
    args = parser.parse_args()
    try:
        pipeline, report = render(
            yaml.safe_load(args.selected_pipeline.read_text()),
            json.loads(args.handoff.read_text()),
            args.artifact_root,
            args.cache_root,
            set(args.only_step_key) if args.only_step_key else None,
            set(args.external_dependency),
        )
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if report["unsupported"]:
            parser.exit(2, "unsupported selected jobs; see the explicit report\n")
        args.output.write_text(yaml.safe_dump(pipeline, sort_keys=False))
    except (ValueError, OSError, yaml.YAMLError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
