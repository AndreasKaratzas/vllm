#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content key of a Dockerfile target.

The key covers everything that can change the target's bytes: the text of the
target stage and of every stage it depends on (FROM, COPY/ADD --from,
RUN --mount from=), the values of the build args those stages see, external
image references, and the contents of the build-context files they read.
Stages supplied as named contexts (--context name=key) contribute only that
key, which is how an artifact's key feeds the images built on top of it.

Usage:
    stage_key.py <dockerfile> <target> [--build-arg K=V]... [--context S=KEY]...
        [--root DIR] [--show]

Prints a 64-hex sha256. --show prints the hashed material instead.
"""

import argparse
import hashlib
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

_ARG_REF = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::[-+][^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass
class Stage:
    name: str
    base: str
    instructions: list[str] = field(default_factory=list)


def logical_lines(text: str) -> list[str]:
    """Join continuations and heredoc bodies into one string per instruction."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [line]
        while parts[-1].rstrip().endswith("\\") and i < len(lines):
            nxt = lines[i]
            i += 1
            if nxt.lstrip().startswith("#"):
                continue
            parts.append(nxt)
        instr = "\n".join(parts)
        heredoc = re.search(r"<<-?'?\"?([A-Za-z_]+)'?\"?", instr)
        if heredoc and instr.split(None, 1)[0].upper() in ("RUN", "COPY", "ADD"):
            body = []
            while i < len(lines) and lines[i].strip() != heredoc.group(1):
                body.append(lines[i])
                i += 1
            i += 1
            instr += "\n" + "\n".join(body) + "\n" + heredoc.group(1)
        out.append(instr)
    return out


def parse(text: str) -> tuple[dict[str, str | None], list[Stage]]:
    global_args: dict[str, str | None] = {}
    stages: list[Stage] = []
    for instr in logical_lines(text):
        keyword, _, rest = instr.partition(" ")
        keyword = keyword.upper()
        if keyword == "FROM":
            words = [w for w in rest.split() if not w.startswith("--")]
            base = words[0]
            name = (
                words[2]
                if len(words) >= 3 and words[1].upper() == "AS"
                else str(len(stages))
            )
            stages.append(Stage(name=name, base=base))
            continue
        if not stages and keyword == "ARG":
            for decl in shlex.split(rest):
                k, eq, v = decl.partition("=")
                global_args[k] = v if eq else None
            continue
        if stages:
            stages[-1].instructions.append(instr)
    return global_args, stages


def substitute(value: str, env: dict[str, str | None]) -> str:
    def repl(m: re.Match) -> str:
        return env.get(m.group(1) or m.group(2)) or ""

    return _ARG_REF.sub(repl, value)


def stage_args(stage: Stage, global_args, build_args) -> dict[str, str | None]:
    """Values of the args visible to RUN in this stage (declared ARGs only)."""
    env: dict[str, str | None] = {}
    for instr in stage.instructions:
        if instr.split(None, 1)[0].upper() != "ARG":
            continue
        for decl in shlex.split(instr.split(None, 1)[1]):
            k, eq, v = decl.partition("=")
            if k in build_args:
                env[k] = build_args[k]
            elif eq:
                env[k] = substitute(v, {**global_args, **env})
            else:
                env[k] = global_args.get(k)
    return env


def references(stage: Stage, env) -> tuple[list[str], list[str]]:
    """Stages/images this stage reads (--from), and build-context paths."""
    froms: list[str] = []
    paths: list[str] = []
    for instr in stage.instructions:
        keyword, _, rest = instr.partition(" ")
        keyword = keyword.upper()
        if keyword not in ("COPY", "ADD", "RUN"):
            continue
        flat = instr.replace("\\\n", " ")
        for m in re.finditer(r"--from=(\S+)", flat):
            froms.append(substitute(m.group(1), env))
        for m in re.finditer(r"--mount=(\S+)", flat):
            opts = dict(o.partition("=")[::2] for o in m.group(1).split(","))
            if opts.get("type", "bind") != "bind":
                continue
            if "from" in opts:
                froms.append(substitute(opts["from"], env))
            elif "source" in opts or "src" in opts:
                paths.append(substitute(opts.get("source") or opts.get("src"), env))
        if keyword in ("COPY", "ADD") and "--from=" not in flat:
            body = flat.split("<<", 1)[0]
            words = [w for w in shlex.split(body)[1:] if not w.startswith("--")]
            paths.extend(substitute(w, env) for w in words[:-1])
    return froms, paths


def hash_path(root: str, rel: str) -> list[str]:
    """Tracked content under rel: the git index when available, so untracked
    local files (bytecode caches, build output) never change a key."""
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        raise SystemExit(f"stage_key: build-context path not found: {rel}")
    try:
        listed = subprocess.run(
            ["git", "-C", root, "ls-files", "-s", "--", rel],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        listed = []
    if listed:
        # "<mode> <blob> <stage>\t<path>": the blob id is the content hash.
        return [f"git {line}" for line in sorted(listed)]
    entries = []
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs)
    for f in files:
        with open(f, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        mode = "x" if os.access(f, os.X_OK) else "-"
        entries.append(f"file {os.path.relpath(f, root)} {mode} {digest}")
    return entries


def material(
    dockerfile: str, target: str, build_args, contexts, root: str
) -> list[str]:
    with open(dockerfile) as fh:
        global_args, stages = parse(fh.read())
    global_args = {k: build_args.get(k, v) for k, v in global_args.items()}
    by_name = {s.name: s for s in stages}
    if target not in by_name:
        raise SystemExit(f"stage_key: no stage named {target!r} in {dockerfile}")

    out: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        if name in contexts:
            out.append(f"context {name} {contexts[name]}")
            return
        if name not in by_name:
            out.append(f"image {name}")
            return
        stage = by_name[name]
        env = stage_args(stage, global_args, build_args)
        base = substitute(stage.base, global_args)
        out.append(f"stage {name} from {base}")
        out.extend(f"  {i}" for i in stage.instructions)
        out.extend(f"  arg {k}={v}" for k, v in sorted(env.items()))
        froms, paths = references(stage, {**global_args, **env})
        for p in paths:
            out.extend(f"  {e}" for e in hash_path(root, p))
        if base != "scratch":
            visit(base)
        for f in froms:
            visit(f)

    visit(target)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dockerfile")
    ap.add_argument("target")
    ap.add_argument("--build-arg", action="append", default=[])
    ap.add_argument("--context", action="append", default=[])
    ap.add_argument("--root", default=".")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args(argv)
    build_args = dict(x.partition("=")[::2] for x in a.build_arg)
    contexts = dict(x.partition("=")[::2] for x in a.context)
    lines = material(a.dockerfile, a.target, build_args, contexts, a.root)
    if a.show:
        print("\n".join(lines))
    else:
        print(hashlib.sha256("\n".join(lines).encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
