#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# AITER on compute partitions: the evaluation behind test_areas/aiter_partition_eval.yaml.
# Runs vLLM's checks (tests/rocm/aiter_partition_eval) and AITER's own partition unit tests from
# the source that docker/Dockerfile.rocm keeps in /opt/aiter-src. A pytest run that passes
# nothing (no tests collected, or all skipped) fails: run-amd-test.sh treats exit 5 as success.
set -uo pipefail

AITER_SRC=${AITER_SRC:-/opt/aiter-src}
WORKSPACE=${WORKSPACE:-/vllm-workspace}
echo "--- AITER under test"
python3 -c 'import importlib.metadata as m; print("amd-aiter", m.version("amd-aiter"))'
if [[ -f $AITER_SRC/AITER_SHA ]]; then
  echo "AITER source $AITER_SRC at $(cat "$AITER_SRC/AITER_SHA")"
else
  echo "No AITER source in the image: the stack's AITER (AITER_GIT_SHA unset at build time)"
fi
echo "expect: partition=${AITER_EVAL_PARTITION:-?} cu=${AITER_EVAL_CU:-?} chip=${AITER_EVAL_FULL_CHIP_CU:-?}"

failed=()
check() {  # check <name> <pytest exit> <log>
  if (($2 != 0)) || ! tail -n 3 "$3" | grep -qE '[0-9]+ passed'; then
    echo "^^^ +++"
    echo "$1: FAILED (pytest exit $2, or nothing passed)"
    failed+=("$1")
  fi
}
log=$(mktemp)

echo "--- (a-d) vLLM checks"
cd "$WORKSPACE"
pytest -v -s -rA -p no:cacheprovider tests/rocm/aiter_partition_eval 2>&1 | tee "$log"
check "(a-d) vLLM checks" "${PIPESTATUS[0]}" "$log"

own=()
if [[ -d $AITER_SRC ]]; then
  mapfile -t own < <(cd "$AITER_SRC" && ls tests/test_chip_info_partition.py \
    tests/test_tuned_cu_partition*.py op_tests/test_opus_a16w16_registry.py \
    op_tests/test_opus_a16w16_policy_parity.py 2>/dev/null)
fi
echo "--- (e) AITER partition unit tests"
if ((${#own[@]})); then
  # pytest, not python -m pytest: the cwd must not shadow the installed aiter.
  (cd "$AITER_SRC" && pytest -v -rA -p no:cacheprovider "${own[@]}") 2>&1 | tee "$log"
  check "(e) AITER partition unit tests" "${PIPESTATUS[0]}" "$log"
else
  echo "none at this AITER ref"
fi
rm -f "$log"

echo "--- Summary"
if ((${#failed[@]})); then
  printf 'FAILED: %s\n' "${failed[@]}"
  exit 1
fi
echo "All parts passed"
