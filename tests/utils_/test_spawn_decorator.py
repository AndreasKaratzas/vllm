# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spawn_new_process_for_each_test decorator."""

import sys

import pytest

from tests.utils import spawn_new_process_for_each_test


@spawn_new_process_for_each_test
def test_spawn_decorator_passing():
    """Passing function should complete normally."""
    assert 1 + 1 == 2


@pytest.mark.xfail(raises=RuntimeError, strict=True)
@spawn_new_process_for_each_test
def test_spawn_decorator_failure_is_caught():
    """Failing function should raise RuntimeError, never silently pass."""
    raise ValueError("intentional failure")


@spawn_new_process_for_each_test
def test_spawn_decorator_skip():
    """pytest.skip inside subprocess should propagate correctly."""
    pytest.skip("intentional skip")


@spawn_new_process_for_each_test
@pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (0, 0, 0)])
def test_spawn_decorator_parametrized(x, y, expected):
    """Args and kwargs must be forwarded correctly to subprocess."""
    assert x + y == expected


@spawn_new_process_for_each_test
def _failing_child_with_output():
    print("child stdout marker")
    print("child stderr marker", file=sys.stderr)
    raise ValueError("intentional output failure")


def test_spawn_decorator_failure_emits_subprocess_output(capfd):
    """Failure output should be emitted before the exception is re-raised."""
    with pytest.raises(RuntimeError, match="intentional output failure"):
        _failing_child_with_output()

    out, err = capfd.readouterr()
    assert "child stdout marker" in out
    assert "child stderr marker" in err
