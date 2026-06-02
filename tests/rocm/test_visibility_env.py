# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.platforms.rocm import RocmPlatform, _sync_hip_cuda_env_vars


def test_ray_worker_prefers_narrower_hip_visible_devices(monkeypatch):
    monkeypatch.setenv("RAY_JOB_ID", "test-job")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")

    _sync_hip_cuda_env_vars()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "2"


def test_non_ray_visibility_conflict_still_raises(monkeypatch):
    monkeypatch.delenv("RAY_JOB_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")

    with pytest.raises(ValueError, match="Inconsistent GPU visibility env vars"):
        _sync_hip_cuda_env_vars()


def test_ray_worker_non_subset_visibility_conflict_still_raises(monkeypatch):
    monkeypatch.setenv("RAY_JOB_ID", "test-job")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")

    with pytest.raises(ValueError, match="Inconsistent GPU visibility env vars"):
        _sync_hip_cuda_env_vars()


def test_rocm_device_control_env_var_updates_cover_visibility_aliases():
    env_vars = RocmPlatform().get_device_control_env_var_updates("0,1")

    assert env_vars == {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HIP_VISIBLE_DEVICES": "0,1",
    }
