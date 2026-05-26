# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def install_list_feature_compat() -> None:
    """Allow older `datasets` releases to read caches written with List."""
    try:
        from datasets import Sequence
        from datasets.features import features

        features._FEATURE_TYPES.setdefault("List", Sequence)
    except Exception:
        # The caller may be running without `datasets` installed.
        pass
