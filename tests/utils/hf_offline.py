"""
Test cases for the functions and classes in module `pypots.utils.hf_offline`.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

import unittest

from pypots.utils.hf_offline import resolve_hf_repo_id


class TestResolveHfRepoId(unittest.TestCase):
    def test_0_short_alias(self):
        # short aliases resolve to their canonical repo ids
        assert resolve_hf_repo_id("gpt2") == "openai-community/gpt2"
        assert resolve_hf_repo_id("bert-base-uncased") == "google-bert/bert-base-uncased"

    def test_1_full_repo_id(self):
        # full repo ids pass through unchanged
        assert resolve_hf_repo_id("openai-community/gpt2") == "openai-community/gpt2"
        assert resolve_hf_repo_id("google-bert/bert-base-uncased") == "google-bert/bert-base-uncased"

    def test_2_canonical_short(self):
        # some canonical repo ids are already short (e.g. t5 family) — pass through
        assert resolve_hf_repo_id("t5-small") == "t5-small"
        assert resolve_hf_repo_id("t5-base") == "t5-base"

    def test_3_local_path(self):
        # local paths and non-hub ids are left alone
        assert resolve_hf_repo_id("./local/model") == "./local/model"
        assert resolve_hf_repo_id("/abs/path") == "/abs/path"

    def test_4_unknown_alias(self):
        # an unmapped short alias is returned as-is; resolution falls back to the Hub
        assert resolve_hf_repo_id("some-unknown-model") == "some-unknown-model"

    def test_5_non_string(self):
        # non-string inputs (e.g. None or a config object) pass through untouched
        assert resolve_hf_repo_id(None) is None


if __name__ == "__main__":
    unittest.main()
