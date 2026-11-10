"""
Offline-friendly resolution of Hugging Face model ids.

`from_pretrained("gpt2")` with a *short* alias needs a call to the Hub API to
resolve the full repo id — that fails in offline/air-gapped environments
(HF_HUB_OFFLINE=1) even when the model is fully cached, because the alias→id
mapping itself is API metadata. Passing the full repo id ("openai-community/gpt2")
reads the local cache directly and needs no network.

This module maps the short aliases used inside PyPOTS to their full repo ids,
so the LLM-backed models (GPT4TS/TimeLLM/MOMENT) load from a pre-seeded HF
cache on machines without Hub connectivity. Users can seed the cache from a
connected machine and set HF_HUB_OFFLINE=1.

Usage:
    from pypots.utils.hf_offline import resolve_hf_repo_id
    GPT2Model.from_pretrained(resolve_hf_repo_id("gpt2"))

The mapping only applies to known aliases; anything else (already-full ids,
local paths) comes back unchanged. When online, resolution is a no-op anyway
because the Hub resolves aliases itself.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

_SHORT_ALIAS_TO_REPO_ID = {
    "gpt2": "openai-community/gpt2",
    "bert-base-uncased": "google-bert/bert-base-uncased",
    "t5-small": "t5-small",  # canonical id is already short
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "t5-3b": "t5-3b",
    "t5-11b": "t5-11b",
}


def resolve_hf_repo_id(model_id: str) -> str:
    """Map a short HF alias to its full repo id; pass through anything else.

    Full repo ids, local directory paths, and unknown strings are returned
    unchanged — this never raises, so it is safe to wrap any from_pretrained
    argument.
    """
    if not isinstance(model_id, str) or "/" in model_id or model_id.startswith("."):
        return model_id
    return _SHORT_ALIAS_TO_REPO_ID.get(model_id, model_id)
