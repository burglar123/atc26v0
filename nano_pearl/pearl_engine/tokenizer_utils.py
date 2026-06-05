from __future__ import annotations

import os
from typing import Any

from transformers import AutoTokenizer


TOKENIZER_COMPAT_PROBES = (
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "用户：你好\n助手：",
    "<|begin_of_text|>",
)


def resolve_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    parent, name = os.path.split(model_path)
    alternatives = []
    if name.startswith("Meta-"):
        alternatives.append(os.path.join(parent, name[len("Meta-") :]))
    else:
        alternatives.append(os.path.join(parent, f"Meta-{name}"))
    for alternative in alternatives:
        if os.path.exists(alternative):
            return alternative
    return model_path


def load_tokenizer(model_path: str, use_fast: bool | None = None):
    kwargs = {"trust_remote_code": True}
    if use_fast is not None:
        kwargs["use_fast"] = use_fast
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(model_path), **kwargs)
    ensure_pad_token(tokenizer)
    return tokenizer


def ensure_pad_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    if getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _as_token_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        ids: list[int] = []
        for item in value:
            if item is not None:
                ids.append(int(item))
        return ids
    return []


def find_eot_token_id(tokenizer) -> int | None:
    token = "<|eot_id|>"
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}
    if token not in vocab:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return None
    if token_id < 0:
        return None
    return token_id


def collect_stop_token_ids(config, tokenizer) -> list[int]:
    stop_ids: list[int] = []
    stop_ids.extend(_as_token_id_list(getattr(config, "eos_token_id", None)))
    stop_ids.extend(_as_token_id_list(getattr(tokenizer, "eos_token_id", None)))
    eot_id = find_eot_token_id(tokenizer)
    if eot_id is not None:
        stop_ids.append(eot_id)
    seen = set()
    deduped = []
    for token_id in stop_ids:
        if token_id not in seen:
            seen.add(token_id)
            deduped.append(token_id)
    return deduped


def compact_stop_token_ids(stop_ids: list[int]) -> int | list[int] | None:
    if not stop_ids:
        return None
    if len(stop_ids) == 1:
        return stop_ids[0]
    return list(stop_ids)


def tokenizer_diagnostics(model_path: str, config, tokenizer) -> dict[str, Any]:
    eot_id = find_eot_token_id(tokenizer)
    return {
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", model_path)),
        "model_path": str(resolve_model_path(model_path)),
        "architectures": list(getattr(config, "architectures", []) or []),
        "vocab_size": getattr(config, "vocab_size", None),
        "len_tokenizer": len(tokenizer),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "config_eos_token_id": getattr(config, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "additional_special_tokens": list(getattr(tokenizer, "additional_special_tokens", []) or []),
        "eot_token_id": eot_id,
        "stop_token_ids": collect_stop_token_ids(config, tokenizer),
        "chat_template_available": bool(getattr(tokenizer, "chat_template", None)),
    }


def check_tokenizer_compatibility(draft_tokenizer, target_tokenizer) -> dict[str, Any]:
    draft_len = len(draft_tokenizer)
    target_len = len(target_tokenizer)
    probe_results = []
    all_probes_match = True
    for probe in TOKENIZER_COMPAT_PROBES:
        draft_ids = draft_tokenizer.encode(probe, add_special_tokens=False)
        target_ids = target_tokenizer.encode(probe, add_special_tokens=False)
        same = list(draft_ids) == list(target_ids)
        all_probes_match = all_probes_match and same
        probe_results.append(
            {
                "probe": probe,
                "same": same,
                "draft_len": len(draft_ids),
                "target_len": len(target_ids),
            }
        )
    vocab_size_equal = draft_len == target_len
    return {
        "tokenizer_compatibility_check_passed": bool(vocab_size_equal or all_probes_match),
        "tokenizer_vocab_size_equal": bool(vocab_size_equal),
        "tokenizer_probe_ids_equal": bool(all_probes_match),
        "draft_len_tokenizer": draft_len,
        "target_len_tokenizer": target_len,
        "tokenizer_compatibility_probes": probe_results,
    }


def prompt_to_token_ids(tokenizer, prompt: Any) -> tuple[list[int], str]:
    if isinstance(prompt, (list, tuple)) and all(isinstance(item, int) for item in prompt):
        return [int(item) for item in prompt], "pretokenized"

    if isinstance(prompt, (list, tuple)) and all(isinstance(item, dict) for item in prompt):
        messages = list(prompt)
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return tokenizer.encode(rendered, add_special_tokens=False), "chat_template"
        text = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
        return tokenizer.encode(text), "plain_text"

    if isinstance(prompt, str):
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return tokenizer.encode(rendered, add_special_tokens=False), "chat_template"
        return tokenizer.encode(prompt), "plain_text"

    return list(prompt), "pretokenized"
