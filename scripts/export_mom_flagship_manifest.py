from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.config import load_config
from aom.data.loaders import load_disamb_pairs_with_manifest
from aom.data.schemas import DisambPair, PromptSide
from aom.prompting import PromptRenderConfig, normalize_prompt_boundary, render_prompt
from aom.provenance.protocol import resolve_protocol_provenance


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_runtime_dependencies() -> tuple[Any, Any, Any, str]:
    """
    Import heavyweight runtime dependencies lazily so `--help` stays lightweight.
    Returns:
      AutoTokenizer class,
      SingleTokenSelectionError class,
      select_single_token_continuation function,
      transformers version string.
    """
    import transformers
    from transformers import AutoTokenizer

    from aom.mechanistic.logit_lens import SingleTokenSelectionError, select_single_token_continuation

    return AutoTokenizer, SingleTokenSelectionError, select_single_token_continuation, str(
        getattr(transformers, "__version__", "")
    )


def _parse_csv_set(raw: str) -> set[str]:
    out = {x.strip() for x in str(raw or "").split(",") if x.strip()}
    return set(out)


def _load_pair_ids(path_raw: str) -> set[str]:
    raw = str(path_raw or "").strip()
    if not raw:
        return set()
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"--pair_ids_path not found: {str(path)}")
    if path.is_dir():
        raise IsADirectoryError(f"--pair_ids_path must be a file, got directory: {str(path)}")
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.add(s)
    return out


def _resolve_protocol_provenance(*, protocol_path_raw: str, protocol_sha256_raw: str) -> tuple[str, str, str, str, bool]:
    prov = resolve_protocol_provenance(
        protocol_path_raw=str(protocol_path_raw or ""),
        protocol_sha256_raw=str(protocol_sha256_raw or ""),
        require_path_for_sha=True,
        require_frozen=True,
    )
    if not str(prov.protocol_path):
        raise ValueError("--protocol_path is required")
    return (
        str(prov.protocol_path),
        str(prov.protocol_sha256),
        str(prov.protocol_name),
        str(prov.protocol_sha256_source),
        bool(prov.protocol_sha256_verified),
    )


def _resolve_axis_labels(
    *,
    protocol_path: str,
    refusal_label_raw: str,
    guidance_label_raw: str,
) -> tuple[str, str]:
    cfg = load_config(protocol_path)
    task = cfg.get("task", None)
    protocol_refusal = ""
    protocol_guidance = ""
    if isinstance(task, Mapping):
        protocol_refusal = str(task.get("refusal_label", "") or "").strip()
        protocol_guidance = str(task.get("guidance_label", "") or "").strip()

    refusal = str(refusal_label_raw or "").strip() or protocol_refusal or "R"
    guidance = str(guidance_label_raw or "").strip() or protocol_guidance or "G"
    if not refusal:
        raise ValueError("Could not resolve refusal label (set --refusal_label or task.refusal_label in protocol)")
    if not guidance:
        raise ValueError("Could not resolve guidance label (set --guidance_label or task.guidance_label in protocol)")
    if refusal == guidance:
        raise ValueError("refusal_label and guidance_label must be different")
    return str(refusal), str(guidance)


def _render_disamb_prompts(
    *,
    items: List[DisambPair],
    tokenizer: Any,
    prompt_mode: str,
    prompt_input: str,
    system_prompt: str | None,
    add_generation_prompt: bool,
    normalize_boundaries: bool,
    boundary_check: str,
) -> List[DisambPair]:
    if str(prompt_mode) not in {"raw", "chat_template"}:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")
    if str(prompt_input) not in {"full_prompt", "user_message"}:
        raise ValueError(f"Unknown prompt_input: {prompt_input!r}")
    if prompt_mode == "raw" and prompt_input != "full_prompt":
        raise ValueError("raw mode expects --prompt_input full_prompt")
    if prompt_mode == "chat_template" and prompt_input != "user_message":
        raise ValueError("chat_template mode expects --prompt_input user_message")
    if boundary_check not in {"off", "warn", "error"}:
        raise ValueError(f"Invalid boundary_check: {boundary_check!r}")

    cfg = PromptRenderConfig(
        prompt_mode=str(prompt_mode), system_prompt=system_prompt, add_generation_prompt=bool(add_generation_prompt)
    )

    def _get_source_text(side: PromptSide) -> str:
        # DisambPair currently stores a single `prompt` field; user_message/full_prompt
        # selects how that text is interpreted, not a different data column.
        if prompt_input == "full_prompt":
            return str(side.prompt)
        if prompt_input == "user_message":
            return str(side.prompt)
        raise ValueError(f"Unknown prompt_input: {prompt_input!r}")

    def _render_one(text: str) -> str:
        if prompt_mode == "raw":
            return str(text)
        rendered, _meta = render_prompt(tokenizer, str(text), cfg)
        return str(rendered)

    def _has_stable_boundary(text: str) -> bool:
        if not text:
            return True
        return bool(text[-1].isspace())

    def _apply_boundary_policy(text: str) -> str:
        out = normalize_prompt_boundary(text) if normalize_boundaries else text
        stable = _has_stable_boundary(out)
        if boundary_check == "warn" and not stable:
            warnings.warn(
                f"Prompt boundary may be unstable (mode={prompt_mode}): prompt does not end with whitespace/newline.",
                UserWarning,
                stacklevel=2,
            )
        elif boundary_check == "error" and not stable:
            raise ValueError("Prompt boundary check failed: prompt does not end with whitespace/newline")
        return str(out)

    out_items: List[DisambPair] = []
    for it in items:
        a_text = _apply_boundary_policy(_render_one(_get_source_text(it.a)))
        b_text = _apply_boundary_policy(_render_one(_get_source_text(it.b)))
        out_items.append(replace(it, a=replace(it.a, prompt=a_text), b=replace(it.b, prompt=b_text)))
    return out_items


def _load_selection(path_raw: str) -> Dict[str, Any]:
    raw = str(path_raw or "").strip()
    if not raw:
        return {}
    p = Path(raw)
    if not p.exists():
        raise FileNotFoundError(f"--selection_json not found: {str(p)}")
    if p.is_dir():
        raise IsADirectoryError(f"--selection_json must be a file, got directory: {str(p)}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping):
        raise ValueError("--selection_json must contain a JSON object")
    return {str(k): v for k, v in obj.items()}


def _normalize_layer_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("selected_layers must be a list")
    out: List[int] = []
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError("selected_layers must be a list[int]")
        out.append(int(x))
    return out


def _normalize_heads_map(raw: Any) -> Dict[str, List[int]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("selected_heads must be a mapping[layer -> list[int]]")
    out: Dict[str, List[int]] = {}
    for k, v in raw.items():
        key = str(k)
        if not isinstance(v, list):
            raise ValueError(f"selected_heads[{key!r}] must be a list[int]")
        vals: List[int] = []
        for h in v:
            if isinstance(h, bool) or not isinstance(h, int):
                raise ValueError(f"selected_heads[{key!r}] must be a list[int]")
            vals.append(int(h))
        out[key] = vals
    return out


def _selection_for_item(selection: Mapping[str, Any], *, pair_id: str, side: str) -> tuple[List[int], Dict[str, List[int]]]:
    default_layers = _normalize_layer_list(selection.get("default_selected_layers", []))
    default_heads = _normalize_heads_map(selection.get("default_selected_heads", {}))

    out_layers = list(default_layers)
    out_heads = dict(default_heads)

    by_pair = selection.get("by_pair_id", {})
    if isinstance(by_pair, Mapping):
        pair_obj = by_pair.get(str(pair_id), None)
        if isinstance(pair_obj, Mapping):
            out_layers = _normalize_layer_list(pair_obj.get("selected_layers", out_layers))
            out_heads = _normalize_heads_map(pair_obj.get("selected_heads", out_heads))
            by_side = pair_obj.get("by_side", None)
            if isinstance(by_side, Mapping):
                side_obj = by_side.get(str(side), None)
                if isinstance(side_obj, Mapping):
                    out_layers = _normalize_layer_list(side_obj.get("selected_layers", out_layers))
                    out_heads = _normalize_heads_map(side_obj.get("selected_heads", out_heads))

    flat_key = f"{pair_id}:{side}"
    flat_obj = selection.get(flat_key, None)
    if isinstance(flat_obj, Mapping):
        out_layers = _normalize_layer_list(flat_obj.get("selected_layers", out_layers))
        out_heads = _normalize_heads_map(flat_obj.get("selected_heads", out_heads))

    out_layers = sorted({int(x) for x in out_layers})
    for layer_s, hs in list(out_heads.items()):
        dedup = sorted({int(x) for x in hs})
        out_heads[str(layer_s)] = dedup
    return out_layers, out_heads


def _extract_direction_id(it: DisambPair, *, side: str) -> str:
    md = it.metadata if isinstance(it.metadata, Mapping) else {}
    if "direction_ids" in md and isinstance(md["direction_ids"], Mapping):
        v = md["direction_ids"].get(str(side), None)
        if isinstance(v, str) and v.strip():
            return str(v)
    key_side = f"direction_id_{side}"
    if isinstance(md.get(key_side, None), str) and str(md[key_side]).strip():
        return str(md[key_side])
    if isinstance(md.get("direction_id", None), str) and str(md["direction_id"]).strip():
        return str(md["direction_id"])

    if side == "a":
        return f"{it.b.expected_label}_to_{it.a.expected_label}"
    return f"{it.a.expected_label}_to_{it.b.expected_label}"


def _extract_template_id(it: DisambPair) -> str:
    md = it.metadata if isinstance(it.metadata, Mapping) else {}
    if isinstance(md.get("template_id", None), str) and str(md["template_id"]).strip():
        return str(md["template_id"])
    return ""


def _extract_risk_domain(it: DisambPair) -> str:
    md = it.metadata if isinstance(it.metadata, Mapping) else {}
    if isinstance(md.get("risk_domain", None), str) and str(md["risk_domain"]).strip():
        return str(md["risk_domain"])
    return ""


def _iter_sides(raw: str) -> Iterable[str]:
    if raw == "a":
        return ("a",)
    if raw == "b":
        return ("b",)
    if raw == "ab":
        return ("a", "b")
    raise ValueError(f"Unsupported --sides value: {raw!r}")


def _encode_ids(tokenizer: Any, text: str) -> List[int]:
    enc = tokenizer(str(text), return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0].tolist()
    return [int(x) for x in ids]


def _single_token_event_check(
    *,
    tokenizer: Any,
    prompt_text: str,
    token_text: str,
    token_id: int,
) -> Dict[str, Any]:
    prompt_ids = _encode_ids(tokenizer, str(prompt_text))
    token_ids = _encode_ids(tokenizer, str(token_text))
    prompt_plus_ids = _encode_ids(tokenizer, str(prompt_text) + str(token_text))

    reasons: List[str] = []
    if len(token_ids) != 1:
        reasons.append(f"token_text_not_single_token(len={len(token_ids)})")
    if len(token_ids) >= 1 and int(token_ids[0]) != int(token_id):
        reasons.append(f"token_id_mismatch(tokenizer={int(token_ids[0])},manifest={int(token_id)})")
    if len(prompt_plus_ids) < len(prompt_ids):
        reasons.append("prompt_plus_shorter_than_prompt")
    else:
        if prompt_plus_ids[: len(prompt_ids)] != prompt_ids:
            reasons.append("prefix_instability(prompt_prefix_changed)")
        if len(prompt_plus_ids) != len(prompt_ids) + 1:
            reasons.append(
                f"prompt_plus_len_delta_not_one(delta={len(prompt_plus_ids) - len(prompt_ids)})"
            )
        elif int(prompt_plus_ids[len(prompt_ids)]) != int(token_id):
            reasons.append(
                f"next_token_id_mismatch(next={int(prompt_plus_ids[len(prompt_ids)])},manifest={int(token_id)})"
            )

    return {
        "valid": len(reasons) == 0,
        "reason": "" if not reasons else ";".join(reasons),
        "prompt_len": int(len(prompt_ids)),
        "token_text_len": int(len(token_ids)),
        "prompt_plus_len": int(len(prompt_plus_ids)),
    }


def _load_system_prompt(args: argparse.Namespace) -> str | None:
    system_prompt_file = str(getattr(args, "system_prompt_file", "") or "").strip()
    system_prompt_inline = str(getattr(args, "system_prompt", "") or "")
    if system_prompt_file and system_prompt_inline.strip():
        raise ValueError("Use at most one of --system_prompt_file or --system_prompt")
    if system_prompt_file:
        return Path(system_prompt_file).read_text(encoding="utf-8")
    if system_prompt_inline.strip():
        return system_prompt_inline
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MoM flagship item manifest for TransformerLens analysis.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--model_revision", type=str, default="", help="Optional model revision (recorded for provenance).")
    p.add_argument("--tokenizer_name", type=str, default="", help="Optional tokenizer path/id; defaults to model.")
    p.add_argument("--tokenizer_revision", type=str, default="", help="Optional tokenizer revision.")
    p.add_argument("--disamb_path", type=str, default=str(ROOT / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--output_path", type=str, default=str(ROOT / "results" / "mom_flagship_manifest.json"))
    p.add_argument("--protocol_path", type=str, default=str(ROOT / "configs" / "mom_flagship_protocol.yaml"))
    p.add_argument("--protocol_sha256", type=str, default="")
    p.add_argument("--selection_json", type=str, default="", help="Optional selected layer/head metadata JSON.")
    p.add_argument(
        "--require_selection_json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed when --selection_json is missing (default: true for flagship runs).",
    )
    p.add_argument("--pair_ids", type=str, default="", help="Optional comma-separated pair_id allowlist.")
    p.add_argument("--pair_ids_path", type=str, default="", help="Optional newline-delimited pair_id allowlist.")
    p.add_argument("--max_pairs", type=int, default=0, help="0 means no limit.")
    p.add_argument("--sides", type=str, default="ab", choices=["a", "b", "ab"])

    p.add_argument("--refusal_label", type=str, default="")
    p.add_argument("--guidance_label", type=str, default="")
    p.add_argument(
        "--require_expected_labels_match_axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when expected labels are not in {refusal_label, guidance_label}.",
    )

    p.add_argument("--prompt_input", type=str, default="full_prompt", choices=["full_prompt", "user_message"])
    p.add_argument("--prompt_mode", type=str, default="raw", choices=["raw", "chat_template"])
    p.add_argument("--system_prompt_file", type=str, default="")
    p.add_argument("--system_prompt", type=str, default="")
    p.add_argument(
        "--add_generation_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="chat_template rendering option.",
    )
    p.add_argument("--normalize_boundaries", action="store_true")
    p.add_argument("--boundary_check", type=str, default="warn", choices=["off", "warn", "error"])
    p.add_argument(
        "--fail_on_tokenization_instability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed when prompt+label tokenization is unstable at the boundary.",
    )

    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    AutoTokenizer, SingleTokenSelectionError, select_single_token_continuation, transformers_version = (
        _load_runtime_dependencies()
    )

    (
        protocol_path,
        protocol_sha256,
        protocol_name,
        protocol_sha_source,
        protocol_sha_verified,
    ) = _resolve_protocol_provenance(
        protocol_path_raw=str(getattr(args, "protocol_path", "") or ""),
        protocol_sha256_raw=str(getattr(args, "protocol_sha256", "") or ""),
    )
    refusal_label, guidance_label = _resolve_axis_labels(
        protocol_path=str(protocol_path),
        refusal_label_raw=str(getattr(args, "refusal_label", "") or ""),
        guidance_label_raw=str(getattr(args, "guidance_label", "") or ""),
    )

    disamb_items, disamb_manifest = load_disamb_pairs_with_manifest(
        str(getattr(args, "disamb_path", "")),
        role="disamb",
        error_policy="raise",
    )

    tokenizer_name = str(getattr(args, "tokenizer_name", "") or "").strip() or str(args.model_name_or_path)
    tokenizer_revision = str(getattr(args, "tokenizer_revision", "") or "").strip()
    model_revision = str(getattr(args, "model_revision", "") or "").strip()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision or None,
        local_files_only=bool(getattr(args, "local_files_only", False)),
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        use_fast=True,
    )

    disamb_items = _render_disamb_prompts(
        items=list(disamb_items),
        tokenizer=tokenizer,
        prompt_mode=str(getattr(args, "prompt_mode", "raw")),
        prompt_input=str(getattr(args, "prompt_input", "full_prompt")),
        system_prompt=_load_system_prompt(args),
        add_generation_prompt=bool(getattr(args, "add_generation_prompt", True)),
        normalize_boundaries=bool(getattr(args, "normalize_boundaries", False)),
        boundary_check=str(getattr(args, "boundary_check", "warn")),
    )

    allow_ids = _parse_csv_set(str(getattr(args, "pair_ids", "") or "")) | _load_pair_ids(
        str(getattr(args, "pair_ids_path", "") or "")
    )
    if allow_ids:
        disamb_items = [it for it in disamb_items if str(it.pair_id) in allow_ids]
    max_pairs = int(getattr(args, "max_pairs", 0) or 0)
    if max_pairs > 0:
        disamb_items = disamb_items[:max_pairs]

    selection_raw = str(getattr(args, "selection_json", "") or "").strip()
    if bool(getattr(args, "require_selection_json", True)) and not selection_raw:
        raise ValueError("--selection_json is required for flagship manifest export (set --no-require_selection_json to override)")
    selection = _load_selection(selection_raw)

    entries: List[Dict[str, Any]] = []
    for it in disamb_items:
        for side_name in _iter_sides(str(getattr(args, "sides", "ab"))):
            side: PromptSide = it.a if side_name == "a" else it.b

            if bool(getattr(args, "require_expected_labels_match_axis", True)):
                allowed = {str(refusal_label), str(guidance_label)}
                if str(side.expected_label) not in allowed:
                    raise ValueError(
                        f"pair_id={it.pair_id} side={side_name}: expected_label={side.expected_label!r} "
                        f"not in axis labels {sorted(allowed)!r}"
                    )

            if refusal_label not in it.choices:
                raise ValueError(f"pair_id={it.pair_id}: refusal_label={refusal_label!r} missing from choices")
            if guidance_label not in it.choices:
                raise ValueError(f"pair_id={it.pair_id}: guidance_label={guidance_label!r} missing from choices")

            try:
                refusal_cont, refusal_tok = select_single_token_continuation(tokenizer, it.choices[refusal_label])
            except SingleTokenSelectionError as e:
                raise ValueError(
                    f"pair_id={it.pair_id}: refusal_label={refusal_label!r} has no single-token continuation"
                ) from e
            try:
                guidance_cont, guidance_tok = select_single_token_continuation(tokenizer, it.choices[guidance_label])
            except SingleTokenSelectionError as e:
                raise ValueError(
                    f"pair_id={it.pair_id}: guidance_label={guidance_label!r} has no single-token continuation"
                ) from e

            prompt_ids = _encode_ids(tokenizer, str(side.prompt))
            seq_len = int(len(prompt_ids))
            if seq_len < 1:
                raise ValueError(f"pair_id={it.pair_id} side={side_name}: prompt tokenized to empty input_ids")
            target_pos = int(seq_len - 1)

            refusal_event = _single_token_event_check(
                tokenizer=tokenizer,
                prompt_text=str(side.prompt),
                token_text=str(refusal_cont),
                token_id=int(refusal_tok),
            )
            guidance_event = _single_token_event_check(
                tokenizer=tokenizer,
                prompt_text=str(side.prompt),
                token_text=str(guidance_cont),
                token_id=int(guidance_tok),
            )
            token_event_valid = bool(refusal_event["valid"]) and bool(guidance_event["valid"])
            if not token_event_valid and bool(getattr(args, "fail_on_tokenization_instability", True)):
                raise ValueError(
                    f"pair_id={it.pair_id} side={side_name}: tokenization instability "
                    f"(refusal={refusal_event['reason']!r}, guidance={guidance_event['reason']!r})"
                )

            selected_layers, selected_heads = _selection_for_item(
                selection,
                pair_id=str(it.pair_id),
                side=str(side_name),
            )

            entries.append(
                {
                    "model_name_or_path": str(args.model_name_or_path),
                    "tokenizer_name": str(tokenizer_name),
                    "protocol_name": str(protocol_name),
                    "protocol_sha256": str(protocol_sha256),
                    "pair_id": str(it.pair_id),
                    "side": str(side_name),
                    "direction_id": _extract_direction_id(it, side=str(side_name)),
                    "template_id": _extract_template_id(it),
                    "risk_domain": _extract_risk_domain(it),
                    "prompt_text": str(side.prompt),
                    "target_pos": int(target_pos),
                    "decision_token_pos": int(seq_len),
                    "refusal_label": str(refusal_label),
                    "guidance_label": str(guidance_label),
                    "refusal_token_text": str(refusal_cont),
                    "guidance_token_text": str(guidance_cont),
                    "refusal_token_id": int(refusal_tok),
                    "guidance_token_id": int(guidance_tok),
                    "token_event_valid": bool(token_event_valid),
                    "token_event_refusal_valid": bool(refusal_event["valid"]),
                    "token_event_refusal_reason": str(refusal_event["reason"]),
                    "token_event_guidance_valid": bool(guidance_event["valid"]),
                    "token_event_guidance_reason": str(guidance_event["reason"]),
                    "selected_layers": [int(x) for x in selected_layers],
                    "selected_heads": {str(k): [int(h) for h in hs] for k, hs in selected_heads.items()},
                }
            )

    entries.sort(key=lambda r: (str(r["pair_id"]), str(r["side"])))

    out_path = Path(str(getattr(args, "output_path", "") or "")).expanduser().resolve()
    disamb_path = Path(str(getattr(args, "disamb_path", ""))).expanduser().resolve()
    disamb_sha256 = _sha256_file(disamb_path)
    selection_path_raw = str(getattr(args, "selection_json", "") or "").strip()
    selection_path = Path(selection_path_raw).expanduser().resolve() if selection_path_raw else None
    selection_sha256 = _sha256_file(selection_path) if selection_path is not None else ""

    out: Dict[str, Any] = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(model_revision),
        "tokenizer_name": str(tokenizer_name),
        "tokenizer_revision": str(tokenizer_revision or model_revision),
        "transformers_version": str(transformers_version),
        "local_files_only": bool(getattr(args, "local_files_only", False)),
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
        "protocol_name": str(protocol_name),
        "protocol_path": str(protocol_path),
        "protocol_path_resolved": str(protocol_path),
        "protocol_sha256": str(protocol_sha256),
        "protocol_sha256_source": str(protocol_sha_source),
        "protocol_sha256_verified": bool(protocol_sha_verified),
        "disamb_path": str(disamb_path),
        "disamb_sha256": str(disamb_sha256),
        "selection_json_path": "" if selection_path is None else str(selection_path),
        "selection_json_sha256": str(selection_sha256),
        "prompt_mode": str(getattr(args, "prompt_mode", "raw")),
        "prompt_input": str(getattr(args, "prompt_input", "full_prompt")),
        "boundary_check": str(getattr(args, "boundary_check", "warn")),
        "normalize_boundaries": bool(getattr(args, "normalize_boundaries", False)),
        "refusal_label": str(refusal_label),
        "guidance_label": str(guidance_label),
        "n_pairs_loaded": int(len(disamb_items)),
        "n_entries": int(len(entries)),
        "dataset_manifest": disamb_manifest.as_dict(),
        "items": entries,
    }
    _write_json(out_path, out)
    print(f"Wrote {len(entries)} entries to {str(out_path)}", flush=True)


if __name__ == "__main__":
    main()
