from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class RunningStat:
    n: int = 0
    sum: float = 0.0
    min: float = math.inf
    max: float = -math.inf

    def update(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.n += 1
        self.sum += float(x)
        self.min = min(self.min, float(x))
        self.max = max(self.max, float(x))

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")


@dataclass(frozen=True)
class AoMRun:
    source_csv: str
    model: str
    arch: Optional[str]
    seed: Optional[int]
    requested_device: Optional[str]
    device: Optional[str]
    torch_dtype_requested: Optional[str]
    model_param_dtype: Optional[str]
    transformers_version: Optional[str]
    torch_version: Optional[str]
    python_version: Optional[str]
    git_commit: Optional[str]
    bootstrap_n: Optional[int]
    ci: Optional[float]
    disamb_n_pairs_total: Optional[int]
    cf_n_items_total: Optional[int]
    coh_n_items_total: Optional[int]
    aom_composite: Optional[float]
    disamb_accuracy: Optional[float]
    cf_shift_direction_accuracy: Optional[float]
    coh_constraint_accuracy: Optional[float]
    cpt_flip_rate_at_best_layer: Optional[float]
    cpt_mean_argmax_layer: Optional[float]
    cpt_mean_max_effect: Optional[float]
    cpt_spec_win_rate: Optional[float]
    clt_cpt_mean_max_effect: Optional[float]


@dataclass
class InductionFileSummary:
    source_csv: str
    models: List[str]
    arch: Optional[str]
    baseline_modes: List[str]
    base_lens: List[int]
    repeats: List[int]
    seeds: List[int]
    devices: List[str]
    torch_dtypes: List[str]
    n_rows: int
    layer_stats: Dict[int, RunningStat]


@dataclass
class LogitLensFileSummary:
    source_csv: str
    models: List[str]
    arches: List[str]
    lenses: List[str]
    n_rows: int
    state_index_min: Optional[int]
    state_index_max: Optional[int]
    block_index_min: Optional[int]
    block_index_max: Optional[int]
    logit_diff: RunningStat
    final_logit_diff: RunningStat


@dataclass
class CSVFileSummary:
    source_csv: str
    kind: str
    n_rows: int
    n_cols: int
    models: List[str]
    notes: List[str]


def _as_str(x: object) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _as_int(x: object) -> Optional[int]:
    s = _as_str(x)
    if s is None:
        return None
    if s.lower() in {"nan", "none"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _as_float(x: object) -> Optional[float]:
    s = _as_str(x)
    if s is None:
        return None
    if s.lower() in {"nan", "none"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    headers = list(headers)
    out: List[str] = []
    out.append("| " + " | ".join(_md_escape(h) for h in headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        rr = list(r)
        if len(rr) != len(headers):
            raise ValueError(f"Row has {len(rr)} columns but expected {len(headers)}: {rr!r}")
        out.append("| " + " | ".join(_md_escape(x) for x in rr) + " |")
    return "\n".join(out)


def _fmt_int(x: Optional[int]) -> str:
    return "" if x is None else str(int(x))


def _fmt_float(x: Optional[float], *, digits: int = 4) -> str:
    if x is None or not math.isfinite(float(x)):
        return ""
    x = float(x)
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    return f"{x:.{digits}f}"


def _summarize_list(xs: Iterable[str], *, max_items: int = 4) -> str:
    uniq = sorted({str(x) for x in xs if str(x).strip()})
    if not uniq:
        return ""
    if len(uniq) <= max_items:
        return ", ".join(uniq)
    return ", ".join(uniq[:max_items]) + f", … (+{len(uniq) - max_items} more)"


def _extract_layer_indices(columns: Sequence[str], *, prefix: str) -> List[int]:
    out: List[int] = []
    for c in columns:
        cs = str(c).strip()
        if not cs.startswith(prefix):
            continue
        idx = _as_int(cs[len(prefix) :])
        if idx is None:
            continue
        out.append(int(idx))
    return sorted(set(out))


def _classify_csv(path: Path, columns: Sequence[str]) -> str:
    colset = {str(c) for c in columns}
    name = path.name.lower()
    if "aom_composite" in colset:
        return "aom_eval"
    if {"induction_advantage", "head", "layer"} <= colset:
        return "induction"
    if ("logit_lens" in name) or ({"lens", "logit_diff"} <= colset):
        if "state_index" in colset or "block_index" in colset:
            return "logit_lens_trace"
        return "logit_lens"
    return "csv"


def _read_csv_header(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return []
    return [str(x).strip() for x in header]


def _read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        for row in reader:
            yield {str(k): (str(v) if v is not None else "") for k, v in row.items()}


def _summarize_aom_eval_csv(
    path: Path,
    *,
    rel: str,
    max_rows: Optional[int],
) -> Tuple[List[AoMRun], CSVFileSummary]:
    runs: List[AoMRun] = []

    models: Counter[str] = Counter()
    n_rows = 0
    columns = _read_csv_header(path)

    for row in _read_csv_rows(path):
        n_rows += 1
        if max_rows is not None and n_rows > max_rows:
            break

        model = _as_str(row.get("model")) or "unknown"
        models[model] += 1

        runs.append(
            AoMRun(
                source_csv=rel,
                model=model,
                arch=_as_str(row.get("arch")),
                seed=_as_int(row.get("seed")),
                requested_device=_as_str(row.get("requested_device")),
                device=_as_str(row.get("device")),
                torch_dtype_requested=_as_str(row.get("torch_dtype_requested")),
                model_param_dtype=_as_str(row.get("model_param_dtype")),
                transformers_version=_as_str(row.get("transformers_version")),
                torch_version=_as_str(row.get("torch_version")),
                python_version=_as_str(row.get("python_version")),
                git_commit=_as_str(row.get("git_commit")),
                bootstrap_n=_as_int(row.get("bootstrap_n")),
                ci=_as_float(row.get("ci")),
                disamb_n_pairs_total=_as_int(row.get("disamb_n_pairs_total")),
                cf_n_items_total=_as_int(row.get("cf_n_items_total")),
                coh_n_items_total=_as_int(row.get("coh_n_items_total")),
                aom_composite=_as_float(row.get("aom_composite")),
                disamb_accuracy=_as_float(row.get("disamb_accuracy")),
                cf_shift_direction_accuracy=_as_float(row.get("cf_shift_direction_accuracy")),
                coh_constraint_accuracy=_as_float(row.get("coh_constraint_accuracy")),
                cpt_flip_rate_at_best_layer=_as_float(row.get("cpt_flip_rate_at_best_layer")),
                cpt_mean_argmax_layer=_as_float(row.get("cpt_mean_argmax_layer")),
                cpt_mean_max_effect=_as_float(row.get("cpt_mean_max_effect")),
                cpt_spec_win_rate=_as_float(row.get("cpt_spec_win_rate")),
                clt_cpt_mean_max_effect=_as_float(row.get("clt_cpt_mean_max_effect")),
            )
        )

    notes: List[str] = []
    cpt_layers = _extract_layer_indices(columns, prefix="cpt_effect_layer_")
    if cpt_layers:
        notes.append(f"cpt_effect layers: {min(cpt_layers)}..{max(cpt_layers)} ({len(cpt_layers)} total)")
    sham_layers = _extract_layer_indices(columns, prefix="cpt_sham_effect_layer_")
    if sham_layers:
        notes.append(f"cpt_sham_effect layers: {min(sham_layers)}..{max(sham_layers)} ({len(sham_layers)} total)")
    clt_layers = _extract_layer_indices(columns, prefix="clt_cpt_effect_layer_")
    if clt_layers:
        notes.append(f"clt_cpt_effect layers: {min(clt_layers)}..{max(clt_layers)} ({len(clt_layers)} total)")
    clt_sham_layers = _extract_layer_indices(columns, prefix="clt_cpt_sham_effect_layer_")
    if clt_sham_layers:
        notes.append(f"clt_cpt_sham_effect layers: {min(clt_sham_layers)}..{max(clt_sham_layers)} ({len(clt_sham_layers)} total)")
    if max_rows is not None and n_rows > max_rows:
        notes.append(f"truncated at {max_rows} rows")

    summary = CSVFileSummary(
        source_csv=rel,
        kind="aom_eval",
        n_rows=min(n_rows, max_rows) if max_rows is not None else n_rows,
        n_cols=len(columns),
        models=sorted(models.keys()),
        notes=notes,
    )
    return runs, summary


def _summarize_induction_csv(
    path: Path,
    *,
    rel: str,
    max_rows: Optional[int],
) -> Tuple[InductionFileSummary, CSVFileSummary]:
    models: Set[str] = set()
    baseline_modes: Set[str] = set()
    base_lens: Set[int] = set()
    repeats: Set[int] = set()
    seeds: Set[int] = set()
    devices: Set[str] = set()
    torch_dtypes: Set[str] = set()
    arches: Set[str] = set()

    layer_stats: DefaultDict[int, RunningStat] = defaultdict(RunningStat)

    n_rows = 0
    for row in _read_csv_rows(path):
        n_rows += 1
        if max_rows is not None and n_rows > max_rows:
            break
        model = _as_str(row.get("model"))
        if model:
            models.add(model)
        arch = _as_str(row.get("arch"))
        if arch:
            arches.add(arch)
        bm = _as_str(row.get("baseline_mode"))
        if bm:
            baseline_modes.add(bm)
        bl = _as_int(row.get("base_len"))
        if bl is not None:
            base_lens.add(int(bl))
        rep = _as_int(row.get("repeats"))
        if rep is not None:
            repeats.add(int(rep))
        seed = _as_int(row.get("seed"))
        if seed is not None:
            seeds.add(int(seed))
        dev = _as_str(row.get("device"))
        if dev:
            devices.add(dev)
        td = _as_str(row.get("torch_dtype"))
        if td:
            torch_dtypes.add(td)

        layer = _as_int(row.get("layer"))
        adv = _as_float(row.get("induction_advantage"))
        if layer is not None and adv is not None:
            layer_stats[int(layer)].update(float(adv))

    arch = sorted(arches)[0] if arches else None

    induction = InductionFileSummary(
        source_csv=rel,
        models=sorted(models) if models else ["unknown"],
        arch=arch,
        baseline_modes=sorted(baseline_modes),
        base_lens=sorted(base_lens),
        repeats=sorted(repeats),
        seeds=sorted(seeds),
        devices=sorted(devices),
        torch_dtypes=sorted(torch_dtypes),
        n_rows=min(n_rows, max_rows) if max_rows is not None else n_rows,
        layer_stats=dict(layer_stats),
    )

    columns = _read_csv_header(path)
    notes: List[str] = []
    if induction.layer_stats:
        layers = sorted(induction.layer_stats.keys())
        notes.append(f"layers observed: {layers[0]}..{layers[-1]} ({len(layers)} total)")
    if max_rows is not None and n_rows > max_rows:
        notes.append(f"truncated at {max_rows} rows")

    summary = CSVFileSummary(
        source_csv=rel,
        kind="induction",
        n_rows=induction.n_rows,
        n_cols=len(columns),
        models=induction.models,
        notes=notes,
    )
    return induction, summary


def _summarize_logit_lens_csv(
    path: Path,
    *,
    rel: str,
    max_rows: Optional[int],
) -> Tuple[LogitLensFileSummary, CSVFileSummary]:
    models: Set[str] = set()
    arches: Set[str] = set()
    lenses: Set[str] = set()
    n_rows = 0

    state_min: Optional[int] = None
    state_max: Optional[int] = None
    block_min: Optional[int] = None
    block_max: Optional[int] = None
    logit_diff = RunningStat()
    final_logit_diff = RunningStat()

    for row in _read_csv_rows(path):
        n_rows += 1
        if max_rows is not None and n_rows > max_rows:
            break
        m = _as_str(row.get("model"))
        if m:
            models.add(m)
        a = _as_str(row.get("arch"))
        if a:
            arches.add(a)
        l = _as_str(row.get("lens"))
        if l:
            lenses.add(l)

        si = _as_int(row.get("state_index"))
        if si is not None:
            state_min = si if state_min is None else min(state_min, si)
            state_max = si if state_max is None else max(state_max, si)
        bi = _as_int(row.get("block_index"))
        if bi is not None:
            block_min = bi if block_min is None else min(block_min, bi)
            block_max = bi if block_max is None else max(block_max, bi)

        ld = _as_float(row.get("logit_diff"))
        if ld is not None:
            logit_diff.update(ld)
        fld = _as_float(row.get("final_logit_diff"))
        if fld is not None:
            final_logit_diff.update(fld)

    ll = LogitLensFileSummary(
        source_csv=rel,
        models=sorted(models) if models else ["unknown"],
        arches=sorted(arches),
        lenses=sorted(lenses),
        n_rows=min(n_rows, max_rows) if max_rows is not None else n_rows,
        state_index_min=state_min,
        state_index_max=state_max,
        block_index_min=block_min,
        block_index_max=block_max,
        logit_diff=logit_diff,
        final_logit_diff=final_logit_diff,
    )

    columns = _read_csv_header(path)
    notes: List[str] = []
    if ll.state_index_min is not None and ll.state_index_max is not None:
        notes.append(f"state_index: {ll.state_index_min}..{ll.state_index_max}")
    if ll.block_index_min is not None and ll.block_index_max is not None:
        notes.append(f"block_index: {ll.block_index_min}..{ll.block_index_max}")
    if max_rows is not None and n_rows > max_rows:
        notes.append(f"truncated at {max_rows} rows")

    summary = CSVFileSummary(
        source_csv=rel,
        kind="logit_lens",
        n_rows=ll.n_rows,
        n_cols=len(columns),
        models=ll.models,
        notes=notes,
    )
    return ll, summary


def _summarize_generic_csv(path: Path, *, rel: str, kind: str, max_rows: Optional[int]) -> CSVFileSummary:
    columns = _read_csv_header(path)
    models: Set[str] = set()
    n_rows = 0
    for row in _read_csv_rows(path):
        n_rows += 1
        if max_rows is not None and n_rows > max_rows:
            break
        m = _as_str(row.get("model"))
        if m:
            models.add(m)
    notes: List[str] = []
    if max_rows is not None and n_rows > max_rows:
        notes.append(f"truncated at {max_rows} rows")
    return CSVFileSummary(
        source_csv=rel,
        kind=kind,
        n_rows=min(n_rows, max_rows) if max_rows is not None else n_rows,
        n_cols=len(columns),
        models=sorted(models),
        notes=notes,
    )


def _pick_best_run(runs: Sequence[AoMRun]) -> Optional[AoMRun]:
    if not runs:
        return None

    def key(r: AoMRun) -> Tuple[float, int, int, str]:
        aom = r.aom_composite if r.aom_composite is not None else float("-inf")
        b = r.bootstrap_n if r.bootstrap_n is not None else -1
        seed_key = -(r.seed if r.seed is not None else 10**9)
        return float(aom), int(b), int(seed_key), str(r.source_csv)

    return max(runs, key=key)


def generate_results_report(
    results_dir: str | Path = "results",
    *,
    max_rows_per_csv: Optional[int] = None,
    include_file_inventory: bool = True,
) -> str:
    """
    Scan a `results/` directory (CSVs + logs + figures) and return a Markdown report.

    The report is designed to be robust to heterogenous CSV schemas:
    - AoM evaluation CSVs: contain `aom_composite`
    - Induction CSVs: contain `induction_advantage`, `layer`, `head`
    - Logit-lens trace CSVs: contain `lens` + `logit_diff` (and usually `state_index`)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"results_dir not found: {results_path}")
    if not results_path.is_dir():
        raise NotADirectoryError(f"results_dir is not a directory: {results_path}")

    max_rows = None if (max_rows_per_csv is None or int(max_rows_per_csv) <= 0) else int(max_rows_per_csv)

    aom_runs: List[AoMRun] = []
    aom_csv_summaries: List[CSVFileSummary] = []
    induction_summaries: List[InductionFileSummary] = []
    logit_lens_summaries: List[LogitLensFileSummary] = []
    other_csv_summaries: List[CSVFileSummary] = []

    model_to_kinds: DefaultDict[str, Set[str]] = defaultdict(set)

    csv_files = sorted([p for p in results_path.rglob("*.csv") if p.is_file()])
    for p in csv_files:
        rel = str(p.relative_to(results_path))
        header = _read_csv_header(p)
        kind = _classify_csv(p, header)

        if kind == "aom_eval":
            runs, summ = _summarize_aom_eval_csv(p, rel=rel, max_rows=max_rows)
            aom_runs.extend(runs)
            aom_csv_summaries.append(summ)
            for r in runs:
                model_to_kinds[r.model].add("aom_eval")
                if r.cpt_flip_rate_at_best_layer is not None or r.cpt_mean_argmax_layer is not None:
                    model_to_kinds[r.model].add("cpt_patching")
                if r.cpt_spec_win_rate is not None:
                    model_to_kinds[r.model].add("cpt_specificity")
                if r.clt_cpt_mean_max_effect is not None:
                    model_to_kinds[r.model].add("clt_patching")
        elif kind == "induction":
            ind, summ = _summarize_induction_csv(p, rel=rel, max_rows=max_rows)
            induction_summaries.append(ind)
            other_csv_summaries.append(summ)
            for m in ind.models:
                model_to_kinds[m].add("induction")
        elif kind.startswith("logit_lens"):
            ll, summ = _summarize_logit_lens_csv(p, rel=rel, max_rows=max_rows)
            logit_lens_summaries.append(ll)
            other_csv_summaries.append(summ)
            for m in ll.models:
                model_to_kinds[m].add("logit_lens")
        else:
            summ = _summarize_generic_csv(p, rel=rel, kind=kind, max_rows=max_rows)
            other_csv_summaries.append(summ)
            for m in summ.models:
                model_to_kinds[m].add("csv")

    # Other (non-CSV) inventory.
    all_files = sorted([p for p in results_path.rglob("*") if p.is_file()])
    by_ext: Counter[str] = Counter()
    total_bytes = 0
    for p in all_files:
        ext = p.suffix.lower() or "<none>"
        by_ext[ext] += 1
        try:
            total_bytes += int(p.stat().st_size)
        except OSError:
            pass

    now = datetime.now().astimezone()
    out: List[str] = []
    out.append(f"# Results report: `{results_path}`")
    out.append("")
    out.append(f"- Generated: {now.isoformat(timespec='seconds')}")
    out.append(f"- CSV files: {len(csv_files)}")
    out.append(f"- Total files: {len(all_files)}")
    out.append(f"- Total size: {total_bytes / (1024 * 1024):.2f} MiB")
    out.append("")

    if by_ext:
        out.append("## Inventory")
        rows = []
        for ext, n in sorted(by_ext.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append([ext, str(n)])
        out.append(_md_table(["ext", "count"], rows))
        out.append("")

    if model_to_kinds:
        out.append("## Models observed")
        rows = []
        for model in sorted(model_to_kinds.keys()):
            rows.append([model, _summarize_list(sorted(model_to_kinds[model]), max_items=10)])
        out.append(_md_table(["model", "experiments"], rows))
        out.append("")

    if aom_runs:
        out.append("## AoM evaluation summary (`aom_eval.py`-style CSVs)")
        by_model: DefaultDict[str, List[AoMRun]] = defaultdict(list)
        for r in aom_runs:
            by_model[r.model].append(r)

        rows = []
        for model in sorted(by_model.keys()):
            best = _pick_best_run(by_model[model])
            if best is None:
                continue
            rows.append(
                [
                    model,
                    str(len(by_model[model])),
                    _fmt_float(best.aom_composite),
                    _fmt_float(best.disamb_accuracy),
                    _fmt_float(best.cf_shift_direction_accuracy),
                    _fmt_float(best.coh_constraint_accuracy),
                    _fmt_float(best.cpt_flip_rate_at_best_layer),
                    _fmt_float(best.cpt_mean_argmax_layer, digits=2),
                    _fmt_float(best.cpt_spec_win_rate),
                    _fmt_int(best.disamb_n_pairs_total),
                    _fmt_int(best.cf_n_items_total),
                    _fmt_int(best.coh_n_items_total),
                    _fmt_int(best.bootstrap_n),
                    _summarize_list([best.requested_device or "", best.device or ""], max_items=2),
                    best.torch_dtype_requested or "",
                    best.transformers_version or "",
                    best.source_csv,
                ]
            )

        out.append(
            _md_table(
                [
                    "model",
                    "n_runs",
                    "aom",
                    "disamb",
                    "cf_shift_dir",
                    "coh_constraint",
                    "cpt_flip@best",
                    "cpt_best_layer",
                    "cpt_spec_win",
                    "disamb_n",
                    "cf_n",
                    "coh_n",
                    "bootstrap_n",
                    "device",
                    "dtype",
                    "tfm",
                    "best_csv",
                ],
                rows,
            )
        )
        out.append("")

        if aom_csv_summaries:
            out.append("### AoM CSV coverage")
            rows2 = []
            for s in sorted(aom_csv_summaries, key=lambda x: x.source_csv):
                rows2.append(
                    [
                        s.source_csv,
                        str(s.n_rows),
                        str(s.n_cols),
                        _summarize_list(s.models, max_items=3),
                        _summarize_list(s.notes, max_items=3),
                    ]
                )
            out.append(_md_table(["file", "rows", "cols", "models", "notes"], rows2))
            out.append("")

    if induction_summaries:
        out.append("## Induction summary")
        rows = []
        for s in sorted(induction_summaries, key=lambda x: x.source_csv):
            best_layer = ""
            best_mean = ""
            if s.layer_stats:
                layer = max(s.layer_stats.keys(), key=lambda k: s.layer_stats[k].mean)
                best_layer = str(layer)
                best_mean = _fmt_float(s.layer_stats[layer].mean)
            rows.append(
                [
                    s.source_csv,
                    _summarize_list(s.models, max_items=2),
                    s.arch or "",
                    _summarize_list(s.baseline_modes, max_items=3),
                    _summarize_list([str(x) for x in s.base_lens], max_items=3),
                    _summarize_list([str(x) for x in s.repeats], max_items=3),
                    best_layer,
                    best_mean,
                    str(len(s.layer_stats)) if s.layer_stats else "0",
                    str(s.n_rows),
                ]
            )
        out.append(
            _md_table(
                [
                    "file",
                    "model",
                    "arch",
                    "baseline_mode",
                    "base_len",
                    "repeats",
                    "best_layer",
                    "best_mean_adv",
                    "n_layers",
                    "rows",
                ],
                rows,
            )
        )
        out.append("")

    if logit_lens_summaries:
        out.append("## Logit lens summary")
        rows = []
        for s in sorted(logit_lens_summaries, key=lambda x: x.source_csv):
            rows.append(
                [
                    s.source_csv,
                    _summarize_list(s.models, max_items=2),
                    _summarize_list(s.arches, max_items=2),
                    _summarize_list(s.lenses, max_items=3),
                    str(s.n_rows),
                    f"{s.state_index_min}..{s.state_index_max}"
                    if s.state_index_min is not None and s.state_index_max is not None
                    else "",
                    f"{s.block_index_min}..{s.block_index_max}"
                    if s.block_index_min is not None and s.block_index_max is not None
                    else "",
                    _fmt_float(s.logit_diff.mean),
                    _fmt_float(s.final_logit_diff.mean),
                ]
            )
        out.append(
            _md_table(
                [
                    "file",
                    "model",
                    "arch",
                    "lens",
                    "rows",
                    "state_index",
                    "block_index",
                    "mean(logit_diff)",
                    "mean(final_logit_diff)",
                ],
                rows,
            )
        )
        out.append("")

    if include_file_inventory:
        out.append("## CSV inventory")
        all_csv_summaries = sorted(other_csv_summaries + aom_csv_summaries, key=lambda x: (x.kind, x.source_csv))
        rows = []
        for s in all_csv_summaries:
            rows.append(
                [
                    s.source_csv,
                    s.kind,
                    str(s.n_rows),
                    str(s.n_cols),
                    _summarize_list(s.models, max_items=3),
                    _summarize_list(s.notes, max_items=3),
                ]
            )
        out.append(_md_table(["file", "kind", "rows", "cols", "models", "notes"], rows))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def write_results_report(
    out_path: str | Path,
    results_dir: str | Path = "results",
    *,
    max_rows_per_csv: Optional[int] = None,
    include_file_inventory: bool = True,
) -> Path:
    """
    Convenience wrapper that writes `generate_results_report(...)` to disk.
    Returns the resolved output path.
    """
    out_p = Path(out_path)
    report = generate_results_report(
        results_dir,
        max_rows_per_csv=max_rows_per_csv,
        include_file_inventory=include_file_inventory,
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(report, encoding="utf-8")
    return out_p.resolve()
