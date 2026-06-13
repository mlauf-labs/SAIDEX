"""Benchmark Runner — run all scenarios against all configured models and compare.

Usage:
    # Run all scenarios with all models
    python benchmarks/run_benchmarks.py

    # Run only a specific group
    python benchmarks/run_benchmarks.py --group invoice
    python benchmarks/run_benchmarks.py --group sentiment
    python benchmarks/run_benchmarks.py --group website
    python benchmarks/run_benchmarks.py --group extras
    python benchmarks/run_benchmarks.py --group vision

    # Run with a specific model override (comma-separated)
    python benchmarks/run_benchmarks.py --models llama3.2:3b,qwen2.5:7b

    # The vision group uses vision-capable models (override with --vision-models)
    python benchmarks/run_benchmarks.py --group vision --vision-models qwen2.5vl:7b

    # Save results to JSON
    python benchmarks/run_benchmarks.py --save results/run_001.json

    # Write the Markdown report to a custom path
    python benchmarks/run_benchmarks.py --report results/report_001.md

    # Skip the Markdown report
    python benchmarks/run_benchmarks.py --no-report

Results are always printed to stdout. By default a timestamped JSON file *and* a
Markdown report are written to benchmarks/results/. The report summarises ranking,
speed, field-level accuracy (where expected values are defined) and per-scenario
extraction details.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Import all scenarios
from .invoice_01 import SCENARIO as INVOICE_01
from .invoice_02 import SCENARIO as INVOICE_02
from .invoice_03 import SCENARIO as INVOICE_03
from .sentiment_01 import SCENARIO as SENTIMENT_01
from .sentiment_02 import SCENARIO as SENTIMENT_02
from .sentiment_03 import SCENARIO as SENTIMENT_03
from .website_cat_01 import SCENARIO as WEBSITE_01
from .website_cat_02 import SCENARIO as WEBSITE_02
from .website_cat_03 import SCENARIO as WEBSITE_03
from .extras_meeting import SCENARIO as MEETING
from .extras_ner import SCENARIO as NER
from .vision_invoice import SCENARIOS as VISION_INVOICE

from pydantic import BaseModel

from ._base import BenchmarkResult, BenchmarkScenario, run_scenario, warmup_model
from ._config import MODELS, OLLAMA_BASE_URL, VISION_MODELS

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

ALL_SCENARIOS: dict[str, list[BenchmarkScenario[Any]]] = {
    "invoice": [INVOICE_01, INVOICE_02, INVOICE_03],
    "sentiment": [SENTIMENT_01, SENTIMENT_02, SENTIMENT_03],
    "website": [WEBSITE_01, WEBSITE_02, WEBSITE_03],
    "extras": [MEETING, NER],
    "vision": list(VISION_INVOICE),
}

# Groups that require vision-capable models instead of the default text models.
VISION_GROUPS: set[str] = {"vision"}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _print_group_summary(
    group_name: str,
    scenarios: list[BenchmarkScenario[Any]],
    all_results: dict[str, list[BenchmarkResult]],
) -> None:
    """Print a compact cross-model summary table for a group of scenarios."""
    print(f"\n{'━' * 72}")
    print(f"  GROUP: {group_name.upper()}")
    print(f"{'━' * 72}")
    header = f"  {'Scenario':<38} {'Model':<20} {'OK':>3}  {'Time':>6}  {'Retries':>7}"
    print(header)
    print(f"  {'-' * 68}")
    for scenario in scenarios:
        results = all_results.get(scenario.name, [])
        for r in results:
            ok = "YES" if r.success else "NO "
            print(f"  {r.scenario_name:<38} {r.model:<20} {ok:>3}  {r.duration_s:>5.1f}s  {r.retries:>7}")
    print()


def _print_speed_ranking(all_results: dict[str, list[BenchmarkResult]]) -> None:
    """Print average response time per model across all scenarios."""
    from collections import defaultdict

    totals: dict[str, list[float]] = defaultdict(list)
    for results in all_results.values():
        for r in results:
            if r.success:
                totals[r.model].append(r.duration_s)

    print(f"\n{'━' * 72}")
    print("  SPEED RANKING (average response time over successful runs)")
    print(f"{'━' * 72}")
    ranked = sorted(totals.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else float("inf"))
    for rank, (model, times) in enumerate(ranked, 1):
        avg = sum(times) / len(times)
        total = sum(times)
        print(f"  #{rank}  {model:<22}  avg={avg:.1f}s  total={total:.1f}s  runs={len(times)}")
    print()


def _print_quality_ranking(all_results: dict[str, list[BenchmarkResult]]) -> None:
    """Print success rate per model."""
    from collections import defaultdict

    success: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for results in all_results.values():
        for r in results:
            total[r.model] += 1
            if r.success:
                success[r.model] += 1

    print(f"{'━' * 72}")
    print("  QUALITY RANKING (extraction success rate)")
    print(f"{'━' * 72}")
    ranked = sorted(total.keys(), key=lambda m: success[m] / total[m] if total[m] else 0, reverse=True)
    for rank, model in enumerate(ranked, 1):
        rate = success[model] / total[model] * 100 if total[model] else 0
        print(f"  #{rank}  {model:<22}  {success[model]}/{total[model]} successful  ({rate:.0f}%)")
    print()


def _save_results(
    all_results: dict[str, list[BenchmarkResult]],
    output_path: Path,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "models": MODELS,
        "scenarios": {},
    }
    for scenario_name, results in all_results.items():
        payload["scenarios"][scenario_name] = [r.as_dict() for r in results]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _normalise(value: Any) -> str:
    """Normalise a value for lenient string comparison."""
    return str(value).strip().lower().replace(" ", "")


def _field_accuracy(
    scenario: BenchmarkScenario[Any],
    result: BenchmarkResult,
) -> tuple[int, int]:
    """Return (correct, total) field matches against the scenario's expected values.

    Only fields listed in ``scenario.expected`` are evaluated. Returns (0, 0)
    when there is nothing to compare against.
    """
    if not scenario.expected or not result.success or result.value is None:
        return (0, len(scenario.expected))

    data = result.value.model_dump()
    correct = 0
    for key, expected_value in scenario.expected.items():
        actual = data.get(key)
        if actual is not None and _normalise(actual) == _normalise(expected_value):
            correct += 1
    return (correct, len(scenario.expected))


def _md_escape(text: Any) -> str:
    """Escape a value so it renders safely inside a Markdown table cell."""
    s = str(text)
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _value_summary(value: BaseModel | None) -> str:
    """Compact one-line summary of an extracted Pydantic model for a table cell."""
    if value is None:
        return "—"
    parts = []
    for k, v in value.model_dump().items():
        if v is None:
            continue
        if isinstance(v, list):
            parts.append(f"{k}=[{len(v)}]")
        elif isinstance(v, str) and len(v) > 40:
            parts.append(f"{k}={v[:37]}…")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "—"


def _scenario_image_link(scenario: BenchmarkScenario[Any], report_dir: Path) -> str | None:
    """Return a Markdown-relative path to a vision scenario's rendered image."""
    from ._base import IMAGE_CACHE_DIR

    safe_name = "".join(c if c.isalnum() else "_" for c in scenario.name)
    image_path = IMAGE_CACHE_DIR / f"{safe_name}.png"
    if not image_path.exists():
        return None
    try:
        rel = os.path.relpath(image_path, report_dir)
    except ValueError:
        return image_path.as_posix()
    return Path(rel).as_posix()


def _build_report_md(
    all_results: dict[str, list[BenchmarkResult]],
    scenarios_by_name: dict[str, BenchmarkScenario[Any]],
    groups: list[str],
    models: list[str],
    report_dir: Path,
) -> str:
    from collections import defaultdict

    lines: list[str] = []
    now = datetime.now()

    # --- Header ---
    lines.append("# LLM Structured Output — Benchmark Report")
    lines.append("")
    lines.append(f"- **Erstellt:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Ollama URL:** `{OLLAMA_BASE_URL}`")
    lines.append(f"- **Gruppen:** {', '.join(groups)}")
    lines.append(f"- **Modelle:** {', '.join(f'`{m}`' for m in models)}")
    total_runs = sum(len(r) for r in all_results.values())
    total_ok = sum(1 for rs in all_results.values() for r in rs if r.success)
    lines.append(f"- **Läufe gesamt:** {total_runs} ({total_ok} erfolgreich)")
    lines.append("")

    # --- Split results into text vs vision scenarios ---
    text_results: dict[str, list[BenchmarkResult]] = {}
    vision_results: dict[str, list[BenchmarkResult]] = {}
    for scenario_name, results in all_results.items():
        scenario = scenarios_by_name.get(scenario_name)
        if scenario is not None and scenario.vision:
            vision_results[scenario_name] = results
        else:
            text_results[scenario_name] = results

    def _emit_ranking(title: str, subset: dict[str, list[BenchmarkResult]]) -> None:
        """Append a ranking + speed table for a subset of scenario results."""
        speed: dict[str, list[float]] = defaultdict(list)
        success_cnt: dict[str, int] = defaultdict(int)
        total_cnt: dict[str, int] = defaultdict(int)
        acc_correct: dict[str, int] = defaultdict(int)
        acc_total: dict[str, int] = defaultdict(int)

        for scenario_name, results in subset.items():
            scenario = scenarios_by_name.get(scenario_name)
            for r in results:
                total_cnt[r.model] += 1
                if r.success:
                    success_cnt[r.model] += 1
                    speed[r.model].append(r.duration_s)
                if scenario is not None:
                    c, t = _field_accuracy(scenario, r)
                    acc_correct[r.model] += c
                    acc_total[r.model] += t

        if not total_cnt:
            return

        lines.append(f"## {title}")
        lines.append("")
        lines.append("| # | Modell | Erfolg | Quote | Feld-Genauigkeit | Ø Zeit | Gesamtzeit |")
        lines.append("|---|--------|--------|-------|------------------|--------|------------|")

        def _sort_key(m: str) -> tuple[float, float]:
            rate = success_cnt[m] / total_cnt[m] if total_cnt[m] else 0.0
            avg = sum(speed[m]) / len(speed[m]) if speed[m] else float("inf")
            return (-rate, avg)

        for rank, model in enumerate(sorted(total_cnt.keys(), key=_sort_key), 1):
            rate = success_cnt[model] / total_cnt[model] * 100 if total_cnt[model] else 0
            avg = sum(speed[model]) / len(speed[model]) if speed[model] else 0.0
            total = sum(speed[model])
            if acc_total[model]:
                acc_pct = acc_correct[model] / acc_total[model] * 100
                acc_str = f"{acc_correct[model]}/{acc_total[model]} ({acc_pct:.0f}%)"
            else:
                acc_str = "—"
            lines.append(
                f"| {rank} | `{model}` | {success_cnt[model]}/{total_cnt[model]} | "
                f"{rate:.0f}% | {acc_str} | {avg:.1f}s | {total:.1f}s |"
            )
        lines.append("")

    if text_results:
        _emit_ranking("Gesamtranking — Text", text_results)
    if vision_results:
        _emit_ranking("Gesamtranking — Vision (Bild-Eingabe)", vision_results)

    # --- Per-group detailed results ---
    lines.append("## Detailergebnisse")
    lines.append("")
    for group_name in groups:
        scenarios = ALL_SCENARIOS.get(group_name, [])
        if not scenarios:
            continue
        lines.append(f"### Gruppe: {group_name.capitalize()}")
        lines.append("")
        for scenario in scenarios:
            results = all_results.get(scenario.name, [])
            if not results:
                continue
            lines.append(f"#### {scenario.name}")
            lines.append("")
            if scenario.description:
                lines.append(f"_{scenario.description}_")
                lines.append("")
            if scenario.vision:
                img_link = _scenario_image_link(scenario, report_dir)
                if img_link:
                    lines.append("**Eingabebild (Vision-Input):**")
                    lines.append("")
                    lines.append(f"![{_md_escape(scenario.name)}]({img_link})")
                    lines.append("")
            lines.append("| Modell | OK | Zeit | Retries | Genauigkeit | Extrahierte Werte |")
            lines.append("|--------|----|------|---------|-------------|-------------------|")
            for r in results:
                ok = "✅" if r.success else "❌"
                if r.success:
                    detail = _md_escape(_value_summary(r.value))
                else:
                    detail = f"FEHLER: {_md_escape(r.error or '—')[:80]}"
                c, t = _field_accuracy(scenario, r)
                acc = f"{c}/{t}" if t else "—"
                lines.append(
                    f"| `{r.model}` | {ok} | {r.duration_s:.1f}s | {r.retries} | "
                    f"{acc} | {detail} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generiert von `benchmarks/run_benchmarks.py`._")
    lines.append("")
    return "\n".join(lines)


def _save_report_md(
    all_results: dict[str, list[BenchmarkResult]],
    scenarios_by_name: dict[str, BenchmarkScenario[Any]],
    groups: list[str],
    models: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report_md(
        all_results, scenarios_by_name, groups, models, output_path.parent
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_benchmarks(
    groups: list[str],
    models: list[str],
    vision_models: list[str],
    save_path: Path | None,
    report_path: Path | None,
) -> None:
    from collections import defaultdict

    all_results: dict[str, list[BenchmarkResult]] = defaultdict(list)
    scenarios_by_name: dict[str, BenchmarkScenario[Any]] = {}
    used_models: list[str] = []

    # --- Build the run plan -------------------------------------------------
    # We iterate model-outer / scenario-inner so that Ollama loads each model
    # once and runs it through all of its scenarios before switching models.
    # This avoids the repeated load/unload churn that happens when the model
    # changes on every request.
    known_groups = [g for g in groups if g in ALL_SCENARIOS]
    for g in groups:
        if g not in ALL_SCENARIOS:
            print(f"  WARNING: unknown group '{g}', skipping.")

    text_scenarios: list[BenchmarkScenario[Any]] = []
    vision_scenarios: list[BenchmarkScenario[Any]] = []
    for group_name in known_groups:
        target = vision_scenarios if group_name in VISION_GROUPS else text_scenarios
        target.extend(ALL_SCENARIOS[group_name])

    for scenario in (*text_scenarios, *vision_scenarios):
        scenarios_by_name[scenario.name] = scenario

    # Each phase pairs a model list with the scenarios those models should run.
    phases: list[tuple[str, list[str], list[BenchmarkScenario[Any]]]] = []
    if text_scenarios:
        phases.append(("Text", models, text_scenarios))
    if vision_scenarios:
        phases.append(("Vision", vision_models, vision_scenarios))

    print(f"\n{'═' * 72}")
    print("  saidex Benchmark Suite")
    if text_scenarios:
        print(f"  Text models:   {', '.join(models)}")
    if vision_scenarios:
        print(f"  Vision models: {', '.join(vision_models)}")
    print(f"  Groups: {', '.join(known_groups)}")
    print("  Order:  model-outer (each model runs all its scenarios before unloading)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 72}\n")

    # --- Execute: one model at a time, all its scenarios in sequence --------
    for phase_label, phase_models, phase_scenarios in phases:
        for model in phase_models:
            if model not in used_models:
                used_models.append(model)
            print(f"\n  {'━' * 68}")
            print(f"  ▶ MODEL: {model}   ({phase_label}, {len(phase_scenarios)} scenarios)")
            print(f"  {'━' * 68}")

            print("    warmup (loading model)…", end=" ", flush=True)
            ok, warm_s, warm_err = await warmup_model(model)
            if ok:
                print(f"ready in {warm_s:.1f}s")
            else:
                print(f"warmup failed after {warm_s:.1f}s: {warm_err}")

            for scenario in phase_scenarios:
                print(f"    {scenario.name:<48}…", end=" ", flush=True)
                result = await run_scenario(scenario, model)
                status = (
                    f"{result.duration_s:.1f}s"
                    if result.success
                    else f"FAILED: {result.error}"
                )
                print(status)
                all_results[scenario.name].append(result)

    # --- Summaries (printed grouped, after all runs) ------------------------
    for group_name in known_groups:
        _print_group_summary(group_name, ALL_SCENARIOS[group_name], all_results)

    _print_speed_ranking(all_results)
    _print_quality_ranking(all_results)

    if save_path:
        _save_results(all_results, save_path)

    if report_path:
        _save_report_md(all_results, scenarios_by_name, groups, used_models, report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run saidex benchmarks against Ollama models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--group",
        dest="groups",
        metavar="GROUP",
        action="append",
        choices=list(ALL_SCENARIOS.keys()),
        help="Scenario group(s) to run. May be repeated. Default: all groups.",
    )
    parser.add_argument(
        "--models",
        metavar="MODEL1,MODEL2",
        help="Comma-separated model names to test. Default: configured MODELS list.",
    )
    parser.add_argument(
        "--vision-models",
        metavar="MODEL1,MODEL2",
        help="Comma-separated vision-capable model names for the 'vision' group. "
        "Default: configured VISION_MODELS list.",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Path to save results as JSON. Directories are created automatically.",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="Path to save a Markdown report. Default: a timestamped file in benchmarks/results/.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable Markdown report generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    groups = args.groups or list(ALL_SCENARIOS.keys())
    models = [m.strip() for m in args.models.split(",")] if args.models else MODELS
    if args.vision_models:
        vision_models = [m.strip() for m in args.vision_models.split(",")]
    elif args.models:
        # An explicit --models override also applies to the vision group.
        vision_models = models
    else:
        vision_models = VISION_MODELS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path: Path | None = None
    if args.save:
        save_path = Path(args.save)
    else:
        save_path = Path(f"benchmarks/results/benchmark_{ts}.json")

    report_path: Path | None = None
    if not args.no_report:
        if args.report:
            report_path = Path(args.report)
        else:
            report_path = Path(f"benchmarks/results/benchmark_{ts}.md")

    try:
        asyncio.run(run_benchmarks(groups, models, vision_models, save_path, report_path))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
