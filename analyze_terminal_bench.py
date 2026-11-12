"""Offline benchmark explorer for LLM-as-a-Verifier (v0.3.0).

只读取本地轨迹和 reward 标签，不调用 LLM，不需要 API Key。

功能：
  1. 按配置选择一个或多个 Benchmark、Best-of-N；
  2. 自动统计 All-pass / Swing / All-fail、Pass@1、Oracle；
  3. 找出轨迹较短的 Swing tasks，辅助低成本功能测试；
  4. 将汇总、任务详情和调试候选列表导出为 CSV/JSON/TXT。

注意：这不是 Verifier 成功率复现，且短轨迹样本不代表整个 Benchmark。
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


# ============================================================================
# 只需修改这里：运行方式不需要任何命令行参数
# ============================================================================

# 支持 "all"，或列表，例如 ["terminal_bench_2.1", "terminal_bench_4.0"]
BENCHMARK_SELECTION: str | list[str] = ["terminal_bench_2.1", "terminal_bench_4.0"]

# 比较 Best-of-3 / Best-of-5。也可设置为 [3]、[5] 或 "auto"（每组用全部可用轨迹）
BEST_OF_VALUES: str | list[int] = [3, 5]

# 为保持与原 run.py 一致，任务轨迹不足 N 条时保留该任务现有的全部轨迹。
# 若设 True，则只保留恰好有至少 N 条轨迹的任务（与官方统计口径可能不同）。
REQUIRE_FULL_N_TRIALS = False

# 控制终端打印多少个最短 Swing task；CSV 始终保存所有 Swing task
TOP_K_SHORTEST_SWING = 10

# 选 "total_chars"（所有选中轨迹之和）或 "shortest_opposite_pair_chars"
# 后者按一个成功轨迹 + 一个失败轨迹的最短组合排序，仅用于调试估算。
SWING_SORT_BY = "total_chars"

# 导出目录：仓库根目录下的 analysis_results/
EXPORT_REPORTS = True
OUTPUT_DIR = "analysis_results"

# ============================================================================
# 主程序：通常不用修改
# ============================================================================

HERE = Path(__file__).resolve().parent
ROOT = next(
    (p for p in (HERE, HERE.parent) if (p / "llm_verifier").is_dir()), None
)
if ROOT is None:
    raise RuntimeError(
        "Place this script in the repository root or scripts/ directory. "
        "Cannot find llm_verifier/."
    )
sys.path.insert(0, str(ROOT))

from llm_verifier.benchmarks import BENCHMARKS  # noqa: E402
from llm_verifier.loaders import LOADERS  # noqa: E402


SUMMARY_COLUMNS = [
    "benchmark", "name", "loader", "best_of", "criteria_count", "criteria",
    "default_K", "default_pivots", "task_count", "trial_count", "incomplete_tasks",
    "all_pass", "swing", "all_fail", "pass_at_1_pct", "oracle_pct",
    "headroom_pct_points", "median_trace_chars", "median_swing_trace_chars",
]
TASK_COLUMNS = [
    "benchmark", "best_of", "task_id", "group", "trials", "successes",
    "failures", "rewards", "trial_names", "trace_chars_each", "total_trace_chars",
    "min_trace_chars", "max_trace_chars", "shortest_opposite_pair_chars",
]


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_chars(n: int | float | None) -> str:
    return f"{n:,.0f}" if n is not None else "n/a"


def selection_names() -> list[str]:
    if BENCHMARK_SELECTION == "all":
        return list(BENCHMARKS)
    if isinstance(BENCHMARK_SELECTION, str):
        names = [BENCHMARK_SELECTION]
    else:
        names = list(dict.fromkeys(BENCHMARK_SELECTION))
    unknown = [name for name in names if name not in BENCHMARKS]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {unknown}; available: {list(BENCHMARKS)}")
    return names


def trial_counts_for(tasks: dict[str, list[dict[str, Any]]]) -> list[int]:
    max_trials = max((len(v) for v in tasks.values()), default=0)
    if BEST_OF_VALUES == "auto":
        return [max_trials] if max_trials else []
    values = list(dict.fromkeys(int(x) for x in BEST_OF_VALUES))
    if any(x <= 0 for x in values):
        raise ValueError("BEST_OF_VALUES must contain positive integers")
    return values


def classify_rewards(rewards: list[int]) -> str:
    if all(r == 1 for r in rewards):
        return "all-pass"
    if all(r == 0 for r in rewards):
        return "all-fail"
    return "swing"


def shortest_opposite_pair_chars(rewards: list[int], lengths: list[int]) -> int | None:
    ok = [length for reward, length in zip(rewards, lengths) if reward == 1]
    bad = [length for reward, length in zip(rewards, lengths) if reward == 0]
    return min(ok) + min(bad) if ok and bad else None


def analyze(tasks: dict[str, list[dict[str, Any]]], cfg: Any, key: str, n: int):
    selected = {}
    for task_id, trials in sorted(tasks.items()):
        picked = trials[:n]
        if not picked:
            continue
        if REQUIRE_FULL_N_TRIALS and len(picked) < n:
            continue
        selected[task_id] = picked
    if not selected:
        return None

    details = []
    for task_id, trials in selected.items():
        raw_rewards = [t.get("reward") for t in trials]
        if any(r not in (0, 1, 0.0, 1.0) for r in raw_rewards):
            raise ValueError(f"{key}/{task_id}: non-binary reward labels: {raw_rewards}")
        rewards = [int(x) for x in raw_rewards]
        lengths = [len(str(t.get("trace") or "")) for t in trials]
        details.append({
            "benchmark": key,
            "best_of": n,
            "task_id": task_id,
            "group": classify_rewards(rewards),
            "trials": len(trials),
            "successes": sum(rewards),
            "failures": len(trials) - sum(rewards),
            "rewards": rewards,
            "trial_names": [str(t.get("trial_name", "")) for t in trials],
            "trace_chars_each": lengths,
            "total_trace_chars": sum(lengths),
            "min_trace_chars": min(lengths),
            "max_trace_chars": max(lengths),
            "shortest_opposite_pair_chars": shortest_opposite_pair_chars(rewards, lengths),
        })

    task_count = len(details)
    all_pass = sum(d["group"] == "all-pass" for d in details)
    swing = sum(d["group"] == "swing" for d in details)
    all_fail = sum(d["group"] == "all-fail" for d in details)
    # Matches the original run.py: each task contributes its mean reward.
    pass1 = sum(d["successes"] / d["trials"] for d in details) / task_count
    oracle = (all_pass + swing) / task_count
    all_lengths = [x for d in details for x in d["trace_chars_each"]]
    swing_lengths = [x for d in details if d["group"] == "swing" for x in d["trace_chars_each"]]
    summary = {
        "benchmark": key,
        "name": cfg.name,
        "loader": cfg.loader,
        "best_of": n,
        "criteria_count": len(cfg.criteria),
        "criteria": list(cfg.criteria),
        "default_K": cfg.n_evaluations,
        "default_pivots": cfg.pivots,
        "task_count": task_count,
        "trial_count": sum(d["trials"] for d in details),
        "incomplete_tasks": sum(d["trials"] < n for d in details),
        "all_pass": all_pass,
        "swing": swing,
        "all_fail": all_fail,
        "pass_at_1_pct": round(pass1 * 100, 5),
        "oracle_pct": round(oracle * 100, 5),
        "headroom_pct_points": round((oracle - pass1) * 100, 5),
        "median_trace_chars": statistics.median(all_lengths),
        "median_swing_trace_chars": (statistics.median(swing_lengths) if swing_lengths else None),
    }
    sort_key = (
        "total_trace_chars" if SWING_SORT_BY == "total_chars"
        else "shortest_opposite_pair_chars"
    )
    if SWING_SORT_BY not in ("total_chars", "shortest_opposite_pair_chars"):
        raise ValueError("SWING_SORT_BY must be 'total_chars' or 'shortest_opposite_pair_chars'")
    swing_rows = sorted(
        (d for d in details if d["group"] == "swing"),
        key=lambda d: (d[sort_key], str(d["task_id"])),
    )
    return summary, details, swing_rows


def make_report_text(summary: dict, swings: list[dict]) -> str:
    s = summary
    lines = [
        f"===== {s['benchmark']} | Best-of-{s['best_of']} =====",
        f"Name: {s['name']}",
        f"Criteria ({s['criteria_count']}): {', '.join(s['criteria'])}",
        f"Config defaults: K={s['default_K']}, pivots={s['default_pivots']}",
        f"Total tasks: {s['task_count']} | Trials: {s['trial_count']}",
        f"All-pass: {s['all_pass']} | Swing: {s['swing']} | All-fail: {s['all_fail']}",
        f"Pass@1: {s['pass_at_1_pct']:.2f}% | Oracle: {s['oracle_pct']:.2f}%",
        f"Oracle - Pass@1: {s['headroom_pct_points']:.2f} percentage points",
        f"Median trace length: {fmt_chars(s['median_trace_chars'])} chars",
        f"Median swing trace length: {fmt_chars(s['median_swing_trace_chars'])} chars",
    ]
    if s["incomplete_tasks"]:
        lines.append(
            f"WARNING: {s['incomplete_tasks']} tasks have fewer than "
            f"{s['best_of']} trials; run.py keeps these partial tasks too."
        )
    lines.extend([
        "",
        f"--- {min(TOP_K_SHORTEST_SWING, len(swings))} shortest Swing tasks (sort={SWING_SORT_BY}) ---",
    ])
    for row in swings[:TOP_K_SHORTEST_SWING]:
        lines.append(
            f"{row['task_id']}: total={row['total_trace_chars']:,} chars; "
            f"rewards={row['rewards']}; lengths={row['trace_chars_each']}; "
            f"shortest opposite-reward pair={fmt_chars(row['shortest_opposite_pair_chars'])} chars"
        )
    if not swings:
        lines.append("No Swing tasks in this selection; verifier selection is not needed.")
    lines.extend([
        "",
        "Notes: character counts are NOT token counts or exact API-cost estimates.",
        "Sorting by short trajectories is for debugging, not representative sampling.",
        "Offline only: no LLM calls; Pass@1 and Oracle are not Verifier results.",
    ])
    return "\n".join(lines)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else val
                for key, val in row.items()
            })


def main() -> None:
    selected_names = selection_names()
    output_dir = ROOT / OUTPUT_DIR
    summaries: list[dict] = []
    errors: list[str] = []
    print(f"Project: {ROOT}")
    print(f"Benchmarks: {selected_names}")
    print(f"Best-of choices: {BEST_OF_VALUES}\n")

    for name in selected_names:
        cfg = BENCHMARKS[name]
        print(f"Loading {name} ...")
        try:
            tasks, _n_runs = LOADERS[cfg.loader](cfg.data, str(ROOT))
        except UnicodeDecodeError as exc:
            msg = (
                f"{name}: text encoding failed: {exc}. "
                "On Windows use explicit encoding='utf-8' when reading JSON in loaders.py."
            )
            errors.append(msg)
            print(f"ERROR: {msg}\n")
            continue
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            msg = f"{name}: cannot load local data: {type(exc).__name__}: {exc}"
            errors.append(msg)
            print(f"ERROR: {msg}\n")
            continue
        if "" in tasks:
            msg = (
                f"{name}: empty task ID detected. On Windows, check load_terminal() "
                "in loaders.py: use os.path.basename(os.path.normpath(task_dir)) "
                "instead of os.path.basename(task_dir.rstrip('/'))."
            )
            errors.append(msg)
            print(f"ERROR: {msg}\n")
            continue
        if not tasks:
            msg = f"{name}: loader returned zero tasks"
            errors.append(msg)
            print(f"WARNING: {msg}\n")
            continue
        for n in trial_counts_for(tasks):
            max_trials = max(len(v) for v in tasks.values())
            if n > max_trials:
                print(f"Skipping {name} Best-of-{n}: only {max_trials} trials available.\n")
                continue
            result = analyze(tasks, cfg, name, n)
            if result is None:
                print(f"Skipping {name} Best-of-{n}: no qualifying tasks.\n")
                continue
            summary, details, swings = result
            report_text = make_report_text(summary, swings)
            print(report_text + "\n")
            summaries.append(summary)
            if EXPORT_REPORTS:
                output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{name}_bo{n}"
                write_csv(output_dir / f"{stem}_tasks.csv", TASK_COLUMNS, details)
                write_csv(output_dir / f"{stem}_swing.csv", TASK_COLUMNS, swings)
                (output_dir / f"{stem}_summary.txt").write_text(report_text + "\n", encoding="utf-8")
                (output_dir / f"{stem}_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

    print("===== Benchmark Comparison =====")
    if summaries:
        print(f"{'Benchmark':<23s} {'BoN':>4s} {'Tasks':>6s} {'Swing':>6s} {'Pass@1':>9s} {'Oracle':>9s} {'Median chars':>13s}")
        for s in summaries:
            print(
                f"{s['benchmark']:<23s} {s['best_of']:>4d} {s['task_count']:>6d} "
                f"{s['swing']:>6d} {s['pass_at_1_pct']:>8.2f}% "
                f"{s['oracle_pct']:>8.2f}% {fmt_chars(s['median_trace_chars']):>13s}"
            )
        if EXPORT_REPORTS:
            write_csv(output_dir / "benchmark_comparison.csv", SUMMARY_COLUMNS, summaries)
            print(f"\nReports saved in: {output_dir}")
    else:
        print("No summaries generated. Check local data and configuration.")
    if errors:
        print("\nWarnings/errors:")
        for msg in errors:
            print(f"- {msg}")
    print("\nOffline analysis completed. No API was called.")


if __name__ == "__main__":
    main()