"""Zero-API controlled ablation: ring-only vs full PPT (Terminal-Bench 2.1, Bo3).

Place this file at the root of the *existing* llm-as-a-verifier repository.
Run inside the ftec5660-verifier environment:
    python .\run_ring_only_ablation.py

Data are READ ONLY:
  analysis_results/small_benchmark/cc129365bcc729e9/scores.json
  analysis_results/small_benchmark/cc129365bcc729e9/report_K2.json

Outputs are written ONLY to:
  analysis_results/small_benchmark/cc129365bcc729e9/ablation_ring_only/

No API client is constructed, no API function is imported, and no network
requests are made. The official pivot_tournament functions are reused;
only the second-stage accumulation is omitted in the ring-only branch.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

# ===== Explicit, local-only settings =====
EXPERIMENT_DIR = Path("analysis_results/small_benchmark/cc129365bcc729e9")
EXPECT_K = 2
EXPECT_SEED = 0
EXPECT_N_TRIALS = 3
EXPECT_TASK_COUNT = 12
EXPECT_CRITERIA = ("specification", "output_match", "error_signals")
# ========================================

ROOT = Path(__file__).resolve().parent
if not (ROOT / "llm_verifier" / "pivot_tournament.py").is_file():
    raise RuntimeError("Put this script in the repository ROOT (alongside llm_verifier/).")

# Import only the original project's pure arithmetic/selection module.
from llm_verifier import pivot_tournament as ppt  # noqa: E402


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required existing experiment file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON dictionary: {path}")
    return data


def choose(w: list[float], counts: list[int]) -> int:
    """Same tie-break rule as original scripts/run.py."""
    return max(range(len(w)), key=lambda i: (w[i] / counts[i] if counts[i] else 0.0, -i))


def mean_preferences(w: list[float], counts: list[int]) -> list[float]:
    return [w[i] / counts[i] if counts[i] else 0.0 for i in range(len(w))]


def run() -> None:
    experiment_dir = ROOT / EXPERIMENT_DIR
    score_file = experiment_dir / "scores.json"
    report_file = experiment_dir / "report_K2.json"
    scores = load_json(score_file)
    report = load_json(report_file)

    # Guard against accidentally comparing a mismatched dataset/cache.
    if report.get("status") != "complete":
        raise ValueError("Baseline report status must be 'complete'.")
    if report.get("K") != EXPECT_K or report.get("ring_seed") != EXPECT_SEED:
        raise ValueError("Expected the completed K=2, seed=0 baseline.")
    if report.get("n_trials") != EXPECT_N_TRIALS:
        raise ValueError("Expected Best-of-3 (n_trials=3).")
    if tuple(report.get("criteria", [])) != EXPECT_CRITERIA:
        raise ValueError("The baseline criteria differ; do not mix caches.")
    if report.get("pivots") != 1 or report.get("selection_mode") != "swing_spread":
        raise ValueError("Expected original 1-pivot, swing_spread baseline.")
    tasks = sorted(report.get("selected_tasks", []))
    if len(tasks) != EXPECT_TASK_COUNT or set(tasks) != {d["task"] for d in report["task_decisions"]}:
        raise ValueError("Expected exactly 12 selected tasks with matching decisions.")
    if report.get("n_swing") != EXPECT_TASK_COUNT:
        raise ValueError("This specific ablation requires 12 swing tasks.")
    baseline = {d["task"]: d for d in report["task_decisions"]}
    required_ring: set[str] = set()
    required_full: set[str] = set()
    rng = random.Random(EXPECT_SEED)
    rows = []

    for task in tasks:
        truth = [int(v) for v in baseline[task]["trial_rewards"]]
        n = len(truth)
        if n != EXPECT_N_TRIALS or not (0 in truth and 1 in truth):
            raise ValueError(f"Unexpected truth labels for {task}: {truth}")

        ring_pairs = ppt.ring_cycle(n, rng)

        def score_pair(a: int, b: int, key_set: set[str]) -> tuple[float, float]:
            # Strictly require every score; NEVER use an invented 0.5 tie.
            if a == b:
                return 0.5, 0.5
            A, B = 0.0, 0.0
            for cid in EXPECT_CRITERIA:
                for rep in range(EXPECT_K):
                    key = f"{cid}|{task}|{a},{b}|{rep}"
                    if key not in scores:
                        raise RuntimeError(f"Missing cached score! NO API fallback: {key}")
                    entry = scores[key]
                    if not isinstance(entry, dict) or not all(k in entry for k in ("score_A", "score_B")):
                        raise ValueError(f"Malformed cached score: {key}")
                    A += float(entry["score_A"])
                    B += float(entry["score_B"])
                    key_set.add(key)
            scale = len(EXPECT_CRITERIA) * EXPECT_K
            return A / scale, B / scale

        # COMMON Phase A, identical candidate ring and cached model scores.
        ring_w = [0.0] * n
        ring_counts = [0] * n
        ppt.accumulate(ring_pairs,
                       lambda a, b: score_pair(a, b, required_ring),
                       ring_w, ring_counts)
        ring_best = choose(ring_w, ring_counts)
        pivot = ppt.select_pivots(ring_w, ring_counts, 1)

        # CONTROL: original full PPT, add Phase B pivot comparisons.
        phase_b = ppt.pivot_round_pairs(n, pivot)
        full_w = ring_w.copy()
        full_counts = ring_counts.copy()
        ppt.accumulate(phase_b,
                       lambda a, b: score_pair(a, b, required_full),
                       full_w, full_counts)
        full_best = choose(full_w, full_counts)
        required_full.update(k for k in required_ring if f"|{task}|" in k)

        expected = baseline[task]
        if full_best + 1 != expected["chosen_trial_1based"]:
            raise AssertionError(f"Full PPT winner disagrees with official report: {task}")
        if ",".join(str(p + 1) for p in pivot) != str(expected["pivot_indices_1based"]):
            raise AssertionError(f"Pivot disagrees with official report: {task}")
        if truth[full_best] != expected["chosen_reward"]:
            raise AssertionError(f"Ground-truth reward disagrees with official report: {task}")

        rows.append({
            "task": task,
            "truth_trial_rewards": "".join(map(str, truth)),
            "ring_only_chosen_trial_1based": ring_best + 1,
            "ring_only_reward": truth[ring_best],
            "full_ppt_chosen_trial_1based": full_best + 1,
            "full_ppt_reward": truth[full_best],
            "same_winner": int(ring_best == full_best),
            "pivot_trial_1based": ",".join(str(p + 1) for p in pivot),
            "ring_logical_comparisons": len(ring_pairs),
            "full_logical_comparisons": len(ring_pairs) + len(phase_b),
            "ring_mean_preference": json.dumps([round(x, 6) for x in mean_preferences(ring_w, ring_counts)]),
            "full_mean_preference": json.dumps([round(x, 6) for x in mean_preferences(full_w, full_counts)]),
        })

    ring_success = sum(r["ring_only_reward"] for r in rows)
    full_success = sum(r["full_ppt_reward"] for r in rows)
    same = sum(r["same_winner"] for r in rows)

    # The union also handles phase-B comparisons already paid in Phase A.
    required_full.update(required_ring)
    # Validate exact original record count and independent reproduction.
    if len(required_ring) != 216 or len(required_full) != 288:
        raise AssertionError(f"Unexpected scored-key counts: ring={len(required_ring)}, full={len(required_full)}")
    if full_success != report["swing_correct"]:
        raise AssertionError("Full PPT accuracy disagrees with original baseline.")

    summary = {
        "experiment": "Controlled offline ablation: remove Phase B pivot round",
        "baseline_report": str(report_file),
        "baseline_score_cache": str(score_file),
        "git_head_from_baseline": report.get("git_head"),
        "N": EXPECT_N_TRIALS,
        "C": len(EXPECT_CRITERIA),
        "K": EXPECT_K,
        "ring_seed": EXPECT_SEED,
        "selection_seed": report.get("selection_seed"),
        "task_scope": "12 length-stratified Swing Tasks (not full 89-task benchmark)",
        "new_api_calls": 0,
        "ring_only": {
            "successes": ring_success,
            "tasks": len(rows),
            "accuracy": ring_success / len(rows),
            "unique_score_records_needed": len(required_ring),
            "logical_comparisons": sum(r["ring_logical_comparisons"] for r in rows),
        },
        "full_ppt": {
            "successes": full_success,
            "tasks": len(rows),
            "accuracy": full_success / len(rows),
            "unique_score_records_needed": len(required_full),
            "logical_comparisons": sum(r["full_logical_comparisons"] for r in rows),
        },
        "same_winner_tasks": same,
        "scoring_call_reduction_fraction": 1.0 - len(required_ring) / len(required_full),
        "limitation": "Post-hoc fixed-data cache replay, one seed, selected swing-only tasks; not general equivalence or a new live-model run.",
        "decisions": rows,
    }

    # Outputs are isolated. NEVER touch scores.json/report_K2.json.
    out = experiment_dir / "ablation_ring_only"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ring_only_vs_full_task_decisions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out / "ring_only_vs_full_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== OFFLINE ABLATION COMPLETE — ZERO API REQUESTS ===")
    print(f"Full PPT baseline: {full_success}/{len(rows)} ({full_success / len(rows):.2%})")
    print(f"Ring-only variant: {ring_success}/{len(rows)} ({ring_success / len(rows):.2%})")
    print(f"Same winning trial on {same}/{len(rows)} tasks")
    print(f"Scoring records: full {len(required_full)}; ring-only {len(required_ring)}")
    print(f"Reduced records: {len(required_full) - len(required_ring)} ({summary['scoring_call_reduction_fraction']:.0%})")
    print(f"Logical comparisons: full {summary['full_ppt']['logical_comparisons']}; ring-only {summary['ring_only']['logical_comparisons']}")
    print("Results:", csv_path)
    print("Results:", json_path)
    print("Important: this is a post-hoc cache-based ablation on 12 swing tasks, not a new API run.")


if __name__ == "__main__":
    run()
