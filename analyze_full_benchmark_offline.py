"""Offline K=1/K=2/K=4 analysis for Terminal-Bench 2.1 Best-of-3.

Reads the ORIGINAL full-89 K=2 report and shared scores.json; optionally
reads the completed full-89 K=4 report. It makes ZERO API calls and never
modifies either input file.

Place in the llm-as-a-verifier repository root and run:
    python analyze_full_benchmark_offline.py

Optional:
    python analyze_full_benchmark_offline.py --report-k2 PATH --report-k4 PATH \
        --scores PATH --out PATH

- Replays ring and full-PPT preference aggregation from persisted scores.
- Reconstructs the pivot independently for each K (never borrows K=2 pivot).
- Checks the reconstructed Full-PPT winners against the official saved report.
- Allows Ring-only K=1 from rep=0 (also rep=1 sensitivity check).
- If a Full-PPT K=1 pivot requires uncached directed comparisons, marks it
  INCOMPLETE instead of filling missing entries with artificial ties.
- Preserves the legacy K=2 offline outputs and writes new cross-K comparisons.
- All scoring request counts are algorithmic records, NOT actual API prices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_EXPERIMENT = Path("analysis_results/small_benchmark/60e63e37b82a1550")
DEFAULT_CRITERIA = ("specification", "output_match", "error_signals")


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def csv_write(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    if not fields:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ring_cycle(n, rng):
    perm = list(range(n))
    rng.shuffle(perm)
    return [(perm[i], perm[(i + 1) % n]) for i in range(n)]


def pivot_round_pairs(n, pivot):
    return [(i, pivot) for i in range(n) if i != pivot]


def winner(means):
    return max(range(len(means)), key=lambda i: (means[i], -i))


def margin(means):
    ordered = sorted(means, reverse=True)
    return ordered[0] - ordered[1]


def evaluate_pairs(scores, task, pairs, criteria, reps, n=3):
    """Exact original arithmetic: average rewards, sigmoid, accumulate wins.

    Missing entries are *never* replaced with 0.5: this routine reports the
    missing cache keys, and a selection is not computed for incomplete data.
    """
    wins = [0.0] * n
    counts = [0] * n
    needed, missing = set(), set()
    for a, b in pairs:
        values = []
        for criterion in criteria:
            for rep in reps:
                key = f"{criterion}|{task}|{a},{b}|{rep}"
                needed.add(key)
                if key not in scores:
                    missing.add(key)
                else:
                    record = scores[key]
                    if not isinstance(record, dict) or not all(
                        isinstance(record.get(z), (int, float)) and
                        math.isfinite(float(record[z]))
                        for z in ("score_A", "score_B")
                    ):
                        raise ValueError(f"Invalid scored record: {key}")
                    values.append((float(record["score_A"]), float(record["score_B"])))
        if len(values) != len(criteria) * len(reps):
            continue
        ra = sum(v[0] for v in values) / len(values)
        rb = sum(v[1] for v in values) / len(values)
        # Stable logistic for large differences (actual scores are in [0,1]).
        diff = ra - rb
        p = (1 / (1 + math.exp(-diff))) if diff >= 0 else (
            math.exp(diff) / (1 + math.exp(diff)))
        wins[a] += p
        wins[b] += 1.0 - p
        counts[a] += 1
        counts[b] += 1
    if missing:
        return None, needed, missing
    means = [wins[i] / counts[i] if counts[i] else 0.0 for i in range(n)]
    return means, needed, set()


def replay_task(scores, name, ring, criteria, reps, method):
    n = len(ring)
    ring_means, ring_keys, ring_missing = evaluate_pairs(
        scores, name, ring, criteria, reps, n)
    if ring_missing:
        return dict(complete=False, ring_missing=ring_missing,
                    missing=ring_missing, keys=ring_keys, winner=None,
                    pivot=None, means=None, ring_means=None)
    pivot = winner(ring_means)
    if method == "ring":
        return dict(complete=True, ring_missing=set(), missing=set(),
                    keys=ring_keys, winner=pivot, pivot=pivot,
                    means=ring_means, ring_means=ring_means)
    full_pairs = ring + pivot_round_pairs(n, pivot)
    full_means, full_keys, missing = evaluate_pairs(
        scores, name, full_pairs, criteria, reps, n)
    return dict(complete=not missing, ring_missing=set(), missing=missing,
                keys=full_keys, winner=winner(full_means) if full_means else None,
                pivot=pivot, means=full_means, ring_means=ring_means)


def get_variants(include_k4):
    c = DEFAULT_CRITERIA
    # K=1 rep=0 is the true K=1 subset. rep=1 is a swapped-slot sensitivity
    # analysis, NOT an independent K=1 API run.
    v = [
        ("full_K2", "full", c, (0, 1), "official_reproduction"),
        ("ring_K2_all3", "ring", c, (0, 1), "controlled_ablation"),
        ("ring_K1_rep0_all3", "ring", c, (0,), "posthoc_K1_first_repeat"),
        ("ring_K1_rep1_all3", "ring", c, (1,), "posthoc_second_only_sensitivity"),
        ("full_K1_rep0_all3", "full", c, (0,), "coverage_dependent"),
        ("full_K1_rep1_all3", "full", c, (1,), "coverage_dependent_sensitivity"),
        ("ring_K2_specification", "ring", (c[0],), (0, 1), "exploratory_criterion"),
        ("ring_K2_output_match", "ring", (c[1],), (0, 1), "exploratory_criterion"),
        ("ring_K2_error_signals", "ring", (c[2],), (0, 1), "exploratory_criterion"),
        ("ring_K2_spec_output", "ring", c[:2], (0, 1), "exploratory_criterion"),
        ("ring_K2_output_error", "ring", c[1:], (0, 1), "exploratory_criterion"),
    ]
    if include_k4:
        v.extend([
            ("full_K4", "full", c, (0, 1, 2, 3), "official_reproduction"),
            ("ring_K4_all3", "ring", c, (0, 1, 2, 3), "controlled_ablation"),
            ("ring_K4_specification", "ring", (c[0],), (0, 1, 2, 3), "exploratory_criterion"),
            ("ring_K4_output_match", "ring", (c[1],), (0, 1, 2, 3), "exploratory_criterion"),
            ("ring_K4_error_signals", "ring", (c[2],), (0, 1, 2, 3), "exploratory_criterion"),
        ])
    return v


def validate_report(report, k):
    if report.get("status") != "complete":
        raise ValueError(f"report_K{k} is not a completed experiment")
    if report.get("K") != k or report.get("selection_mode") != "full":
        raise ValueError(f"report_K{k} must describe FULL 89-task K={k}, not a pilot")
    if report.get("n_tasks") != 89 or report.get("n_swing") != 24:
        raise ValueError(f"report_K{k} has unexpected task counts")
    if report.get("n_trials") != 3 or report.get("pivots") != 1:
        raise ValueError(f"report_K{k} must use N=3 and pivots=1")
    if tuple(report.get("criteria", ())) != DEFAULT_CRITERIA:
        raise ValueError(f"report_K{k} has different criteria")
    if len(report.get("task_decisions", [])) != 24:
        raise ValueError(f"report_K{k} has missing swing task decisions")
    decisions = {r["task"]: r for r in report["task_decisions"]}
    if len(decisions) != 24:
        raise ValueError(f"report_K{k} has duplicated task decisions")
    return decisions


def compare_configs(report2, report4):
    must_equal = (
        "selected_tasks", "selected_task_total_chars", "subset_categories",
        "model", "reasoning_effort", "max_output_tokens", "n_trials",
        "criteria", "pivots", "ring_seed", "max_workers", "n_tasks",
        "n_swing", "pass1", "oracle",
    )
    mismatches = [k for k in must_equal if report2.get(k) != report4.get(k)]
    if mismatches:
        raise ValueError("K2/K4 experiment conditions differ: " + ", ".join(mismatches))
    if report2.get("git_head") != report4.get("git_head"):
        print("WARNING: K2 and K4 git_head differ; verify scoring code changes", file=sys.stderr)


def run(report_k2, report_k4, score_path, output_dir):
    report2 = read_json(report_k2)
    decisions2 = validate_report(report2, 2)
    report4 = None
    decisions4 = None
    if report_k4.is_file():
        report4 = read_json(report_k4)
        decisions4 = validate_report(report4, 4)
        compare_configs(report2, report4)
        if set(decisions2) != set(decisions4):
            raise ValueError("K2 and K4 swing task sets differ")
        for task, old in decisions2.items():
            if old["trial_rewards"] != decisions4[task]["trial_rewards"]:
                raise ValueError(f"K2/K4 ground-truth rewards differ for {task}")
        print("Full-89 K4 report detected and configuration checked.")
    else:
        print("Full-89 K4 report not found; running K1/K2 only.")
        print("After K4 completes, run this script again. NO API calls here.")
    scores = read_json(score_path)
    if not isinstance(scores, dict) or not scores:
        raise ValueError("scores.json is missing or invalid")
    names = sorted(decisions2)
    rings, rng = {}, random.Random(report2["ring_seed"])
    for name in names:
        rings[name] = ring_cycle(3, rng)

    trials_truth = {}
    for name in names:
        t = decisions2[name]["trial_rewards"]
        if len(t) != 3 or set(t) - {"0", "1"} or set(t) != {"0", "1"}:
            raise ValueError(f"Expected 3 binary mixed trial_rewards for {name}, got {t}")
        trials_truth[name] = [int(x) for x in t]

    variants = get_variants(include_k4=(report4 is not None))
    result = {}
    records = []
    all_pass = int(report2["subset_categories"]["all_pass"])
    n_tasks = int(report2["n_tasks"])
    all_checks = {}
    for label, method, criteria, reps, interpretation in variants:
        task_outputs, used, missing = {}, set(), set()
        wins, evaluated = 0, 0
        for task in names:
            res = replay_task(scores, task, rings[task], criteria, reps, method)
            task_outputs[task] = res
            used.update(res["keys"])
            missing.update(res["missing"])
            if res["complete"]:
                evaluated += 1
                wins += trials_truth[task][res["winner"]]
        status = "complete" if evaluated == len(names) else "incomplete_cache"
        aggregate = None if status != "complete" else (all_pass + wins)
        row = dict(
            variant=label,
            status=status,
            method=method,
            criteria=",".join(criteria),
            reps=",".join(map(str, reps)),
            tasks_evaluated=evaluated,
            total_swing_tasks=len(names),
            correct_swing_24=wins if status == "complete" else "",
            overall_correct_89=aggregate if aggregate is not None else "",
            overall_accuracy_pct=(round(aggregate / n_tasks * 100, 3)
                                  if aggregate is not None else ""),
            score_requests_needed=(len(used) if status == "complete" else ""),
            missing_score_keys=len(missing),
            scenario=interpretation,
            limitation=("All replay variants use cached scores, not new API calls."
                        if method == "ring" or "K1" in label else
                        "Replay verified against official report."),
        )
        records.append(row)
        result[label] = dict(summary=row, tasks=task_outputs,
                             score_keys=used, missing=missing)

    for k, official in [(2, decisions2), (4, decisions4)]:
        if official is None:
            continue
        label = f"full_K{k}"
        d = result[label]
        if d["summary"]["status"] != "complete":
            sample = sorted(d["missing"])[:4]
            raise ValueError(f"Official full K={k} cache is incomplete: " + repr(sample))
        for task in names:
            res = d["tasks"][task]
            row = official[task]
            if res["winner"] + 1 != int(row["chosen_trial_1based"]):
                raise AssertionError(f"K={k} winner mismatch for {task}: "
                                     f"replay={res['winner']+1}, official={row['chosen_trial_1based']}")
            if res["pivot"] + 1 != int(row["pivot_indices_1based"]):
                raise AssertionError(f"K={k} pivot mismatch for {task}")
            if trials_truth[task][res["winner"]] != int(row["chosen_reward"]):
                raise AssertionError(f"K={k} ground truth mismatch for {task}")
        correct = sum(trials_truth[t][d["tasks"][t]["winner"]] for t in names)
        if correct != int((report2 if k == 2 else report4)["swing_correct"]):
            raise AssertionError(f"K={k} replay swing accuracy mismatch")
        expected = (report2 if k == 2 else report4)["algorithmic_verification_requests"]
        if len(d["score_keys"]) != expected:
            raise AssertionError(f"K={k} score-key count differs from report")
        all_checks[f"full_K{k}_matches_official_24"] = True

    # Cross-variant winner changes (not just change in aggregate accuracy).
    full2 = result["full_K2"]["tasks"]
    ring2 = result["ring_K2_all3"]["tasks"]
    full4 = result["full_K4"]["tasks"] if report4 else None
    ring4 = result["ring_K4_all3"]["tasks"] if report4 else None
    records_by_name = {r["variant"]: r for r in records}
    for label, d in result.items():
        row = records_by_name[label]
        if row["status"] == "complete":
            row["different_winning_trials_vs_full_K2"] = sum(
                full2[t]["winner"] != d["tasks"][t]["winner"] for t in names)
            row["different_winning_trials_vs_K2_ring"] = sum(
                ring2[t]["winner"] != d["tasks"][t]["winner"] for t in names)
            row["reduction_vs_full_K2_pct"] = round(
                (1.0 - row["score_requests_needed"] / len(result["full_K2"]["score_keys"])) * 100, 2)
        else:
            row["different_winning_trials_vs_full_K2"] = ""
            row["different_winning_trials_vs_K2_ring"] = ""
            row["reduction_vs_full_K2_pct"] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_write(output_dir / "K1_K2_K4_variant_comparison.csv", records)

    # Preserve the previously published K2 breakdown shape for report/plots.
    details, failures, cross_k_rows, k2k4_changes, followups, coverage = [], [], [], [], [], []
    for task in names:
        truth = trials_truth[task]
        f2, r2 = full2[task], ring2[task]
        k1r0 = result["ring_K1_rep0_all3"]["tasks"][task]
        k1r1 = result["ring_K1_rep1_all3"]["tasks"][task]
        om2 = result["ring_K2_output_match"]["tasks"][task]
        detail = dict(
            task=task,
            total_trajectory_chars=report2["selected_task_total_chars"][task],
            trial_rewards="".join(map(str, truth)),
            n_successful_trials=sum(truth),
            ring_K2_chosen_trial_1based=r2["winner"] + 1,
            ring_K2_reward=truth[r2["winner"]],
            full_K2_chosen_trial_1based=f2["winner"] + 1,
            full_K2_reward=truth[f2["winner"]],
            ring_full_same_trial=int(r2["winner"] == f2["winner"]),
            K1_rep0_ring_trial_1based=k1r0["winner"] + 1,
            K1_rep0_ring_reward=truth[k1r0["winner"]],
            K1_rep1_ring_trial_1based=k1r1["winner"] + 1,
            K1_rep1_ring_reward=truth[k1r1["winner"]],
            output_match_only_K2_ring_trial_1based=om2["winner"] + 1,
            output_match_only_K2_ring_reward=truth[om2["winner"]],
            ring_margin=round(margin(r2["means"]), 8),
            full_margin=round(margin(f2["means"]), 8),
            ring_preferences=json.dumps([round(x, 6) for x in r2["means"]]),
            full_preferences=json.dumps([round(x, 6) for x in f2["means"]]),
        )
        details.append(detail)
        if not truth[f2["winner"]]:
            failures.append(detail)

        row = dict(task=task, trial_rewards="".join(map(str, truth)),
                   full_K2_trial=f2["winner"] + 1,
                   full_K2_correct=truth[f2["winner"]],
                   ring_K2_trial=r2["winner"] + 1,
                   ring_K2_correct=truth[r2["winner"]],
                   K1_rep0_ring_trial=k1r0["winner"] + 1,
                   K1_rep0_ring_correct=truth[k1r0["winner"]],
                   K1_rep1_ring_trial=k1r1["winner"] + 1,
                   K1_rep1_ring_correct=truth[k1r1["winner"]],
                   K1_full_rep0_trial=(result["full_K1_rep0_all3"]["tasks"][task]["winner"] + 1
                                       if result["full_K1_rep0_all3"]["tasks"][task]["complete"] else ""),
                   K1_full_rep0_missing_keys=len(result["full_K1_rep0_all3"]["tasks"][task]["missing"]),
                   K1_full_rep1_trial=(result["full_K1_rep1_all3"]["tasks"][task]["winner"] + 1
                                       if result["full_K1_rep1_all3"]["tasks"][task]["complete"] else ""),
                   K1_full_rep1_missing_keys=len(result["full_K1_rep1_all3"]["tasks"][task]["missing"]),
                   full_K4_trial=f"{full4[task]['winner']+1}" if full4 else "",
                   full_K4_correct=truth[full4[task]["winner"]] if full4 else "",
                   ring_K4_trial=f"{ring4[task]['winner']+1}" if ring4 else "",
                   ring_K4_correct=truth[ring4[task]["winner"]] if ring4 else "",
                   full_K2_K4_same_trial=(int(f2["winner"] == full4[task]["winner"]) if full4 else ""),
                   K2_K4_outcome=(
                       "corrected" if not truth[f2["winner"]] and truth[full4[task]["winner"]]
                       else "regressed" if truth[f2["winner"]] and not truth[full4[task]["winner"]]
                       else "both_correct" if truth[f2["winner"]] else "both_wrong"
                       ) if full4 else "",
                   K2_ring_margin=round(margin(r2["means"]), 8),
                   K2_full_margin=round(margin(f2["means"]), 8),
                   K4_ring_margin=round(margin(ring4[task]["means"]), 8) if ring4 else "",
                   K4_full_margin=round(margin(full4[task]["means"]), 8) if full4 else "",
                   K2_pivot_trial=f2["pivot"] + 1,
                   K4_pivot_trial=full4[task]["pivot"] + 1 if full4 else "")
        cross_k_rows.append(row)
        if full4 and f2["winner"] != full4[task]["winner"]:
            k2k4_changes.append(row)
        if not truth[f2["winner"]]:
            followups.append(row)
        for k1label in ("full_K1_rep0_all3", "full_K1_rep1_all3"):
            k1res = result[k1label]["tasks"][task]
            coverage.append({
                "task":task,"variant":k1label,
                "status":"complete" if k1res["complete"] else "missing_cache",
                "K1_pivot_trial":(k1res["pivot"] + 1 if k1res["pivot"] is not None else ""),
                "K2_pivot_trial":f2["pivot"] + 1,
                "needed_but_missing_score_keys":len(k1res["missing"]),
                "missing_score_keys":",".join(sorted(k1res["missing"])),
            })

    csv_write(output_dir / "K1_K2_K4_task_decisions.csv", cross_k_rows)
    csv_write(output_dir / "K2_vs_K4_changed_winners.csv", k2k4_changes,
              list(cross_k_rows[0]))
    csv_write(output_dir / "K2_failures_followup_K4.csv", followups,
              list(cross_k_rows[0]))
    csv_write(output_dir / "K1_full_ppt_cache_coverage.csv", coverage)
    csv_write(output_dir / "tasks_24_breakdown.csv", details)
    csv_write(output_dir / "failures_5.csv", failures, list(details[0]))

    # Legacy K2 ablations: same eight variants, same columns, same order.
    legacy_ablation = []
    for v in ("ring_K2_all3", "ring_K1_rep0_all3", "ring_K1_rep1_all3",
              "ring_K2_specification", "ring_K2_output_match", "ring_K2_error_signals",
              "ring_K2_spec_output", "ring_K2_output_error"):
        r = records_by_name[v]
        legacy_ablation.append(dict(
            variant=v, criteria=r["criteria"], reps=r["reps"],
            correct_swing_24=r["correct_swing_24"],
            overall_correct_89=r["overall_correct_89"],
            overall_accuracy_pct=r["overall_accuracy_pct"],
            score_requests_needed=r["score_requests_needed"],
            reduction_vs_full_pct=r["reduction_vs_full_K2_pct"],
            different_winning_trials_vs_K2_ring=r["different_winning_trials_vs_K2_ring"],
            warning="Post-hoc cached-score analysis, not independent fresh-model experiment"
        ))
    csv_write(output_dir / "offline_ablation_variants.csv", legacy_ablation)

    criterion_rows = []
    for k in (2, 4):
        if k == 4 and not report4:
            continue
        for criterion in DEFAULT_CRITERIA:
            counts = Counter()
            for key in result[f"full_K{k}"]["score_keys"]:
                crit, task, pair, rep = key.split("|")
                if crit != criterion:
                    continue
                a, b = map(int, pair.split(","))
                truth = trials_truth[task]
                if truth[a] == truth[b]:
                    continue
                val = scores[key]
                signed = (val["score_A"] - val["score_B"]) * (
                    1 if truth[a] > truth[b] else -1)
                group = ("correct_direction" if signed > 0.005 else
                         "incorrect_direction" if signed < -0.005 else "near_tie")
                counts[group] += 1
            criterion_rows.append(dict(
                K=k, criterion=criterion,
                correct_direction=counts["correct_direction"],
                incorrect_direction=counts["incorrect_direction"],
                near_tie=counts["near_tie"],
                total_cross_label_records=sum(counts.values()),
            ))
    csv_write(output_dir / "criterion_score_directions_by_K.csv", criterion_rows)
    legacy_criteria = [{k: v for k, v in r.items() if k != "K"}
                       for r in criterion_rows if r["K"] == 2]
    csv_write(output_dir / "criterion_score_directions.csv", legacy_criteria)

    full2_winners = {t: full2[t]["winner"] for t in names}
    ring2_winners = {t: ring2[t]["winner"] for t in names}
    changed = (len(k2k4_changes) if full4 else None)
    fixed = (sum(r["K2_K4_outcome"] == "corrected" for r in cross_k_rows)
             if full4 else None)
    regressed = (sum(r["K2_K4_outcome"] == "regressed" for r in cross_k_rows)
                 if full4 else None)
    def mini(label):
        r = records_by_name[label]
        return dict(status=r["status"], correct_swing_24=r["correct_swing_24"],
                    overall_correct_89=r["overall_correct_89"],
                    overall_accuracy_pct=r["overall_accuracy_pct"],
                    score_requests_needed=r["score_requests_needed"],
                    missing_score_keys=r["missing_score_keys"])
    summary = {
        "status":"offline_completed_and_validated" if report4 else "K2_complete_K4_pending",
        "source_git_head":report2.get("git_head"),
        "source_report":str(report_k2),
        "source_report_K4":str(report_k4) if report4 else None,
        "source_cache":str(score_path),
        "new_api_requests":0,
        "cached_score_entries_currently_present":len(scores),
        "K1_ring_rep0":mini("ring_K1_rep0_all3"),
        "K1_ring_rep1_sensitivity":mini("ring_K1_rep1_all3"),
        "K1_full_rep0":mini("full_K1_rep0_all3"),
        "K1_full_rep1_sensitivity":mini("full_K1_rep1_all3"),
        "original_full_ppt":{
            "overall_success_89":records_by_name["full_K2"]["overall_correct_89"],
            "swing_success_24":records_by_name["full_K2"]["correct_swing_24"],
            "requests":len(result["full_K2"]["score_keys"])},
        "ring_only_K2":{
            "overall_success_89":all_pass + sum(trials_truth[t][ring2_winners[t]] for t in names),
            "swing_success_24":sum(trials_truth[t][ring2_winners[t]] for t in names),
            "requests":len(result["ring_K2_all3"]["score_keys"]),
        },
        "identical_full_ring_winners":sum(full2_winners[t] == ring2_winners[t] for t in names),
        "phase_B_increased_top2_margin":sum(
            margin(full2[t]["means"]) > margin(ring2[t]["means"]) + 1e-12
            for t in names),
        "phase_B_decreased_top2_margin":sum(
            margin(full2[t]["means"]) < margin(ring2[t]["means"]) - 1e-12
            for t in names),
        "failures":[r["task"] for r in failures],
        "K4_full":mini("full_K4") if report4 else {"status":"pending_report"},
        "K4_ring":mini("ring_K4_all3") if report4 else {"status":"pending_report"},
        "K2_K4_comparison":{
            "status":"complete" if full4 else "pending_K4",
            "same_winner_tasks":len(names) - changed if full4 else None,
            "changed_winner_tasks":changed,
            "K2_errors_corrected_by_K4":fixed,
            "K2_correct_cases_regressed_at_K4":regressed,
            "K2_K4_pivot_changes":sum(f2["pivot"] != full4[t]["pivot"]
                                       for t, f2 in full2.items()) if full4 else None,
        },
        "variants":records,
        "validation":all_checks,
        "limitations":[
            "Single fixed ring seed and cached scores; no multi-seed variance estimate.",
            "K1 rep=1 is a second-repeat sensitivity check, not an independent K1 run.",
            "Full PPT K1 is not identifiable from cache when its pivot introduces a new directed pair.",
            "Comparing cached-score variants does not constitute independent LLM reruns.",
            "Algorithmic score-record counts are not monetary cost or wall-clock time.",
            "Model/prompt changes are not tested by replay."
        ]
    }
    (output_dir / "full_benchmark_offline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== OFFLINE (ZERO API CALLS) ===")
    print("Cached entries:", len(scores), "K4 official report:", bool(report4))
    print("K2 full:", records_by_name["full_K2"]["overall_correct_89"], "/89",
          "K2 ring:", records_by_name["ring_K2_all3"]["overall_correct_89"], "/89")
    for label in ("ring_K1_rep0_all3", "ring_K1_rep1_all3", "full_K1_rep0_all3",
                  "full_K1_rep1_all3", "full_K4", "ring_K4_all3"):
        if label not in records_by_name:
            print(f"  {label}: pending K4 full report" if "K4" in label else "")
            continue
        r = records_by_name[label]
        if r["status"] == "complete":
            print(f"  {label}: {r['overall_correct_89']}/89, {r['score_requests_needed']} score records")
        else:
            print(f"  {label}: INCOMPLETE CACHE: {r['missing_score_keys']} missing records;")
            print("    See K1_full_ppt_cache_coverage.csv (DO NOT treat as a measured accuracy)")
    if full4:
        print(f"K2 -> K4: changed winners={changed}/24, "
              f"corrected errors={fixed}, regressions={regressed}")
    print("Saved output:", output_dir)
    print("New API requests: ZERO")
    return summary


def main():
    root = Path(__file__).resolve().parent
    default = root / DEFAULT_EXPERIMENT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-k2", type=Path, default=default / "report_K2.json")
    ap.add_argument("--report-k4", type=Path, default=default / "report_K4.json")
    ap.add_argument("--scores", type=Path, default=default / "scores.json")
    ap.add_argument("--out", type=Path, default=default / "full_benchmark_offline_analysis")
    args = ap.parse_args()
    run(args.report_k2, args.report_k4, args.scores, args.out)


if __name__ == "__main__":
    main()
