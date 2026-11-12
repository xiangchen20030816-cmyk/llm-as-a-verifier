"""Run an explicitly bounded, reproducible Terminal-Bench 2.1 Best-of-3 study.

This is a no-CLI VS Code runner. It calls the ORIGINAL scripts/run.py
`run_benchmark` (including ring pass, pivot selection and pivot round).
Only the dataset loader is filtered in memory to a fixed small subset.
The underlying grading, PPT, cache and metrics are unchanged.

First run with RUN_PAID_EXPERIMENT=False and inspect its dry-run preview.
Then explicitly enable one K at a time. No requests are made by default.

IMPORTANT: A pilot or a stratified small subset is NOT the full 89-task
published benchmark. Report its accuracy as *subset accuracy* only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ===================== EDIT THESE SETTINGS ================================
BENCHMARK_KEY = "terminal_bench_2.1"
N_TRIALS = 3                  # official best-of-3
PIVOTS = 1                    # official best-of-3
SEED = 0                      # fixed for BOTH K values
K = 2                         # 2 = baseline, 4 = experimental change
MODEL = "deepseek-flash"

# Choose ONE way to select the dataset:
# "pilot" : just kv-store-grpc (smoke/pilot, cannot estimate benchmark accuracy)
# "stratified" : fixed-seed sample of all-pass, swing, all-fail tasks;
#                intentionally not population-representative.
SELECTION_MODE = "pilot"
PILOT_TASKS = [
    "kv-store-grpc",
    "merge-diff-arc-agi-task",
    "write-compressor",
]
STRATIFIED_COUNTS = {"all_pass": 2, "swing": 3, "all_fail": 1}
SELECTION_SEED = 2026

# OFF by default. Flip to True ONLY after reading the preview/cost budget.
RUN_PAID_EXPERIMENT = True
MAX_NEW_API_CALLS = 100
MAX_PROMPT_CHARS = 160_000
# MAX_NEW_API_CALLS = 30        # HARD preflight upper-bound cap, not a price cap
# MAX_PROMPT_CHARS = 100_000    # HARD per-prompt length check, not a token cap
MAX_WORKERS = 1               # minimize parallel surprises and costs

# Fixed across baseline K=2 and modified K=4. The original project's high
# thinking setting and output budget are used here; they can be COSTLY.
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 65_536

ALLOW_OVERWRITE_RESULT = False  # don't silently rerun completed K
# ===========================================================================

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, HERE.parent)
             if (p / "llm_verifier").is_dir()
             and (p / "scripts" / "run.py").is_file()
             and (p / "pyproject.toml").exists()), None)
if ROOT is None:
    raise RuntimeError("Put this file in the repo root or scripts/ directory")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run as official_run  # noqa: E402
from llm_verifier import pivot_tournament as ppt  # noqa: E402
from llm_verifier.benchmarks import BENCHMARKS  # noqa: E402
from llm_verifier import fine_grained_reward as fgr  # noqa: E402
from llm_verifier.prompts import load_prompts, select_criteria  # noqa: E402


def classify_complete_trials(tasks):
    groups = {"all_pass": [], "swing": [], "all_fail": []}
    for name, trials in sorted(tasks.items()):
        if len(trials) != N_TRIALS:
            raise ValueError(f"{name}: expected {N_TRIALS} trials; got {len(trials)}")
        rewards = [t["reward"] for t in trials]
        if any(r not in (0, 1) for r in rewards):
            raise ValueError(f"{name}: expected binary 0/1 rewards, got {rewards}")
        category = "all_pass" if all(rewards) else (
            "all_fail" if not any(rewards) else "swing")
        groups[category].append(name)
    return groups


def choose_subset(tasks, groups):
    if SELECTION_MODE == "pilot":
        chosen = list(dict.fromkeys(PILOT_TASKS))
        if not chosen:
            raise ValueError("PILOT_TASKS is empty")
        missing = set(chosen) - set(tasks)
        if missing:
            raise KeyError(f"Selected task(s) not in data: {sorted(missing)}")
        return sorted(chosen)
    if SELECTION_MODE == "stratified":
        rng = random.Random(SELECTION_SEED)
        chosen = []
        for category in ("all_pass", "swing", "all_fail"):
            n = STRATIFIED_COUNTS[category]
            if n < 0 or n > len(groups[category]):
                raise ValueError(f"Impossible sample count for {category}: {n}")
            chosen += rng.sample(groups[category], n)
        if not chosen:
            raise ValueError("Stratified sample is empty")
        return sorted(chosen)
    raise ValueError(f"Unknown SELECTION_MODE: {SELECTION_MODE}")


def fingerprint(cfg, selected_tasks, note, criteria):
    """Include inputs, model and prompts in cache identity; intentionally omit K.

    K=4 can safely reuse K=2 repetitions 0 and 1 when all other settings
    are identical. A different model/prompt/reasoning setting gets a new cache.
    """
    items = {
        "benchmark": BENCHMARK_KEY, "name": cfg.name,
        "selection": selected_tasks, "model": MODEL,
        "n_trials": N_TRIALS, "pivots": PIVOTS, "seed": SEED,
        "criteria": criteria, "note": note,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "traces": [
            (name, [{"problem": t["problem"], "trace": t["trace"]}
                    for t in selected_tasks[name]])
            for name in sorted(selected_tasks)
        ],
    }
    # Include the content rather than only filenames so edits cannot silently
    # reuse scores from an earlier, different prompt/trajectory.
    raw = json.dumps(items, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cached_score_keys(path):
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Existing score cache has invalid format")
    return set(data)


def worst_case_new_api_calls(selected_tasks, criteria, existing_keys):
    """Upper bound that includes both PPT phases for EVERY possible pivot.

    The phase-B pivot is only known after phase A. Enumerating the 3 possible
    pivots ensures a safe ceiling before we make any paid call.
    """
    rng = random.Random(SEED)
    total_max = 0
    exact_ring_new = 0
    for name, trials in sorted(selected_tasks.items()):
        rewards = [t["reward"] for t in trials]
        if not (0 in rewards and 1 in rewards):
            continue
        n = len(trials)
        ring = ppt.ring_cycle(n, rng)
        ring_keys = {
            fgr.cache_key(c["id"], name, a, b, rep)
            for a, b in ring for c in criteria for rep in range(K)
        }
        exact_ring_new += len(ring_keys - existing_keys)
        worst_for_this_task = 0
        for pivot in range(n):  # k=1 in this specific Bo3 experiment
            all_pairs = set(ring) | set(ppt.pivot_round_pairs(n, [pivot]))
            keys = {
                fgr.cache_key(c["id"], name, a, b, rep)
                for a, b in all_pairs for c in criteria for rep in range(K)
            }
            worst_for_this_task = max(
                worst_for_this_task, len(keys - existing_keys))
        total_max += worst_for_this_task
    return exact_ring_new, total_max


def max_prompt_size(selected_tasks, criteria, note):
    largest, largest_task = 0, None
    for name, trials in sorted(selected_tasks.items()):
        rewards = [t["reward"] for t in trials]
        if not (0 in rewards and 1 in rewards):
            continue
        for a in range(len(trials)):
            for b in range(len(trials)):
                if a == b:
                    continue
                for crit in criteria:
                    length = len(fgr.build_prompt(
                        trials[a]["problem"], trials[a]["trace"],
                        trials[b]["trace"], crit, note,
                        n_images=len(fgr.as_image_list(trials[a].get("images")))
                    ))
                    if length > largest:
                        largest, largest_task = length, name
    return largest, largest_task


def git_head():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def reconstruct_details(selected_tasks, criteria, cache_path):
    """Post-run audit: recover winners with ORIGINAL ppt functions.

    Guard against incomplete score caches before claiming a valid result.
    """
    scores = json.loads(cache_path.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    rows, correct, required_keys = [], 0, set()
    for name, trials in sorted(selected_tasks.items()):
        rewards = [t["reward"] for t in trials]
        if not (0 in rewards and 1 in rewards):
            continue
        n = len(trials)
        ring = ppt.ring_cycle(n, rng)

        def directed(a, b):
            if a != b:
                for cid in [c["id"] for c in criteria]:
                    for rep in range(K):
                        key = fgr.cache_key(cid, name, a, b, rep)
                        if key not in scores:
                            raise RuntimeError(f"Missing required score: {key}")
                        required_keys.add(key)
            return fgr.directed_reward(
                scores, name, a, b, [c["id"] for c in criteria], K)

        w, counts = [0.] * n, [0] * n
        ppt.accumulate(ring, directed, w, counts)
        pivots = ppt.select_pivots(w, counts, PIVOTS)
        pr_pairs = ppt.pivot_round_pairs(n, pivots)
        ppt.accumulate(pr_pairs, directed, w, counts)
        best = max(range(n), key=lambda i: (w[i] / counts[i]
                    if counts[i] else 0., -i))
        correct += rewards[best]
        rows.append({
            "task": name, "chosen_trial_1based": best + 1,
            "chosen_reward": rewards[best],
            "trial_rewards": "".join(map(str, rewards)),
            "pivot_indices_1based": ",".join(str(x + 1) for x in pivots),
            "comparisons": len(ring) + len(pr_pairs),
        })
    return rows, correct, required_keys


def main():
    fgr.load_dotenv(str(ROOT))
    if BENCHMARK_KEY != "terminal_bench_2.1":
        raise ValueError("This controlled runner is restricted to TB 2.1")
    if N_TRIALS != 3 or PIVOTS != 1 or K not in (2, 4):
        raise ValueError("Use N=3, pivots=1, K=2 or 4 for this experiment")
    if MAX_WORKERS != 1:
        raise ValueError("Keep MAX_WORKERS=1 for the bounded initial runs")
    if REASONING_EFFORT not in ("off", "low", "high", "max"):
        raise ValueError("Invalid DeepSeek effort setting")

    cfg = BENCHMARKS[BENCHMARK_KEY]
    note, all_criteria = load_prompts(cfg.prompts)
    criteria = select_criteria(all_criteria, cfg.criteria)
    original_loader = official_run.LOADERS[cfg.loader]
    print("=== Terminal-Bench 2.1 Best-of-3 | Small PPT Experiment ===", flush=True)
    print("Loading original trajectories for offline preflight...", flush=True)
    all_tasks, _ = original_loader(cfg.data, str(ROOT))
    all_tasks = {name: trials[:N_TRIALS] for name, trials in all_tasks.items()}
    groups = classify_complete_trials(all_tasks)
    chosen = choose_subset(all_tasks, groups)
    subset = {name: all_tasks[name] for name in chosen}
    selected_groups = classify_complete_trials(subset)
    n_swing = len(selected_groups["swing"])

    if not n_swing:
        raise ValueError("Chosen subset has no Swing Task; no verifier calls")
    cache_id = fingerprint(cfg, subset, note, criteria)
    output_dir = ROOT / "analysis_results" / "small_benchmark" / cache_id
    cache_path = output_dir / "scores.json"
    txt_path = output_dir / f"report_K{K}.txt"
    json_path = output_dir / f"report_K{K}.json"
    csv_path = output_dir / f"task_decisions_K{K}.csv"
    cached = cached_score_keys(cache_path)
    ring_new, maximum_new = worst_case_new_api_calls(subset, criteria, cached)
    largest_prompt, largest_task = max_prompt_size(subset, criteria, note)

    pass1 = sum(sum(t["reward"] for t in trials) / len(trials)
                for trials in subset.values()) / len(subset)
    oracle = sum(any(t["reward"] == 1 for t in trials)
                 for trials in subset.values()) / len(subset)
    print(f"Git HEAD: {git_head()}")
    print(f"Selection: {SELECTION_MODE}, seed={SELECTION_SEED}")
    print(f"Selected tasks ({len(chosen)}): {chosen}")
    print("Subset composition: " + ", ".join(
        f"{g}={len(v)}" for g, v in selected_groups.items()))
    print(f"Subset Pass@1={pass1:.2%}, Oracle={oracle:.2%}")
    print(f"N={N_TRIALS}, C={len(criteria)}, pivot={PIVOTS}, "
          f"K={K}, ring seed={SEED}, workers={MAX_WORKERS}")
    print(f"Verifier: {MODEL}, effort={REASONING_EFFORT}, "
          f"max_output_tokens={MAX_OUTPUT_TOKENS:,}")
    print(f"Largest possible pairwise prompt: {largest_prompt:,} chars "
          f"(task: {largest_task})")
    print(f"Already cached score keys: {len(cached)}")
    print(f"Phase-A uncached calls: {ring_new}")
    print(f"Maximum NEW API calls (including any pivot): {maximum_new}")
    print(f"Safeguard limit: {MAX_NEW_API_CALLS} calls; "
          f"{MAX_PROMPT_CHARS:,} prompt chars")
    print(f"Cache shared by K=2 and K=4: {cache_path}")
    print(f"Result: {json_path}")
    print("NOTE: call upper bound excludes provider-side/internal retries "
          "(SDK retries are disabled); it is NOT a price cap.")
    print("NOTE: small/stratified subset accuracy is NOT full-89-task accuracy.")

    if not RUN_PAID_EXPERIMENT:
        print("\nOFFLINE PREVIEW COMPLETE. No API calls made.")
        print("Inspect the call budget, then set RUN_PAID_EXPERIMENT=True "
              "if you choose to pay for this K run.")
        return
    if maximum_new > MAX_NEW_API_CALLS:
        print("\nSTOP: predicted worst-case NEW calls exceed MAX_NEW_API_CALLS.")
        print("No API requests made. Increase limit ONLY after reviewing cost.")
        return
    if largest_prompt > MAX_PROMPT_CHARS:
        print("\nSTOP: a possible PPT comparison exceeds MAX_PROMPT_CHARS.")
        print("No API requests made. Do not silently truncate the trajectories.")
        return
    if json_path.exists() and not ALLOW_OVERWRITE_RESULT:
        print("\nSTOP: results for this K already exist; avoid repeated billing.")
        return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\nSTOP: DEEPSEEK_API_KEY is missing; no API request made.")
        return
    if os.environ.get("OPENAI_BASE_URL"):
        print("\nSTOP: OPENAI_BASE_URL would override DeepSeek in the official code.")
        print("Unset that variable to use the intended verification backend.")
        return

    # We do not modify scripts/run.py. Instead its OWN global LOADERS registry
    # is replaced with an isolated copy that returns only these chosen tasks.
    # All later calculations remain inside the original run_benchmark().
    official_run.LOADERS = dict(official_run.LOADERS)

    def selected_loader(_data, _root_dir):
        return subset, N_TRIALS

    official_run.LOADERS[cfg.loader] = selected_loader

    # Original score_directed_pairs already caches successful calls. Failures
    # are normally treated as 0.5 ties; require an error instead of allowing
    # an API outage to affect the measured benchmark success rate.
    original_scorer = official_run.score_directed_pairs

    def strict_scorer(*args, **kwargs):
        kwargs["on_error"] = "raise"
        kwargs["model"] = MODEL
        return original_scorer(*args, **kwargs)

    official_run.score_directed_pairs = strict_scorer

    class NoRetryDeepSeekClient:
        def __init__(self):
            self.client = None

        def get(self):
            if self.client is None:
                client = fgr.create_deepseek_client(model=MODEL)
                client = client.with_options(max_retries=0)
                # with_options() can discard these custom tags.
                client._llm_verifier_model = MODEL
                client._llm_verifier_deepseek = True
                self.client = client
            return self.client

    official_run.LazyClient = NoRetryDeepSeekClient

    # The cache hash includes these settings. NEVER use the off-reasoning
    # smoke-test cache here as though it were an official baseline.
    os.environ["DEEPSEEK_EFFORT"] = REASONING_EFFORT
    os.environ["DEEPSEEK_MAX_TOKENS"] = str(MAX_OUTPUT_TOKENS)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nPAID RUN STARTING — same official PPT algorithm, filtered tasks.", flush=True)
    try:
        result = official_run.run_benchmark(
            cfg, pivots=PIVOTS, n_evaluations=K, seed=SEED,
            max_workers=MAX_WORKERS, n_trials=N_TRIALS,
            cache_file=str(cache_path), results_file=str(txt_path),
        )
        decisions, audited_correct, used_keys = reconstruct_details(
            subset, criteria, cache_path)
        expected_correct = result["verifier"] - len(selected_groups["all_pass"])
        if audited_correct != expected_correct:
            raise RuntimeError("Post-run task decision audit disagrees with official metrics")
        official_run.report(cfg, **result)
    except Exception as exc:
        print(f"\nINCOMPLETE RUN ({type(exc).__name__}): {exc}")
        print("Do not interpret incomplete results as measured accuracy.")
        print("Successful prior scores may be persisted in the cache.")
        raise

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "chosen_trial_1based", "chosen_reward",
            "trial_rewards", "pivot_indices_1based", "comparisons"])
        writer.writeheader()
        writer.writerows(decisions)

    usage = result["usage"].snapshot()
    already_reused = len(used_keys & cached)
    newly_required = len(used_keys - cached)
    if usage["calls"] != newly_required:
        print("WARNING: response-reported API calls do not equal newly required "
              "score keys. Inspect usage metadata and cache before analysis.")
    print(f"Algorithmic verification requests at K={K}: {len(used_keys)}")
    print(f"Reused score entries from earlier runs: {already_reused}")
    print(f"Newly scored entries: {newly_required}")
    print("IMPORTANT: recorded token usage reflects only newly executed calls; "
          "it is NOT standalone K-run token cost if scores were reused.")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "git_head": git_head(),
        "experiment": "Terminal-Bench 2.1 Best-of-3 small-subset PPT",
        "NOT_full_89_task_benchmark": True,
        "selection_mode": SELECTION_MODE, "selection_seed": SELECTION_SEED,
        "selected_tasks": chosen,
        "subset_categories": {g: len(v) for g, v in selected_groups.items()},
        "model": MODEL, "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "n_trials": N_TRIALS, "criteria": cfg.criteria,
        "pivots": PIVOTS, "K": K, "ring_seed": SEED,
        "n_tasks": result["n_tasks"], "n_swing": result["n_swing"],
        "pass1": result["pass1"] / len(subset),
        "verifier_rate": result["verifier"] / len(subset),
        "oracle": result["oracle"] / len(subset),
        "swing_accuracy": audited_correct / len(decisions),
        "swing_correct": audited_correct,
        "comparisons": result["comparisons"],
        "algorithmic_verification_requests": len(used_keys),
        "reused_previous_scores": already_reused,
        "newly_required_score_entries": newly_required,
        "new_api_calls_reported": usage["calls"],
        "incremental_token_usage_NOT_standalone_cost": usage,
        "task_decisions_csv": str(csv_path),
        "cache_path": str(cache_path),
        "task_decisions": decisions,
    }
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"Task decisions: {csv_path}")
    print(f"JSON report: {json_path}")
    print("\nCOMPLETED. Keep all settings except K fixed for comparison.")


if __name__ == "__main__":
    main()
