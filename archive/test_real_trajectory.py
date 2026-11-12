"""One-pair, one-criterion smoke test on a REAL Terminal-Bench 2.1 trajectory.

Place in the repository root (beside pyproject.toml) or in scripts/.
Run directly in VS Code with the ftec5660-verifier Conda interpreter.

Safety measures:
- RUN_API_TEST=False by default: reading local data is free.
- EXACTLY ONE application-level verifier comparison (no full benchmark).
- SDK automatic retries are disabled for this experimental request.
- Prompt-character ceiling blocks unexpectedly large inputs.
- An existing JSON output blocks repeat billing unless explicitly allowed.
- Ground-truth reward labels are NEVER included in the LLM prompt.

This is a cost/compatibility probe, NOT a benchmark score or paper reproduction.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ============================================================================
# EDIT CONFIGURATION HERE; NO COMMAND-LINE ARGUMENTS ARE REQUIRED
# ============================================================================
BENCHMARK_KEY = "terminal_bench_2.1"
TASK_NAME = "kv-store-grpc"
BEST_OF_N = 3
CRITERION_ID = "specification"  # one official criterion, not all three
MODEL = "deepseek-flash"

RUN_API_TEST = False            # IMPORTANT: change to True only when ready
ALLOW_REPEAT_API_CALL = False   # True only when you really intend to re-run
MAX_PROMPT_CHARS = 65_000       # a safeguard, NOT a token or money limit

# These are deliberately CHEAPER diagnostic settings. Formal reproduction
# needs separately documented parameters; do not treat this as paper settings.
TEST_REASONING_EFFORT = "off"
TEST_MAX_OUTPUT_TOKENS = 4096

# ============================================================================
# APPLICATION CODE (normally no need to edit)
# ============================================================================
HERE = Path(__file__).resolve().parent
ROOT = next(
    (p for p in (HERE, HERE.parent)
     if (p / "llm_verifier").is_dir() and (p / "pyproject.toml").exists()),
    None,
)
if ROOT is None:
    raise RuntimeError(
        "Cannot locate repo root. Put this file in the project root or scripts/."
    )
sys.path.insert(0, str(ROOT))

from llm_verifier.benchmarks import BENCHMARKS  # noqa: E402
from llm_verifier.fine_grained_reward import (  # noqa: E402
    USAGE,
    _find_tag_logprobs,
    build_prompt,
    call_verifier,
    create_deepseek_client,
    extract_score,
    load_dotenv,
)
from llm_verifier.loaders import LOADERS  # noqa: E402
from llm_verifier.prompts import load_prompts, select_criteria  # noqa: E402


def pick_shortest_opposite_reward_pair(trials: list[dict]) -> tuple[int, int]:
    """Return (failed index, successful index), using only the first N trials.

    Shortest-pair selection is for DEBUGGING ONLY and is NOT representative
    sampling for formal benchmark accuracy experiments.
    """
    failures = [i for i, t in enumerate(trials) if t["reward"] == 0]
    successes = [i for i, t in enumerate(trials) if t["reward"] == 1]
    if not failures or not successes:
        raise ValueError("Task is not a swing task: need both reward=0 and reward=1")
    failed = min(failures, key=lambda i: (len(trials[i]["trace"]), i))
    succeeded = min(successes, key=lambda i: (len(trials[i]["trace"]), i))
    return failed, succeeded


def main() -> None:
    load_dotenv(str(ROOT))
    print("=== Real trajectory verifier smoke test (one comparison) ===")
    print("Project root:", ROOT)
    print("Benchmark:", BENCHMARK_KEY)
    print("Task:", TASK_NAME)
    print("Best-of-N:", BEST_OF_N)
    print("Criterion:", CRITERION_ID)
    print("Model:", MODEL)
    print("Key configured:", bool(os.environ.get("DEEPSEEK_API_KEY")))
    print("API enabled:", RUN_API_TEST)

    if BENCHMARK_KEY not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {BENCHMARK_KEY}")
    cfg = BENCHMARKS[BENCHMARK_KEY]

    # Read the original benchmark data and original criteria, exactly as the
    # official run.py pipeline does. The user's Windows UTF-8 loader fix is
    # still needed in llm_verifier/loaders.py.
    tasks, _ = LOADERS[cfg.loader](cfg.data, str(ROOT))
    if TASK_NAME not in tasks:
        print("Example task names:", sorted(tasks)[:8])
        raise KeyError(f"Task {TASK_NAME!r} not found")
    trials = tasks[TASK_NAME][:BEST_OF_N]
    if len(trials) != BEST_OF_N:
        raise ValueError(
            f"Expected {BEST_OF_N} trials for {TASK_NAME}; got {len(trials)}"
        )
    bad_index, good_index = pick_shortest_opposite_reward_pair(trials)
    bad, good = trials[bad_index], trials[good_index]

    try:
        note, criteria_all = load_prompts(cfg.prompts)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Windows text decoding issue in llm_verifier/prompts.py. "
            "Add encoding='utf-8' to both open(path) and open(candidate) "
            "inside _read_criteria(), then rerun."
        ) from exc
    criterion = select_criteria(criteria_all, [CRITERION_ID])[0]

    # Candidate A = shorter FAILED trace; B = shorter SUCCESSFUL trace.
    # Reward is used only here for selecting a diagnostic pair and AFTER
    # inference for evaluation. It is never sent in the prompt.
    problem = bad["problem"]
    if good["problem"] != problem:
        print("NOTE: trial problem texts differ; using trial A's task statement")
    prompt = build_prompt(problem, bad["trace"], good["trace"], criterion, note)

    print("\n--- Offline preflight ---")
    print("Selected trial indices (1-based):", bad_index + 1, good_index + 1)
    print("Trace A chars:", f"{len(bad['trace']):,}")
    print("Trace B chars:", f"{len(good['trace']):,}")
    print("Prompt chars:", f"{len(prompt):,}")
    print("Prompt under safeguard:", len(prompt) <= MAX_PROMPT_CHARS)
    print("Reasoning effort:", TEST_REASONING_EFFORT)
    print("Maximum output tokens:", TEST_MAX_OUTPUT_TOKENS)
    print("IMPORTANT: prompt characters are NOT billed input tokens or a cost cap.")

    output_file = (
        ROOT / "analysis_results" /
        f"smoke_{BENCHMARK_KEY}_{TASK_NAME}_{CRITERION_ID}.json"
    )
    print("Output file:", output_file)
    if not RUN_API_TEST:
        print("\nOFFLINE ONLY: No API call was made.")
        print("Review the above information, then set RUN_API_TEST = True.")
        return

    if len(prompt) > MAX_PROMPT_CHARS:
        print("STOP: prompt is larger than MAX_PROMPT_CHARS; no request sent.")
        return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("STOP: DEEPSEEK_API_KEY missing; no request sent.")
        return
    if output_file.exists() and not ALLOW_REPEAT_API_CALL:
        print("STOP: output JSON exists. A repeat request could incur charges.")
        print("Only set ALLOW_REPEAT_API_CALL=True if you want to run again.")
        return

    # This mode deliberately differs from the author's long-budget,
    # high-reasoning full reproduction, so report it as a smoke test only.
    os.environ["DEEPSEEK_EFFORT"] = TEST_REASONING_EFFORT
    os.environ["DEEPSEEK_MAX_TOKENS"] = str(TEST_MAX_OUTPUT_TOKENS)

    client = create_deepseek_client(model=MODEL)
    # Disable OpenAI SDK's implicit retries to avoid unexpected extra paid
    # attempts on transient errors. Preserve the project's custom tags.
    client = client.with_options(max_retries=0)
    client._llm_verifier_model = MODEL
    client._llm_verifier_deepseek = True

    print("\nSending a SINGLE real-data comparison request...")
    usage_before = USAGE.copy()
    try:
        reply, tokens, position_logprobs = call_verifier(
            client, prompt, model=MODEL
        )
    except Exception as exc:
        print(f"API call failed ({type(exc).__name__}): {exc}")
        print("No second request will be attempted by this script.")
        return
    usage = (USAGE - usage_before).snapshot()

    lp_a = _find_tag_logprobs(tokens, position_logprobs, "<score_A>")
    lp_b = _find_tag_logprobs(tokens, position_logprobs, "<score_B>")
    score_a = extract_score(reply, tokens, position_logprobs, "<score_A>")
    score_b = extract_score(reply, tokens, position_logprobs, "<score_B>")
    lp_ok = bool(lp_a) and bool(lp_b)

    # Only now print the held-out reward labels; they were never in prompt.
    print("\n=== Verifier result ===")
    print(f"Candidate A: score={score_a:.6f}, ground-truth reward={bad['reward']}")
    print(f"Candidate B: score={score_b:.6f}, ground-truth reward={good['reward']}")
    print("A logprobs found:", bool(lp_a))
    print("B logprobs found:", bool(lp_b))
    print("Correct pair preference (B > A):", score_b > score_a)
    print("Input tokens:", usage["input_tokens"])
    print("Cached input tokens:", usage["cached_input_tokens"])
    print("Output tokens:", usage["output_tokens"])
    print("Reasoning tokens:", usage["reasoning_tokens"])
    print("Recorded API calls:", usage["calls"])
    if not lp_ok:
        print("WARNING: Missing scoring-token logprobs; score may be fallback.")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "one-pair diagnostic only; NOT full Bo3 benchmark reproduction",
        "benchmark": BENCHMARK_KEY,
        "best_of": BEST_OF_N,
        "task": TASK_NAME,
        "criterion": CRITERION_ID,
        "model": MODEL,
        "reasoning_effort": TEST_REASONING_EFFORT,
        "max_output_tokens": TEST_MAX_OUTPUT_TOKENS,
        "pair_selection": "shortest failed and shortest successful among first 3",
        "trial_a": {"index_1based": bad_index + 1, "trial_name": bad.get("trial_name", ""),
                    "reward": bad["reward"], "trace_chars": len(bad["trace"]),
                    "score": score_a, "logprobs_found": bool(lp_a)},
        "trial_b": {"index_1based": good_index + 1, "trial_name": good.get("trial_name", ""),
                    "reward": good["reward"], "trace_chars": len(good["trace"]),
                    "score": score_b, "logprobs_found": bool(lp_b)},
        "prompt_chars": len(prompt),
        "correct_pair_preference": score_b > score_a,
        "token_usage": usage,
        "response_tail": (reply or "")[-600:],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Results saved:", output_file)
    print("\nResponse ending (last 600 chars):")
    print((reply or "<empty>")[-600:])
    if lp_ok:
        print("\nSUCCESS: both score-token logprobs were extracted.")
    else:
        print("\nINCOMPLETE: inspect score tags before using this for research.")


if __name__ == "__main__":
    main()
