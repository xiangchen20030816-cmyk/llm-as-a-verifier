"""One-request smoke test for the LLM-as-a-Verifier DeepSeek backend.

Place this file at the repository ROOT (beside pyproject.toml).
Open it in VS Code and select the `ftec5660-verifier` Conda interpreter.

Safety: RUN_API_TEST is False by default, so the first run is offline.
When ready for a single paid call, set it to True and create a local .env
containing DEEPSEEK_API_KEY=... (never commit .env).

This is ONLY a connectivity/scoring check, not a benchmark reproduction.
"""

#1. 导入现成的评分功能
#2. 控制是否调用 API
#3. 读取 API Key
#4. 设置简化的模型参数
#5. 创建一个简单的测试任务
#6. 定义评价标准 Criterion
#7. 构建 Prompt
#8. 真正调用 DeepSeek
#9. 计算最终分数
#10. 统计 Token 消耗
from __future__ import annotations

import os
from pathlib import Path

from llm_verifier.fine_grained_reward import (
    USAGE,
    _find_tag_logprobs,
    build_prompt,
    call_verifier,
    create_deepseek_client,
    extract_score,
    load_dotenv,
)

ROOT = Path(__file__).resolve().parent
RUN_API_TEST = False  # IMPORTANT: change to True only when ready for a paid call


def main() -> None:
    load_dotenv(str(ROOT))
    print("=== Verifier API smoke test ===")
    print("Project root:", ROOT)
    print("Key configured:", bool(os.environ.get("DEEPSEEK_API_KEY")))
    print("Test enabled:", RUN_API_TEST)

    if not RUN_API_TEST:
        print("Offline preflight complete; no API request was made.")
        print("Set RUN_API_TEST = True after configuring the API key.")
        return

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("No DEEPSEEK_API_KEY found in environment or local .env.")
        print("No API request was made.")
        return

    # Keep the first tiny connectivity test separate from reproduction settings.
    # Normal benchmark experiments should use separately documented parameters.
    os.environ["DEEPSEEK_EFFORT"] = "off"
    os.environ["DEEPSEEK_MAX_TOKENS"] = "1024"

    problem = "Write a Python function reverse_string(s) that reverses a string."
    trace_a = (
        "Implemented: def reverse_string(s): return s[::-1]\n"
        "Test: reverse_string('abc') -> 'cba'"
    )
    trace_b = (
        "Implemented: def reverse_string(s): return s\n"
        "Test: reverse_string('abc') -> 'abc'"
    )
    criterion = {
        "id": "correctness",
        "name": "Correctness",
        "description": "Does the function reverse an input string as requested?",
    }
    prompt = build_prompt(
        problem,
        trace_a,
        trace_b,
        criterion,
        "Judge using the shown code and test outputs, not hidden labels.",
    )

    print("Sending ONE short comparison request to deepseek-flash ...")
    client = create_deepseek_client()
    before = USAGE.copy()
    text, tokens, position_logprobs = call_verifier(
        client, prompt, model="deepseek-flash"
    )
    usage = (USAGE - before).snapshot()

    tag_a = _find_tag_logprobs(tokens, position_logprobs, "<score_A>")
    tag_b = _find_tag_logprobs(tokens, position_logprobs, "<score_B>")
    score_a = extract_score(text, tokens, position_logprobs, "<score_A>")
    score_b = extract_score(text, tokens, position_logprobs, "<score_B>")

    print("\n=== Result ===")
    print(f"Candidate A score: {score_a:.6f}  (known-correct example)")
    print(f"Candidate B score: {score_b:.6f}  (known-incorrect example)")
    print("Score-token logprobs found for A:", bool(tag_a))
    print("Score-token logprobs found for B:", bool(tag_b))
    print("A scored higher than B:", score_a > score_b)
    print("Input tokens:", usage["input_tokens"])
    print("Output tokens:", usage["output_tokens"])
    print("Reasoning tokens (if reported):", usage["reasoning_tokens"])
    print("API calls recorded:", usage["calls"])
    print("\nResponse ending (for debugging):")
    print((text or "<empty>")[-500:])

    if not tag_a or not tag_b:
        print("\nWARNING: Score-token logprobs were not found for both candidates.")
        print("Any non-0.5 score may have come from text fallback; investigate.")
    else:
        print("\nLogprob extraction succeeded for both candidates.")


if __name__ == "__main__":
    main()
