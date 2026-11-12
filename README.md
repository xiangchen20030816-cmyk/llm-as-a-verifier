# FTEC5660 Reproducibility Project: LLM-as-a-Verifier

**Terminal-Bench 2.1 Reproduction and Offline Pivot Round Ablation**

**Course:** FTEC5660 — Agentic AI for Business and FinTech  
**Author:** Chen Xiang      1155246146 

## 1. Project Overview

This project uses the public code and agent execution trajectories from *LLM-as-a-Verifier* to reproduce the **Terminal-Bench 2.1 Best-of-3 trajectory-selection benchmark** reported in the authors' official repository. It also examines how scoring repetitions (`K`) and the Pivot Round of the Probabilistic Pivot Tournament (PPT) affect trajectory selection.

The underlying agent trajectories were **not regenerated**. Full PPT at `K=2` and `K=4` was evaluated using newly obtained verifier API scores. Ring-only, `K=1`, and evaluation-criterion ablations were conducted through **offline replay of the existing score cache**. Ring-only is an ablation analysis, not a newly implemented or independently deployed ranking algorithm.

## 2. Main Results

The full benchmark contains **89 tasks**. Based on the ground-truth rewards of the first three candidate trajectories, these comprise **58 All-pass, 24 Swing, and 7 All-fail** tasks. In the authors' reward-aware offline evaluation procedure, only Swing tasks require paid verifier comparisons.

| Configuration    | Successful Swing tasks |    Overall success | Required score records | Evaluation method    |
| ---------------- | ---------------------: | -----------------: | ---------------------: | -------------------- |
| Full PPT, `K=2`  |                  19/24 | **77/89 (86.52%)** |                    576 | API-generated scores |
| Ring-only, `K=2` |                  19/24 | **77/89 (86.52%)** |                    432 | Offline ablation     |
| Full PPT, `K=4`  |                  19/24 | **77/89 (86.52%)** |                  1,152 | API-generated scores |
| Ring-only, `K=4` |                  19/24 | **77/89 (86.52%)** |                    864 | Offline ablation     |

The official repository reports **86.5% ± 1.1%** verifier success for this Best-of-3 benchmark. Our `K=2` result matches its displayed value. The corresponding **Pass@1 is 79.40%**, and **Oracle (Best-of-3) is 92.13%**.

The main observation is that removing the Pivot Round preserved the **same winning trajectory on all 24/24 Swing tasks** at both `K` values, while reducing the *required number of score records* by **25%**. Increasing `K` from 2 to 4 did not improve overall success, although three tasks switched between successful candidate trajectories. In an additional exploratory offline ablation, using `output_match` alone at `K=2` yielded **78/89 (87.64%)**, but this was not independently validated.

> **Interpretation:** The 25% reduction is a counterfactual reduction in required scoring records calculated offline, **not a measured reduction in API expenditure or runtime**. Ablation findings are specific to the benchmark, Ring seed, and cached score realizations used here.

## 3. Experimental Configuration

| Item                     | Setting                                                      |
| ------------------------ | ------------------------------------------------------------ |
| Benchmark and candidates | Terminal-Bench 2.1, 89 tasks, first three released trajectories per task (Best-of-3) |
| Trajectory source        | Authors' `mini-swe-agent_deepseek-v4-flash` dataset          |
| Verifier                 | `deepseek-flash` API alias; the exact historical backend model version was not pinned |
| Evaluation criteria      | `specification`, `output_match`, `error_signals`             |
| PPT configuration        | `N=3`, one pivot, Ring seed `0`                              |
| Scoring                  | `K=2` and `K=4`; `reasoning_effort=high`; `max_output_tokens=65536` |
| Base code commit         | `90e8fc215606447d6d31e9e24bff7200fb0c0994` (Git commit recorded for the benchmark runs) |

Both full evaluations used a fixed Ring seed rather than independent repetitions over multiple seeds. The original evaluation prompts were retained; no controlled prompt-rewriting experiment was conducted. Because the model API conditions may differ from those of the historical experiment, this project reproduces the **PPT verification and selection procedure**, not the complete original self-verification setting.

## 4. Repository Structure and Files

Only the main files and directories relevant to this coursework reproduction are shown below. Other files from the upstream repository are retained.

```text
llm-as-a-verifier/
├── README.md                          # Project overview and reproduction guide
├── run_benchmark_extended.py          # Main experiment runner
├── analyze_terminal_bench.py          # Offline statistics for trajectories and task rewards
├── analyze_full_benchmark_offline.py  # Reconstruct PPT rankings, validate results, replay ablations
├── analysis_results/                  # Pilot/full reports, score cache, and analysis outputs
├── archive/                           # Early small-scale experiments and helper scripts
│   ├── figures/                       # Early or supplementary figures
│   ├── test_verifier_api.py           # API connectivity smoke test
│   ├── test_real_trajectory.py        # Scoring test using a real trajectory
│   ├── run_small_benchmark.py         # Early small-scale benchmark experiments
│   ├── run_ring_only_ablation.py      # Early Ring-only ablation check
│   └── generate_report_figures_from_results.py  # Supplementary figure generation
├── llm_verifier/                       # Authors' core verifier and PPT implementation
├── scripts/                            # Original scripts, including scripts/run.py
├── data/                               # Released benchmark tasks and agent trajectories
├── criteria/                           # Evaluation criteria
├── assets/                             # Original repository assets
├── .env.example                        # Example environment variables (no real secrets)
├── pyproject.toml                      # Dependencies and package configuration
└── Reproducibility_Report.pdf          # Course report (add before submission)
```

**Main coursework additions:**

- **`run_benchmark_extended.py`** — Runs pilots, stratified subsets, and the full 89-task benchmark using the original PPT implementation in `scripts/run.py`. It adds request safeguards, persistent score-cache resumption, and result export.
- **`analyze_terminal_bench.py`** — **No API calls.** Analyzes task categories, Pass@1, Oracle, and trajectory lengths to support pilot selection. It does not evaluate verifier accuracy.
- **`analyze_full_benchmark_offline.py`** — **No API calls.** Reconstructs and validates Full PPT rankings from `report_K2.json`, `report_K4.json`, and `scores.json`, then performs Ring-only, `K=1`, and evaluation-criterion ablations.
- **`analysis_results/`** — Stores pilot results, completed `K=2`/`K=4` benchmark reports, the shared score cache, and offline CSV/JSON analysis outputs.
- **`archive/`** — Contains simple early experiments, API checks, and plotting utilities kept as a record of exploration. **It is not required for the main reproduction workflow.**

## 5. Installation

The study was conducted mainly in Windows PowerShell with Conda. From the repository root:

```powershell
conda create -n ftec5660-verifier python=3.11 -y
conda activate ftec5660-verifier
python -m pip install -e .
```

No API key is needed for offline analysis. To run new online verifier scoring, configure `DEEPSEEK_API_KEY` using `.env.example` as a template. **Never commit `.env` or real API credentials.**

## 6. Reproduce the Existing Results (Recommended: No API Calls)

Ensure the following files are present in `analysis_results/small_benchmark/60e63e37b82a1550/`:

- `report_K2.json`
- `report_K4.json`
- `scores.json`

Then run from the repository root:

```powershell
python .\analyze_full_benchmark_offline.py
```

The script verifies the original Full PPT outcomes and generates offline analyses for different `K` values, Ring-only, individual evaluation criteria, and failure cases. Expected key results include **77/89 successful tasks for both `K=2` and `K=4`**, identical Full PPT and Ring-only winners at each `K`, and **three changed winners** from `K=2` to `K=4`.

If the inputs are stored elsewhere, use `--report-k2`, `--report-k4`, `--scores`, and `--out`:

```powershell
python .\analyze_full_benchmark_offline.py `
  --report-k2 "path\to\report_K2.json" `
  --report-k4 "path\to\report_K4.json" `
  --scores "path\to\scores.json" `
  --out "path\to\offline_analysis_output"
```

The shared cache lacks some directed comparisons needed for a complete Full PPT evaluation at `K=1`; therefore, the available `K=1` replays must **not** be presented as complete Full PPT `K=1` results.

## 7. Re-run the Online Experiment (Optional: Paid API Requests)

`run_benchmark_extended.py` is configured by editing constants near the beginning of the script, **not by command-line arguments**. Set `SELECTION_MODE="full"`, `N_TRIALS=3`, `PIVOTS=1`, and `SEED=0`, then choose `K=2` or `K=4`.

**First, leave `RUN_PAID_EXPERIMENT=False`** and run:

```powershell
python .\run_benchmark_extended.py
```

Review the dry-run preview, including the selected tasks, prompt lengths, and request safeguards. Only when intentionally repeating paid scoring should you configure the API key and explicitly set `RUN_PAID_EXPERIMENT=True`. The runner caches completed scores to support resumption after interruptions. **Do not overwrite the completed benchmark results just to demonstrate the script.**

## 8. Debug Notes and Limitations

The main practical issues were Windows GBK decoding errors, network/API interruptions, and the high cost of scoring long trajectories. These were addressed through encoding fixes, persistent caching and resumption, progressive testing from small pilots to the full benchmark, and offline reuse of existing scores.

This study did not rerun the underlying agent, reproduce variability across multiple independent Ring seeds, or independently measure actual Ring-only cost savings. Inspection of selected failure trajectories revealed possible implementation differences, but whether verifier prompt design caused the incorrect preferences remains an untested hypothesis.

## 9. Attribution

This project builds on the [official LLM-as-a-Verifier implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier) and its released trajectories. The original algorithm and core code are credited to their authors. The coursework-specific contributions mainly concern experiment management, result validation, and offline ablation analysis. Refer to the upstream repository for its license and citation information.