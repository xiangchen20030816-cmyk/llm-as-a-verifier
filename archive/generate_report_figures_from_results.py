#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produce reproducible figures for FTEC5660 from real Terminal-Bench results.

Usage (from the repository root):
    python generate_report_figures_from_results.py --root . --out figures

Or point to an archive of uploaded results:
    python generate_report_figures_from_results.py --root "path/to/results" --out figures

Requires: matplotlib (no API key, no network, no model calls).

Sources used:
  * report_K2.json of the FULL benchmark (n_tasks=89, selection_mode=full)
  * scores.json of that benchmark (all score records, checked for completeness)
  * full_benchmark_offline_summary.json
  * offline_ablation_variants.csv
  * tasks_24_breakdown.csv
  * optional pilot K=2/K=4, 12-task, and 3-task reports

No result values are hardcoded in this script. Only the display order, styles,
labels, and identifiers of variants to plot are specified in code. Missing
required data or inconsistent statistics produce actionable errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# These are identifiers of RESULTS TO SELECT, not result numbers.
MAIN_VARIANTS = [
    ("full_ppt", "Full PPT (K=2)"),
    ("ring_K2_all3", "Ring-only (K=2)"),
    ("ring_K1_rep0_all3", "Ring-only (K=1, rep 0)"),
    ("ring_K2_output_match", "Output Match only (exploratory)"),
]

# Graphical style is editable here. Data comes only from files.
NAVY = "#17324f"
TEAL = "#087f8c"
ORANGE = "#c17826"
GRAY = "#64748b"
BORDER = "#d7e0e7"
BG = "#f6f9fc"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "figure.dpi": 170,
    "savefig.dpi": 220,
    # IMPORTANT: leave text as editable <text> in SVG, not path outlines.
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def die(message: str) -> None:
    raise ValueError(message)


def files_named(root: Path, pattern: str) -> list[Path]:
    return sorted(x for x in root.rglob(pattern) if x.is_file())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def match_report(path: Path, *, mode: str, n_tasks: int, k: int = 2) -> bool:
    try:
        d = load_json(path)
        return (isinstance(d, dict)
                and d.get("selection_mode") == mode
                and int(d.get("n_tasks", -1)) == n_tasks
                and int(d.get("K", -1)) == k
                and d.get("status") in ("complete", "completed"))
    except (ValueError, OSError, TypeError):
        return False


def choose_report(candidates: list[Path], *, mode: str, n_tasks: int,
                  k: int = 2, optional: bool = False) -> Path | None:
    matched = [p for p in candidates if match_report(p, mode=mode, n_tasks=n_tasks, k=k)]
    if not matched:
        if optional:
            return None
        die(f"No completed {mode} (n_tasks={n_tasks}, K={k}) report found. "
            "Point --root to the folder containing analysis_results, or pass --full-report.")
    # A local archive can have several snapshots of one report. Choose newest.
    return max(matched, key=lambda p: (load_json(p).get("timestamp_utc", ""), str(p)))


def choose_data_file(root: Path, explicit: str | None, pattern: str, label: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            die(f"{label} not found: {path}")
        return path
    matches = files_named(root, pattern)
    if not matches:
        die(f"{label} not found under {root}, pattern {pattern!r}")
    # Prefer the conventional output from the offline analysis directory.
    return sorted(matches, key=lambda p: ("full_benchmark_offline_analysis" not in str(p), len(p.parts), str(p)))[0]


def score_task(key: str) -> str:
    # key: criterion|task|directed_pair|repetition_index
    parts = key.split("|")
    if len(parts) != 4:
        die(f"Unrecognized score-cache key: {key!r}")
    return parts[1]


def load_all(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        die(f"Data folder does not exist: {root}")
    candidates = files_named(root, "report_K*.json")

    if args.full_report:
        full_path = Path(args.full_report).expanduser().resolve()
        if not full_path.is_file():
            die(f"Full report missing: {full_path}")
    else:
        # Do NOT accidentally use the 1-task report_K2.json as the full report.
        full_path = choose_report(candidates, mode="full", n_tasks=89, k=2)
    full = load_json(full_path)
    if full.get("selection_mode") != "full":
        die("The selected full report is NOT selection_mode=full.")

    pilot_path = choose_report(candidates, mode="pilot", n_tasks=1, k=2, optional=True)
    pilot4_path = choose_report(candidates, mode="pilot", n_tasks=1, k=4, optional=True)
    sample_path = choose_report(candidates, mode="swing_spread", n_tasks=12, k=2, optional=True)

    three_path: Path | None = None
    if args.three_report:
        three_path = Path(args.three_report).expanduser().resolve()
        if not three_path.is_file():
            die(f"Three-task report missing: {three_path}")
        if int(load_json(three_path).get("n_tasks", -1)) != 3:
            die("--three-report must point to a machine-readable report with n_tasks=3")
    else:
        # Only accept an actual 3-task report. Do NOT synthesize 3/3 from prose.
        options = [p for p in candidates if isinstance(load_json(p), dict)
                   and load_json(p).get("n_tasks") == 3]
        if options:
            three_path = max(options, key=lambda p: str(p))

    summary_path = choose_data_file(root, args.summary, "full_benchmark_offline_summary.json", "Offline summary")
    variants_path = choose_data_file(root, args.variants, "offline_ablation_variants*.csv", "Ablation CSV")
    breakdown_path = choose_data_file(root, args.breakdown, "tasks_24_breakdown*.csv", "24-task breakdown CSV")
    summary = load_json(summary_path)
    variants_list = load_csv(variants_path)
    breakdown = load_csv(breakdown_path)
    variants = {r["variant"]: r for r in variants_list}

    req = int(full["algorithmic_verification_requests"])
    n_trials = int(full["n_trials"])
    n_swing = int(full["n_swing"])
    n_tasks = int(full["n_tasks"])
    reps = int(full["K"])
    criteria = full["criteria"]
    categories = full["subset_categories"]

    if sum(int(v) for v in categories.values()) != n_tasks:
        die("Full report's category counts do not sum to n_tasks")
    if int(categories["swing"]) != n_swing:
        die("Full report's Swing category count and n_swing disagree")
    expected_ring = n_swing * n_trials * len(criteria) * reps
    if req <= expected_ring:
        die("Full score requests do not exceed the Ring requests; check the report")

    if args.scores:
        scores_path = Path(args.scores).expanduser().resolve()
        if not scores_path.is_file():
            die(f"Score cache missing: {scores_path}")
    else:
        # On the original repository, report_K2.json and scores.json share a folder.
        candidates_scores = []
        sibling = full_path.parent / "scores.json"
        if sibling.is_file():
            candidates_scores.append(sibling)
        candidates_scores.extend(files_named(root, "scores*.json"))
        scores_path = None
        for path in dict.fromkeys(candidates_scores):
            try:
                dat = load_json(path)
                if isinstance(dat, dict) and len(dat) == req:
                    scores_path = path
                    break
            except (ValueError, OSError):
                continue
        if scores_path is None:
            die(f"Cannot locate full score cache containing exactly {req} records; pass --scores FILE.")

    scores = load_json(scores_path)
    if not isinstance(scores, dict) or len(scores) != req:
        die(f"Cache records ({len(scores)}) != report requests ({req})")
    expected_tasks = {r["task"] for r in breakdown}
    got_tasks = {score_task(key) for key in scores}
    if got_tasks != expected_tasks:
        die("Score cache tasks do not match 24-task breakdown. Check --scores or --breakdown.")
    if len(breakdown) != n_swing:
        die("Breakdown row count does not match n_swing")
    for key, entry in scores.items():
        crit, task, pair, rep = key.split("|")
        if crit not in criteria or task not in got_tasks or int(rep) not in range(reps):
            die(f"Bad score key: {key}")
        if not all(name in entry and math.isfinite(float(entry[name])) for name in ("score_A", "score_B")):
            die(f"Bad numeric score values in {key}")

    correct_full = sum(int(r["full_K2_reward"]) for r in breakdown)
    correct_ring = sum(int(r["ring_K2_reward"]) for r in breakdown)
    agreement = sum(int(r["ring_full_same_trial"]) for r in breakdown)
    all_pass = int(categories["all_pass"])
    total_full = all_pass + correct_full
    total_ring = all_pass + correct_ring
    assert_rate = float(full["verifier_rate"])
    if not math.isclose(total_full / n_tasks, assert_rate, abs_tol=1e-9):
        die("Full report accuracy disagrees with actual per-task decisions")
    if correct_full != int(full["swing_correct"]):
        die("Full report and Swing breakdown disagree")

    sfull = summary["original_full_ppt"]
    sring = summary["ring_only_K2"]
    if (int(sfull["requests"]) != req
        or int(sfull["overall_success_89"]) != total_full
        or int(sring["overall_success_89"]) != total_ring
        or int(sring["requests"]) != expected_ring
        or int(summary["identical_full_ring_winners"]) != agreement):
        die("Offline summary disagrees with full report / actual breakdown / expected ring requests")
    if int(variants["ring_K2_all3"]["score_requests_needed"]) != expected_ring:
        die("Variant CSV and Ring-derived request counts disagree")
    if int(variants["ring_K2_all3"]["overall_correct_89"]) != total_ring:
        die("Variant CSV and Ring-derived correctness disagree")

    # Validate all exploratory accuracy statistics by recomputation from counts.
    for var in variants_list:
        success = int(var["overall_correct_89"])
        pct = float(var["overall_accuracy_pct"])
        if not math.isclose(100 * success / n_tasks, pct, abs_tol=0.0015):
            die(f"Ablation row {var['variant']} has inconsistent accuracy")

    used = [full_path, scores_path, summary_path, variants_path, breakdown_path]
    used += [p for p in [pilot_path, pilot4_path, sample_path, three_path] if p]
    return {
        "full": full,
        "pilot": load_json(pilot_path) if pilot_path else None,
        "pilot4": load_json(pilot4_path) if pilot4_path else None,
        "sample": load_json(sample_path) if sample_path else None,
        "three": load_json(three_path) if three_path else None,
        "summary": summary,
        "variants": variants,
        "variants_list": variants_list,
        "breakdown": breakdown,
        "scores": scores,
        "requests_full": req,
        "requests_ring": expected_ring,
        "phase_b_new": req - expected_ring,
        "correct_full": total_full,
        "correct_ring": total_ring,
        "agreement": agreement,
        "sources": [{"path": str(f), "sha256": sha256(f)} for f in dict.fromkeys(used)],
        "three_report_missing": three_path is None,
    }


def box(ax, x: float, y: float, w: float, h: float, title: str, body: str,
        *, face: str = WHITE, edge: str = BORDER, title_color: str = NAVY,
        title_size: float = 11.5, body_size: float = 10.0) -> None:
    shape = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.022",
                           linewidth=1.15, edgecolor=edge, facecolor=face)
    ax.add_patch(shape)
    ax.text(x + w / 2, y + h * .68, title, ha="center", va="center",
            fontsize=title_size, weight="bold", color=title_color)
    ax.text(x + w / 2, y + h * .32, body, ha="center", va="center",
            fontsize=body_size, color=NAVY, linespacing=1.35)


def arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = GRAY) -> None:
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>", mutation_scale=15,
                                 lw=1.6, color=color))


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  {stem}.png  +  {stem}.svg")


def figure1(d: dict, out: Path) -> None:
    full = d["full"]
    total = int(full["n_tasks"])
    trials = int(full["n_trials"])
    swing = int(full["n_swing"])
    full_req, ring_req = d["requests_full"], d["requests_ring"]
    reduction = 100 * (full_req - ring_req) / full_req

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.7), facecolor=WHITE)
    fig.suptitle("Figure 1  |  Original PPT vs. Ring-only", fontsize=17,
                 color=NAVY, fontweight="bold", y=.99)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    for idx, ax in enumerate(axes):
        is_full = idx == 0
        req = full_req if is_full else ring_req
        winner = d["correct_full"] if is_full else d["correct_ring"]
        ax.text(.50, .93, "Original PPT" if is_full else "Modified Ring-only", ha="center",
                va="center", fontsize=14, fontweight="bold", color=NAVY)
        box(ax,.16,.76,.68,.115,"Input",f"{trials} candidate trajectories",face=BG)
        arrow(ax,.5,.755,.5,.69)
        box(ax,.16,.565,.68,.115,"Ring Pass",f"{ring_req // swing} scores/task  |  {ring_req} total",
            face="#e7f5f3",edge="#9ccdc7")
        arrow(ax,.5,.56,.5,.50)
        if is_full:
            box(ax,.16,.37,.68,.115,"Pivot Round",
                f"+{d['phase_b_new'] // swing} new scores/task  |  +{d['phase_b_new']} total",
                face="#fef2e2",edge="#e6cda9")
        else:
            box(ax,.16,.37,.68,.115,"Pivot Round removed",
                f"{reduction:.0f}% fewer score requests",
                face="#f3f5f7",edge="#bfcbd6")
        arrow(ax,.5,.365,.5,.31)
        box(ax,.16,.195,.68,.115,"Winner selection",
            f"{winner}/{total} tasks  |  {100*winner/total:.2f}% success",
            face="#eaf1fa",edge="#b3cce8")
        ax.text(.5,.106,f"{req} required score records",ha="center",va="center",
                fontsize=12.0,fontweight="bold",color=TEAL if not is_full else NAVY)
    fig.text(.50,.026,
             f"Validated against {len(d['scores'])} cached score records  |  "
             f"Ring-only: post-hoc replay (not a second API experiment)",
             ha="center",va="center",fontsize=9.0,color=GRAY)
    fig.subplots_adjust(left=.06,right=.94,top=.88,bottom=.10,wspace=.13)
    save(fig,out,"figure1_ppt_vs_ringonly")


def figure2(d: dict, out: Path) -> None:
    pilot = d["pilot"]
    pilot4 = d["pilot4"]
    sample = d["sample"]
    three = d["three"]
    full = d["full"]

    def successes(r: dict) -> str:
        return f"{int(r['swing_correct'])}/{int(r['n_swing'])} Swing correct"

    if pilot:
        stage1 = f"{pilot['n_tasks']} task  |  {successes(pilot)}\nK={pilot['K']}: {pilot['algorithmic_verification_requests']} scores"
        if pilot4:
            stage1 += f"\nK={pilot4['K']}: {pilot4['algorithmic_verification_requests']} scores"
    else:
        stage1 = "No 1-task report found"

    if three:
        stage2 = (f"{three['n_tasks']} tasks  |  {successes(three)}\n"
                  f"{three['algorithmic_verification_requests']} score records")
    else:
        stage2 = "3-task pilot described\nin report narrative\nNo machine-readable\nresult file"

    if sample:
        stage3 = (f"{sample['n_tasks']} Swing tasks\n{successes(sample)}  |  "
                  f"{100*sample['swing_accuracy']:.2f}%\n"
                  f"{sample['algorithmic_verification_requests']} scores")
    else:
        stage3 = "No 12-task report found"

    stage4 = (f"{full['n_tasks']} tasks, {full['n_swing']} Swing\n"
              f"{d['correct_full']}/{full['n_tasks']} overall correct  |  "
              f"{100*full['verifier_rate']:.2f}%\n{d['requests_full']} cached scores")
    stage5 = (f"Full vs Ring: {d['agreement']}/{full['n_swing']} same winners\n"
              f"{d['requests_full']} -> {d['requests_ring']} scores\n"
              f"{d['summary']['new_api_requests']} new API calls")
    stages = [
        ("I  Single-task", "Validate complete pipeline", stage1),
        ("II  Multi-task", "Expand beyond one task", stage2),
        ("III  12-task sample", "Explore decision stability", stage3),
        ("IV  Full benchmark", "Reproduce the reported result", stage4),
        ("V  Offline ablation", "Test removal of Pivot Round", stage5),
    ]
    fig, ax = plt.subplots(figsize=(15.2, 5.5), facecolor=WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Figure 2  |  Progressive evaluation design", fontsize=17,
                 color=NAVY,fontweight="bold",y=.97)
    xstart, w, gap, y, h = .020,.172,.026,.23,.54
    for i,(title,goal,body) in enumerate(stages):
        x=xstart+i*(w+gap)
        if i==3:
            face="#e7f5f3"; edge="#9ccdc7"
        elif i==1 and three is None:
            face="#f5f6f7";edge="#c8d1d8"
        else:
            face=BG;edge=BORDER
        shape = FancyBboxPatch((x,y),w,h,
                   boxstyle="round,pad=0.008,rounding_size=.018",linewidth=1.1,
                   facecolor=face,edgecolor=edge)
        ax.add_patch(shape)
        ax.text(x+w/2,y+h*.88,title,fontsize=10.5,weight="bold",color=NAVY,ha="center",va="center")
        ax.text(x+w/2,y+h*.70,goal,fontsize=8.9,color=GRAY,ha="center",va="center")
        ax.plot([x+.02,x+w-.02],[y+h*.59,y+h*.59],color=edge,lw=1)
        ax.text(x+w/2,y+h*.35,body,fontsize=8.6,color=NAVY,ha="center",va="center",linespacing=1.45)
        if i<4:
            arrow(ax,x+w+.003,y+h/2,x+w+gap-.004,y+h/2)
    fig.text(.5,.095,
       "Pilot/sample results use Swing-only correctness; the full benchmark uses overall 89-task success. "
       "Do not interpret them as one accuracy time series.",
       fontsize=9.2,color=GRAY,ha="center")
    save(fig,out,"figure2_progressive_pipeline")


def variant_plot_values(d: dict) -> list[dict[str, Any]]:
    full=d["full"]
    n_tasks=int(full["n_tasks"])
    data=[]
    for key,label in MAIN_VARIANTS:
        if key=="full_ppt":
            success=d["correct_full"]
            requests=d["requests_full"]
            kind="Reproduced baseline"
        else:
            if key not in d["variants"]:
                die(f"Variant {key!r} is missing in the CSV")
            rec=d["variants"][key]
            success=int(rec["overall_correct_89"])
            requests=int(rec["score_requests_needed"])
            kind="Controlled cached replay" if key=="ring_K2_all3" else "Exploratory post-hoc ablation"
        data.append({
            "variant": key,
            "label": label,
            "success": success,
            "n_tasks": n_tasks,
            "accuracy_pct": round(100*success/n_tasks, 7),
            "score_requests": requests,
            "evidence_type": kind,
        })
    return data


def figure3(d: dict, out: Path) -> list[dict[str, Any]]:
    rows=variant_plot_values(d)
    fig, ax=plt.subplots(figsize=(10.0, 6.4), facecolor=WHITE)
    ax.set_title("Figure 3  |  Accuracy vs. verifier score requests",pad=20,
                 fontweight="bold",fontsize=16,color=NAVY)
    colors={"full_ppt":NAVY,"ring_K2_all3":TEAL,
            "ring_K1_rep0_all3":ORANGE,"ring_K2_output_match":"#7356a8"}
    markers={"full_ppt":"o","ring_K2_all3":"D","ring_K1_rep0_all3":"^","ring_K2_output_match":"s"}
    for row in rows:
        ax.scatter(row["score_requests"],row["accuracy_pct"],s=160,
                   marker=markers[row["variant"]],color=colors[row["variant"]],
                   label=row["label"],zorder=3)

    # Label positions are layout only, not modified data points.
    offsets = {
        "full_ppt":(-10,-32), "ring_K2_all3":(-10,13),
        "ring_K1_rep0_all3":(-10,-32), "ring_K2_output_match":(10,13),
    }
    for row in rows:
        dx,dy=offsets[row["variant"]]
        ax.annotate(f"{row['accuracy_pct']:.2f}% | {row['score_requests']} scores",
                    (row["score_requests"],row["accuracy_pct"]),
                    xytext=(dx,dy),textcoords="offset points",
                    ha="right" if dx<0 else "left",va="bottom",fontsize=9.2,
                    color=colors[row["variant"]],fontweight="medium")

    baseline=rows[0]
    ring=next(x for x in rows if x["variant"]=="ring_K2_all3")
    ax.axhline(baseline["accuracy_pct"],ls="--",lw=1.0,color="#a6b6c4",zorder=0)
    ax.annotate("",xy=(ring["score_requests"],ring["accuracy_pct"]+.16),
                xytext=(baseline["score_requests"],baseline["accuracy_pct"]+.16),
                arrowprops=dict(arrowstyle="->",color=TEAL,lw=1.8))
    save_pct=100*(baseline["score_requests"]-ring["score_requests"])/baseline["score_requests"]
    ax.text((ring["score_requests"]+baseline["score_requests"])/2,
            baseline["accuracy_pct"]+.20,f"{save_pct:.0f}% fewer score requests",
            color=TEAL,ha="center",va="bottom",fontsize=9.5)

    ax.set_xlim(70,max(r["score_requests"] for r in rows)+85)
    lo=min(r["accuracy_pct"] for r in rows)-.65
    hi=max(r["accuracy_pct"] for r in rows)+.85
    ax.set_ylim(lo,hi)
    ax.set_xlabel("Required score records (not measured API cost)")
    ax.set_ylabel("Overall task success rate (%)")
    ax.grid(True,alpha=.24,ls="--",zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="upper right",fontsize=9.2)
    fig.text(.51,.018,
             "All variants share the same 89 tasks. Only Full PPT involved fresh model scoring; "
             "other points replay existing scores.",
             ha="center",fontsize=8.9,color=GRAY)
    fig.subplots_adjust(bottom=.16,top=.87,left=.13,right=.95)
    save(fig,out,"figure3_accuracy_efficiency")
    return rows


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root",default=str(Path(__file__).resolve().parent),
                   help="Root directory of repository or an archive containing output files")
    p.add_argument("--out",default="figures",help="Directory for generated plots and provenance files")
    p.add_argument("--full-report",help="Override automatic full-89 report selection")
    p.add_argument("--scores",help="Override automatic full scores.json selection")
    p.add_argument("--summary",help="Override offline summary JSON path")
    p.add_argument("--variants",help="Override offline ablation CSV path")
    p.add_argument("--breakdown",help="Override 24-Swing breakdown CSV path")
    p.add_argument("--three-report",help="Optional machine-readable 3-task report with n_tasks=3")
    args=p.parse_args()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    data=load_all(args)
    print("Verified underlying results:")
    print(f"  Full PPT: {data['correct_full']}/{data['full']['n_tasks']} "
          f"({100*data['full']['verifier_rate']:.2f}%), {data['requests_full']} score records")
    print(f"  Ring-only: {data['correct_ring']}/{data['full']['n_tasks']} "
          f"({100*data['correct_ring']/data['full']['n_tasks']:.2f}%), "
          f"{data['requests_ring']} score records needed")
    print(f"  Chosen trial agreement: {data['agreement']}/{data['full']['n_swing']} Swing tasks")
    if data["three_report_missing"]:
        print("  NOTE: 3-task pilot JSON absent. Figure 2 labels the stage as unverified.")
    print("Generating:")
    figure1(data,out)
    figure2(data,out)
    plot_rows=figure3(data,out)

    with (out/"figure3_plot_data.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    provenance={
        "description":"Automatically generated from verified local experiment data",
        "generation_mode":"offline (zero new API calls)",
        "git_head":data["full"].get("git_head"),
        "n_tasks":data["full"]["n_tasks"],
        "n_swing":data["full"]["n_swing"],
        "full_score_records":data["requests_full"],
        "ring_score_records_required":data["requests_ring"],
        "chosen_trial_agreement":data["agreement"],
        "three_task_stage_has_machine_readable_report":not data["three_report_missing"],
        "figure3_values":plot_rows,
        "source_files_and_sha256":data["sources"],
        "important_limitation":(
            "Ring-only and criterion/repetition ablations are post-hoc replay of cached scores, "
            "not separate fresh-model experiments. Report token costs refer only to individual resumes, "
            "not necessarily full-run billing."
        ),
    }
    (out/"figure_data_sources.json").write_text(
        json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved source audit: {out/'figure_data_sources.json'}")
    print("No API was called.")


if __name__=="__main__":
    main()
