"""
Benchmarks skill-status classification (HAS / WEAK / LEARNING / NONE) on
tests/fixtures/skill_status_sentences.jsonl.

Methods:
  A  old matcher regexes (HARD -> NONE, NON_EVIDENCE -> LEARNING, else HAS), copied
     here because the matcher no longer has them
  B  new rules only (app.services.skill_status.classify_rules)
  C  local LLM only, every item, raw labels (missing label -> HAS)
  D  new hybrid: rule label if a rule matched, else the LLM label (with the NONE guard)
  D* D without the NONE guard, from the same LLM output (ablation)

Usage (from backend/):
  python scripts/benchmark_skill_status.py              # A, B, C, D x3 LLM runs
  python scripts/benchmark_skill_status.py --no-llm     # A and B only
  python scripts/benchmark_skill_status.py --split seen # restrict to one split
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app  # noqa: E402,F401  (sets USE_TF=0)
from app.core.config import settings  # noqa: E402
from app.services import skill_status as ss  # noqa: E402

FIXTURE = BACKEND_DIR / "tests" / "fixtures" / "skill_status_sentences.jsonl"
LABELS = list(ss.LABELS)

# The matcher's regexes before this change (hybrid_matcher.py), for method A.
OLD_HARD = re.compile(
    r"(?:don't\s+know|do\s+not\s+know|no\s+experience|not\s+experienced|never\s+used|"
    r"haven't\s+used|not\s+familiar|don't\s+have|do\s+not\s+have|no\s+knowledge)",
    re.IGNORECASE,
)
OLD_NON_EVIDENCE = re.compile(
    r"(?:don't\s+know|do\s+not\s+know|no\s+experience|not\s+experienced|never\s+used|"
    r"haven't\s+used|not\s+familiar|don't\s+have|do\s+not\s+have|no\s+knowledge|"
    r"currently\s+learning|want(?:ing)?\s+to\s+learn|wish(?:ing)?\s+to\s+learn|"
    r"trying\s+to\s+learn|studying)",
    re.IGNORECASE,
)


def load_items(split: str | None) -> list[dict]:
    items = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item for item in items if split in (None, item["split"])]


def method_a(item: dict) -> str:
    if OLD_HARD.search(item["sentence"]):
        return ss.NONE
    if OLD_NON_EVIDENCE.search(item["sentence"]):
        return ss.LEARNING
    return ss.HAS


def method_b(item: dict) -> ss.SkillAssessment:
    return ss.classify_rules(item["skill"], item["sentence"])


GROUPS = {
    "seen": lambda item: item["split"] == "seen",
    "held-out": lambda item: item["split"] == "heldout",
    "  prior held-out": lambda item: item["split"] == "heldout" and item["origin"] == "prior",
    "  new held-out": lambda item: item["split"] == "heldout" and item["origin"] == "new",
    "all": lambda item: True,
}


def accuracy(items: list[dict], predictions: list[str], group) -> tuple[int, int]:
    pairs = [(item, pred) for item, pred in zip(items, predictions) if group(item)]
    return sum(item["label"] == pred for item, pred in pairs), len(pairs)


def fmt(ok: int, total: int) -> str:
    return f"{ok / total:4.0%} ({ok}/{total})" if total else "   -"


def has_rejected(items: list[dict], predictions: list[str]) -> int:
    return sum(item["label"] == ss.HAS and pred == ss.NONE for item, pred in zip(items, predictions))


def print_table(items: list[dict], methods: dict[str, list[list[str]]]) -> None:
    groups = [name for name, group in GROUPS.items() if any(group(item) for item in items)]
    print(f"{'method':24}" + "".join(f"{name:>20}" for name in groups) + f"{'HAS->NONE':>12}")
    for name, runs in methods.items():
        cells = []
        for group_name in groups:
            scores = [accuracy(items, run, GROUPS[group_name]) for run in runs]
            ok, total = scores[0]
            if len(runs) == 1:
                cells.append(fmt(ok, total))
            else:
                values = [s[0] / s[1] for s in scores if s[1]]
                cells.append(f"{statistics.mean(values):4.0%} [{min(values):.0%}-{max(values):.0%}]")
        rejected = [has_rejected(items, run) for run in runs]
        rejected_cell = str(rejected[0]) if len(runs) == 1 else f"{statistics.mean(rejected):.1f} [{min(rejected)}-{max(rejected)}]"
        print(f"{name:24}" + "".join(f"{cell:>20}" for cell in cells) + f"{rejected_cell:>12}")


def print_details(name: str, items: list[dict], predictions: list[str]) -> None:
    print(f"\n--- {name}: per-label recall and confusion matrix (rows = gold, cols = predicted) ---")
    confusion = Counter((item["label"], pred) for item, pred in zip(items, predictions))
    print(f"{'gold':10}" + "".join(f"{label:>10}" for label in LABELS) + f"{'recall':>10}")
    for gold in LABELS:
        total = sum(confusion[(gold, pred)] for pred in LABELS)
        recall = confusion[(gold, gold)] / total if total else 0.0
        print(f"{gold:10}" + "".join(f"{confusion[(gold, pred)]:>10}" for pred in LABELS) + f"{recall:>10.0%}")


def print_errors(name: str, items: list[dict], predictions: list[str], split: str) -> None:
    errors = [(item, pred) for item, pred in zip(items, predictions) if item["split"] == split and item["label"] != pred]
    print(f"\n--- {name}: {split} errors ({len(errors)}) ---")
    for item, pred in errors:
        print(f"  gold={item['label']:8} pred={pred:8} [{item['skill']}] {item['sentence']}")


async def run_llm(items: list[tuple[str, str]]) -> tuple[dict[int, str], float]:
    return await ss.llm_classify(items, timeout=300)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["seen", "heldout"], help="Only evaluate one split.")
    parser.add_argument("--no-llm", action="store_true", help="Skip methods C and D.")
    parser.add_argument("--llm-runs", type=int, default=3)
    parser.add_argument("--errors", action="store_true", help="List misclassified items.")
    args = parser.parse_args()

    items = load_items(args.split)
    print(f"{len(items)} items from {FIXTURE.name}: {dict(Counter(item['split'] for item in items))}")
    a_pred = [method_a(item) for item in items]
    b_assess = [method_b(item) for item in items]
    b_pred = [assessment.status for assessment in b_assess]
    unresolved = [index for index, assessment in enumerate(b_assess) if not assessment.matched]
    print(f"rules resolved {len(items) - len(unresolved)}/{len(items)} items; {len(unresolved)} go to the LLM in method D")
    methods: dict[str, list[list[str]]] = {"A old regexes": [a_pred], "B new rules": [b_pred]}

    c_runs: list[list[str]] = []
    d_runs: list[list[str]] = []
    d_raw_runs: list[list[str]] = []
    latencies: dict[str, list[float]] = {"C": [], "D": []}
    if not args.no_llm:
        try:
            await ss.llm_classify([("python", "Wrote Python services.")], timeout=120)  # load the model
        except ss.LLMUnavailable as exc:
            print(f"\nLLM unreachable ({exc}); skipping C and D. Start Ollama with {settings.ollama_parsing_model}.")
            args.no_llm = True
    if not args.no_llm:
        all_items = [(item["skill"], item["sentence"]) for item in items]
        d_items = [all_items[index] for index in unresolved]
        for run in range(args.llm_runs):
            labels, seconds = await run_llm(all_items)
            c_runs.append([labels.get(index + 1, ss.HAS) for index in range(len(items))])
            latencies["C"].append(seconds / len(all_items))

            labels, seconds = await run_llm(d_items)
            d_pred, d_raw = list(b_pred), list(b_pred)
            for position, index in enumerate(unresolved, start=1):
                label = labels.get(position)
                if label is None:
                    continue
                d_raw[index] = label
                if ss.accept_llm_label(label, items[index]["sentence"]):
                    d_pred[index] = label
            d_runs.append(d_pred)
            d_raw_runs.append(d_raw)
            latencies["D"].append(seconds / max(1, len(d_items)))
            print(f"LLM run {run + 1}/{args.llm_runs} done")
        methods["C LLM only"] = c_runs
        methods["D rules + LLM"] = d_runs
        methods["D* no NONE guard"] = d_raw_runs

    print("\n=== Accuracy (LLM methods: mean [min-max] over runs) ===")
    print_table(items, methods)
    for name, runs in methods.items():
        print_details(f"{name} (run 1)", items, runs[0])
    if latencies["C"]:
        print("\n=== LLM latency, seconds per sentence (batched) ===")
        for name, values in latencies.items():
            print(f"{name}: mean {statistics.mean(values):.2f}  min {min(values):.2f}  max {max(values):.2f}")
        print(f"model: {settings.ollama_parsing_model}, batch size {settings.skill_status_llm_batch_size}")
    if args.errors:
        for name, runs in methods.items():
            for split in ("seen", "heldout"):
                print_errors(name, items, runs[0], split)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
