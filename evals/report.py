"""Turn evaluation results into markdown tables and keep README.md up to date."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
MODE_NAMES = {
    "vector": "Vector only",
    "bm25": "BM25 only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}


def _pct(value) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def retrieval_table(result: dict) -> str:
    k = result["meta"]["k"]
    lines = [
        f"| Retrieval mode | Hit rate@{k} | MRR@{k} | Multi-doc source recall@{k} | p50 latency |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, s in result["summary"].items():
        lines.append(
            f"| {MODE_NAMES.get(mode, mode)} | {_pct(s[f'hit_rate@{k}'])} | {s[f'mrr@{k}']:.3f} | "
            f"{_pct(s[f'multi_doc_source_recall@{k}'])} | {s['latency_ms_p50']:.0f} ms |"
        )
    meta = result["meta"]
    n = next(iter(result["summary"].values()))["questions"] if result["summary"] else 0
    lines.append("")
    lines.append(
        f"_{n} labelled questions · embeddings `{meta['embedding_model']}` · reranker `{meta['reranker_model']}` · "
        f"chunks {meta['chunk_size']}/{meta['chunk_overlap']} chars · run {meta['date']}_"
    )
    return "\n".join(lines) + "\n"


def answers_table(runs: dict) -> str:
    lines = [
        "| Configuration | Correctness | Fully correct | Faithfulness | Answers with citations | "
        "Abstains when it should | False abstentions | Retried | p50 latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    meta = {}
    for label, result in runs.items():
        s, meta = result["summary"], result["meta"]
        lines.append(
            f"| {label} | {_pct(s['correctness'])} | {_pct(s['fully_correct'])} | {_pct(s['faithfulness'])} | "
            f"{_pct(s['answers_with_citations'])} | {_pct(s['abstention_accuracy'])} | "
            f"{_pct(s['false_abstention_rate'])} | "
            f"{_pct(s['retried'])} | {s['latency_s_p50']:.1f} s |"
        )
    if meta:
        lines.append("")
        lines.append(
            f"_{next(iter(runs.values()))['summary']['questions']} questions (incl. unanswerable ones) · "
            f"LLM `{meta['llm']}` · judge `{meta['judge']}` · run {meta['date']}_"
        )
    return "\n".join(lines) + "\n"


def update_readme(section: str, table: str, readme: Path = README) -> bool:
    """Replace the text between <!-- EVAL:{section}:START --> and <!-- EVAL:{section}:END -->."""
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- EVAL:{section}:START -->\n).*?(<!-- EVAL:{section}:END -->)", re.DOTALL)
    if not pattern.search(text):
        print(f"README has no EVAL:{section} markers; skipped.")
        return False
    readme.write_text(pattern.sub(lambda m: m.group(1) + table + m.group(2), text), encoding="utf-8")
    print(f"Updated the {section.lower()} table in {readme.name}.")
    return True
