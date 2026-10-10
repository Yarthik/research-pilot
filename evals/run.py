"""Evaluation harness.

python -m evals retrieval                    # retrieval quality per mode (free: no LLM calls)
python -m evals answers --limit 10           # end-to-end answer quality (uses the LLM + judge)
python -m evals answers --compare-retry      # ablation: 1 retrieval round vs 2
python -m evals answers --rounds 1           # one config; saved after every question, re-run to resume
python -m evals rejudge --rounds 1           # re-grade saved answers after a judge-prompt change
python -m evals all --update-readme          # everything, and refresh the tables in README.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from researchpilot.config import RETRIEVAL_MODES, get_settings

from . import metrics
from .report import answers_table, retrieval_table, update_readme

log = logging.getLogger("evals")
HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
RESULTS = HERE / "results"


# --------------------------------------------------------------------------- data
def load_dataset(path: Path = DATASET) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in the dataset")
    for row in rows:
        if row["type"] != "unanswerable" and not (row["sources"] and row["keywords"]):
            raise ValueError(f"question {row['id']} needs sources and keywords")
    return rows


def ensure_corpus(dataset: list[dict], ingest_missing: bool = True) -> None:
    """Make sure every source referenced by the dataset is in the index."""
    from researchpilot.ingestion.pipeline import ingest
    from researchpilot.retrieval.hybrid import get_retriever

    needed = sorted({s for row in dataset for s in row["sources"]})
    present = [s["source"] for s in get_retriever().store.sources()]
    missing = [s for s in needed if not any(metrics.same_source(p, s) for p in present)]
    if not missing:
        return
    if not ingest_missing:
        raise SystemExit(f"Missing sources in the index: {missing}. Ingest them first or drop --no-ingest.")
    print(f"Ingesting {len(missing)} source(s) the dataset needs...")
    report = ingest(missing)
    if report["failed"]:
        raise SystemExit(f"Could not ingest: {report['failed']}")


def _save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _meta() -> dict:
    s = get_settings()
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "llm": f"{s.llm_provider}/{s.llm_model}",
        "embedding_model": s.embedding_model,
        "reranker_model": s.reranker_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
    }


# --------------------------------------------------------------------------- retrieval
def run_retrieval(dataset: list[dict], modes: list[str], k: int, retriever=None) -> dict:
    if retriever is None:
        from researchpilot.retrieval.hybrid import get_retriever

        retriever = get_retriever()
    questions = [row for row in dataset if row["type"] != "unanswerable"]
    summary, details = {}, {}
    for mode in modes:
        rows = []
        for row in questions:
            start = time.perf_counter()
            results = retriever.retrieve(row["question"], mode=mode, final_k=k)
            latency_ms = (time.perf_counter() - start) * 1000
            hits = [metrics.Hit(chunk.source, chunk.text) for chunk, _ in results]
            scores = metrics.retrieval_scores(hits, row["sources"], row["keywords"])
            rows.append({"id": row["id"], "type": row["type"], "latency_ms": round(latency_ms, 1), **scores})
        multi = [r for r in rows if r["type"] == "multi"]
        summary[mode] = {
            "questions": len(rows),
            f"hit_rate@{k}": metrics.mean([r["hit"] for r in rows]),
            f"mrr@{k}": metrics.mean([r["reciprocal_rank"] for r in rows]),
            f"multi_doc_source_recall@{k}": metrics.mean([r["source_recall"] for r in multi]),
            "latency_ms_p50": metrics.percentile([r["latency_ms"] for r in rows], 50),
        }
        details[mode] = rows
    never_found = [row["id"] for i, row in enumerate(questions) if all(details[m][i]["hit"] == 0 for m in modes)]
    return {"meta": {**_meta(), "k": k}, "summary": summary, "never_found": never_found, "details": details}


# --------------------------------------------------------------------------- answers
class QuotaExhausted(RuntimeError):
    """An LLM rate limit or quota stopped the run. Progress is saved; re-run to resume."""


_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "too many requests", "quota")


def is_rate_limit(error: object) -> bool:
    """True for rate-limit / quota errors (an exception or an error string from the agent's trace)."""
    if getattr(error, "status_code", None) == 429 or type(error).__name__ == "RateLimitError":
        return True
    return any(marker in str(error).lower() for marker in _RATE_LIMIT_MARKERS)


def _load_checkpoint(path: Path | None, meta: dict) -> tuple[list[dict], str | None]:
    """Rows already evaluated by an earlier run with the same configuration, and that run's date."""
    if path is None or not path.exists():
        return [], None
    saved = json.loads(path.read_text(encoding="utf-8"))
    keys = ("llm", "mode", "max_rounds", "judge", "judge_prompt")
    if any(saved["meta"].get(k) != meta.get(k) for k in keys):
        theirs = {k: saved["meta"].get(k) for k in keys}
        raise SystemExit(f"{path} was made with a different configuration {theirs}; delete it to start over.")
    return saved["details"], saved["meta"].get("date")


def run_answers(
    dataset: list[dict],
    mode: str | None = None,
    max_rounds: int | None = None,
    delay: float = 0.0,
    ask_fn=None,
    judge=None,
    checkpoint: Path | None = None,
) -> dict:
    """Run the agent and the judge on each question.

    With `checkpoint`, results are saved after every question and questions already in the file are
    skipped, so an interrupted run resumes where it stopped. A rate-limit or quota error (raised, or
    swallowed by an agent step and left in its trace) saves progress and raises QuotaExhausted
    without recording the question, so quota trouble never shows up as a wrong answer.
    """
    from .judge import CORRECTNESS_PROMPT_VERSION, judge_correctness, judge_faithfulness

    if ask_fn is None:
        from researchpilot.agent.graph import ask as ask_fn
    judge_name = "custom"
    if judge is None:
        from .judge import get_judge

        judge, judge_name = get_judge()

    meta = {
        **_meta(),
        "mode": mode or get_settings().retrieval_mode,
        "max_rounds": max_rounds or get_settings().max_retrieval_rounds,
        "judge": judge_name,
        "judge_prompt": CORRECTNESS_PROMPT_VERSION,
    }
    wanted = {row["id"] for row in dataset}
    saved_rows, saved_date = _load_checkpoint(checkpoint, meta)
    if saved_date:
        meta["date"] = saved_date  # a resumed run keeps the date it started on
    rows = [r for r in saved_rows if r["id"] in wanted]
    done = {r["id"] for r in rows}
    if done:
        print(f"  resuming: {len(done)} of {len(dataset)} questions already in {checkpoint.name}")

    order = {row["id"]: i for i, row in enumerate(dataset)}

    def snapshot() -> dict:
        ordered = sorted(rows, key=lambda r: order[r["id"]])
        complete = len(rows) == len(dataset)
        return {"meta": meta, "complete": complete, "summary": summarize_answers(ordered), "details": ordered}

    def save() -> None:
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(snapshot(), indent=2, default=str), encoding="utf-8")

    def stop(row: dict, error: object) -> QuotaExhausted:
        save()
        done_n = f"{len(rows)} of {len(dataset)} done"
        return QuotaExhausted(f"rate limit / quota reached at {row['id']} ({done_n}): {str(error)[:200]}")

    for index, row in enumerate(dataset, start=1):
        if row["id"] in done:
            continue
        print(f"  [{index}/{len(dataset)}] {row['id']}: {row['question'][:70]}", flush=True)
        try:
            result = ask_fn(row["question"], mode=mode, max_rounds=max_rounds)
        except Exception as exc:
            if is_rate_limit(exc):
                raise stop(row, exc) from exc
            log.error("agent failed on %s: %s", row["id"], exc)
            rows.append({"id": row["id"], "type": row["type"], "error": str(exc)})
            save()
            continue
        # Agent steps degrade instead of raising (e.g. a plain-text fallback), so check their errors too.
        limited = [step["error"] for step in result.get("trace", []) if is_rate_limit(step.get("error") or "")]
        if limited:
            raise stop(row, limited[0])

        record = {
            "id": row["id"],
            "type": row["type"],
            "answer": result["answer"],
            "insufficient": result["insufficient"],
            "cited": [c["n"] for c in result["citations"]],
            "invalid_citations": result.get("invalid_citations", []),
            "rounds": result.get("round") or 0,
            "seconds": result.get("seconds", 0.0),
        }
        if row["type"] == "unanswerable":
            record["abstained"] = metrics.abstained(result["answer"], result["insufficient"])
        else:
            record["abstained"] = metrics.declined(result["answer"], result["insufficient"], record["cited"])
            try:
                record["correctness"] = judge_correctness(judge, row["question"], row["reference"], result["answer"])
                if result["citations"]:
                    record["faithfulness"] = judge_faithfulness(
                        judge, row["question"], result["answer"], result["citations"]
                    )
                elif not record["abstained"]:
                    record["faithfulness"] = {"score": 0.0, "claims": None, "unsupported": ["answer has no citations"]}
            except Exception as exc:
                if is_rate_limit(exc):
                    raise stop(row, exc) from exc
                log.error("judge failed on %s: %s", row["id"], exc)
                record["judge_error"] = str(exc)
        rows.append(record)
        save()
        if delay:
            time.sleep(delay)

    return snapshot()


def rejudge_correctness(path: Path, dataset: list[dict], judge=None, delay: float = 0.0) -> dict:
    """Re-grade the correctness of saved answers with the current judge prompt; the agent is not re-run.

    Also recomputes abstention with the current definition. Verdicts already made with the current
    prompt are kept, so an interrupted re-grade resumes. Faithfulness is left as it is.
    """
    from .judge import CORRECTNESS_PROMPT_VERSION, get_judge, judge_correctness

    saved = json.loads(path.read_text(encoding="utf-8"))
    if judge is None:
        judge, judge_name = get_judge()
        if judge_name != saved["meta"]["judge"]:
            raise SystemExit(f"{path} was judged by {saved['meta']['judge']}, not {judge_name}; set JUDGE_MODEL.")
    by_id = {row["id"]: row for row in dataset}

    def save() -> None:
        saved["summary"] = summarize_answers(saved["details"])
        path.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")

    for record in saved["details"]:
        if record["type"] == "unanswerable" or "error" in record:
            continue
        row = by_id[record["id"]]
        record["abstained"] = metrics.declined(record["answer"], record["insufficient"], record["cited"])
        if record.get("correctness", {}).get("prompt_version") == CORRECTNESS_PROMPT_VERSION:
            continue
        print(f"  re-judging {record['id']}", flush=True)
        try:
            record["correctness"] = judge_correctness(judge, row["question"], row["reference"], record["answer"])
        except Exception as exc:
            if is_rate_limit(exc):
                save()
                message = f"rate limit / quota reached while re-judging {record['id']}: {str(exc)[:200]}"
                raise QuotaExhausted(message) from exc
            raise
        record.pop("judge_error", None)
        save()
        if delay:
            time.sleep(delay)
    saved["meta"]["judge_prompt"] = CORRECTNESS_PROMPT_VERSION
    save()
    return saved


def summarize_answers(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["type"] != "unanswerable"]
    unanswerable = [r for r in ok if r["type"] == "unanswerable"]
    judged = [r for r in answerable if "correctness" in r]
    faith = [r["faithfulness"]["score"] for r in answerable if r.get("faithfulness", {}).get("score") is not None]
    answered = [r for r in answerable if not r["abstained"]]
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "correctness": metrics.mean([r["correctness"]["score"] for r in judged]),
        "fully_correct": metrics.mean([1.0 if r["correctness"]["verdict"] == "correct" else 0.0 for r in judged]),
        "faithfulness": metrics.mean(faith),
        "answers_with_citations": metrics.mean([1.0 if r["cited"] else 0.0 for r in answered]),
        "invalid_citation_rate": metrics.mean([1.0 if r["invalid_citations"] else 0.0 for r in ok]),
        "false_abstention_rate": metrics.mean([1.0 if r["abstained"] else 0.0 for r in answerable]),
        "abstention_accuracy": metrics.mean([1.0 if r["abstained"] else 0.0 for r in unanswerable]),
        "retried": metrics.mean([1.0 if r["rounds"] > 1 else 0.0 for r in ok]),
        "latency_s_p50": metrics.percentile([r["seconds"] for r in ok], 50),
        "latency_s_p95": metrics.percentile([r["seconds"] for r in ok], 95),
    }


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals", description="ResearchPilot evaluation harness")
    parser.add_argument("command", choices=["retrieval", "answers", "rejudge", "all"])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES))
    parser.add_argument("--k", type=int, default=5, help="cut-off for retrieval metrics")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, help="retrieval mode for the answer eval")
    parser.add_argument("--rounds", type=int, help="max retrieval rounds for the answer eval")
    parser.add_argument("--compare-retry", action="store_true", help="answer eval with 1 round vs 2 rounds")
    parser.add_argument("--limit", type=int, help="only the first N questions (answer eval)")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    parser.add_argument("--no-ingest", action="store_true", help="fail instead of ingesting missing sources")
    parser.add_argument("--update-readme", action="store_true", help="write the result tables into README.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    dataset = load_dataset(args.dataset)
    ensure_corpus(dataset, ingest_missing=not args.no_ingest)

    if args.command in ("retrieval", "all"):
        print(
            f"\nRetrieval eval: {sum(r['type'] != 'unanswerable' for r in dataset)} questions x {len(args.modes)} modes"
        )
        result = run_retrieval(dataset, args.modes, args.k)
        _save("retrieval.json", result)
        table = retrieval_table(result)
        (RESULTS / "retrieval.md").write_text(table, encoding="utf-8")
        print("\n" + table)
        if result["never_found"]:
            print(f"Questions no mode found (check their keywords): {', '.join(result['never_found'])}")
        if args.update_readme:
            update_readme("RETRIEVAL", table)

    if args.command == "rejudge":
        settings = get_settings()
        mode, n_rounds = args.mode or settings.retrieval_mode, args.rounds or settings.max_retrieval_rounds
        checkpoint = RESULTS / f"answers_{mode}_{n_rounds}round.json"
        print(f"\nRe-judging correctness in {checkpoint.name}")
        try:
            rejudge_correctness(checkpoint, dataset, delay=args.delay)
        except QuotaExhausted as exc:
            print(f"\nStopped: {exc}\nRun the same command again to resume.")
            return 2
        return 0

    if args.command in ("answers", "all"):
        subset = dataset[: args.limit] if args.limit else dataset
        runs = {}
        configs = (
            [("1 round (no retry)", 1), ("2 rounds (self-correcting)", 2)]
            if args.compare_retry
            else [(None, args.rounds)]
        )
        settings = get_settings()
        for label, rounds in configs:
            print(f"\nAnswer eval{f' - {label}' if label else ''}: {len(subset)} questions")
            mode, n_rounds = args.mode or settings.retrieval_mode, rounds or settings.max_retrieval_rounds
            checkpoint = RESULTS / f"answers_{mode}_{n_rounds}round.json"
            try:
                result = run_answers(subset, mode=mode, max_rounds=n_rounds, delay=args.delay, checkpoint=checkpoint)
            except QuotaExhausted as exc:
                print(f"\nStopped: {exc}\nProgress is saved in {checkpoint}; run the same command again to resume.")
                return 2
            key = label or f"{mode}, {n_rounds} round(s)"
            runs[key] = result
        _save("answers.json", runs)
        table = answers_table(runs)
        (RESULTS / "answers.md").write_text(table, encoding="utf-8")
        print("\n" + table)
        if args.update_readme:
            update_readme("ANSWERS", table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
