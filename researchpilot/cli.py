"""Command-line interface.

    researchpilot ingest --file data/sample_sources.txt
    researchpilot ingest paper.pdf https://example.com/article
    researchpilot ask "your question" --trace
    researchpilot sources
    researchpilot delete <source>
    researchpilot warmup          # download the local embedding + reranker models

(`python -m researchpilot.cli ...` works too.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import RETRIEVAL_MODES


def read_sources_file(path: str) -> list[str]:
    """One source per line; blank lines, '#' comments and text after the source are ignored."""
    sources = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            sources.append(line.split()[0])
    return sources


def cmd_ingest(args) -> int:
    from .ingestion.pipeline import ingest
    from .retrieval.hybrid import get_retriever

    sources = list(args.sources)
    if args.file:
        sources += read_sources_file(args.file)
    if not sources:
        raise SystemExit("Give sources as arguments or with --file.")
    if args.if_empty and get_retriever().store.count() > 0:
        print("Index already has documents; skipping (--if-empty).")
        return 0

    print(f"Ingesting {len(sources)} source(s)...")
    report = ingest(sources, on_progress=lambda s, i, n: print(f"  [{i}/{n}] {s}", flush=True))
    for item in report["ingested"]:
        note = "" if item.get("english", True) else "  [not English: answers may be less reliable]"
        print(f"  OK    {item['title'][:70]}  ({item['chunks']} chunks){note}")
    for item in report["failed"]:
        print(f"  FAIL  {item['source']}: {item['error']}")
    return 1 if report["failed"] and not report["ingested"] else 0


def cmd_ask(args) -> int:
    from .agent.graph import ask

    result = ask(args.question, mode=args.mode, max_rounds=args.rounds)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    print("\n" + result["answer"] + "\n")
    if result["citations"]:
        print("Sources:")
        for c in result["citations"]:
            page = f", p. {c['page']}" if c.get("page") else ""
            print(f"  [{c['n']}] {c['title'][:70]}{page}  <{c['source']}>")
    if args.trace:
        print("\nTrace:")
        for step in result["trace"]:
            if step.get("skipped"):
                details = f"grader skipped (no rounds left), seconds={step.get('seconds')}"
            else:
                details = ", ".join(f"{k}={v}" for k, v in step.items() if k != "step")
            print(f"  {step['step']}: {details}")
    print(f"\nTotal time: {result['seconds']}s")
    return 0


def cmd_sources(_args) -> int:
    from .retrieval.hybrid import get_retriever

    sources = get_retriever().store.sources()
    if not sources:
        print("Nothing ingested yet.")
    for s in sources:
        print(f"  {s['title'][:70]}  ({s['chunks']} chunks)  <{s['source']}>")
    return 0


def cmd_delete(args) -> int:
    from .ingestion.pipeline import delete_source

    deleted = delete_source(args.source)
    print(f"Deleted {deleted} chunk(s) for {args.source}" if deleted else f"No chunks found for {args.source}")
    return 0 if deleted else 1


def cmd_warmup(_args) -> int:
    from .retrieval import embeddings, reranker

    print("Downloading / loading the embedding model...")
    embeddings.warmup()
    print("Downloading / loading the reranker model...")
    reranker.warmup()
    print("Models ready.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchpilot", description="Agentic research assistant over your documents")
    parser.add_argument("-v", "--verbose", action="store_true", help="show info-level logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest PDFs, text files or URLs")
    p.add_argument("sources", nargs="*")
    p.add_argument("--file", help="text file with one source per line")
    p.add_argument("--if-empty", action="store_true", help="only ingest if the index is empty")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--mode", choices=RETRIEVAL_MODES, help="retrieval mode (default from settings)")
    p.add_argument("--rounds", type=int, help="max retrieval rounds (default from settings)")
    p.add_argument("--trace", action="store_true", help="show the agent's steps")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("sources", help="list ingested sources")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("delete", help="remove a source from the index")
    p.add_argument("source", help="the source exactly as shown by `sources`")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("warmup", help="download the local models ahead of time")
    p.set_defaults(func=cmd_warmup)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # show a clean message instead of a traceback
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
