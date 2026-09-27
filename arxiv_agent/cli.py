import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_settings
from .deps import Deps, build_deps
from .graph import build_ingest_graph
from .qa import build_qa_graph

EXIT_WORDS = {"", "q", "quit", "exit", ":q"}


def _print_sources(result: dict) -> None:
    for c in result.get("citations", []):
        print(f"   [{c['n']}] {c['section']} (p. {c['page']})")


def ask_once(deps: Deps, qa_graph, session: dict, question: str) -> dict:
    result = qa_graph.invoke(
        {
            "arxiv_id": session["paper"]["arxiv_id"],
            "title": session["paper"]["title"],
            "chunks": session["chunks"],
            "history": session.get("history", []),
            "question": question,
        }
    )
    turn = {
        "question": question,
        "answer": result["answer"],
        "status": result.get("status"),
        "citations": result.get("citations", []),
        "rejected_drafts": result.get("rejected", []),
        "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    session.setdefault("history", []).append(turn)
    deps.sessions.save(session)
    return result


def qa_loop(deps: Deps, arxiv_id: str) -> None:
    session = deps.sessions.load(arxiv_id)
    qa_graph = build_qa_graph(deps)
    print(f"\nAsk about \"{session['paper']['title']}\". Empty line or 'quit' to stop.")
    suggestions = session.get("briefing", {}).get("follow_up_questions", [])[:3]
    if suggestions:
        print("Ideas: " + " | ".join(suggestions))
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in EXIT_WORDS:
            return
        result = ask_once(deps, qa_graph, session, question)
        print(f"\nagent> {result['answer']}")
        _print_sources(result)


def cmd_brief(deps: Deps, args) -> int:
    graph = build_ingest_graph(deps)
    state = {"user_input": args.query, "refresh": args.refresh, "warnings": []}
    final: dict = {}
    for update in graph.stream(state, stream_mode="updates"):
        for node, change in update.items():
            print(f"  -> {node}")
            if change:
                final.update({k: v for k, v in change.items() if k != "warnings"})
                if change.get("warnings"):
                    for warning in change["warnings"]:
                        print(f"     ! {warning}")
    if final.get("error"):
        print(f"\nStopped: {final['error']}", file=sys.stderr)
        return 1
    if final.get("cached"):
        print("  (using the saved briefing and index; pass --refresh to rebuild)")
    print("\n" + final["briefing_markdown"])
    if final.get("output_dir"):
        print(f"Saved to {final['output_dir']}/briefing.md and briefing.json")
    if not args.no_qa:
        qa_loop(deps, final["arxiv_id"])
    return 0


def cmd_ask(deps: Deps, args) -> int:
    try:
        session = deps.sessions.load(args.arxiv_id)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.question:
        result = ask_once(deps, build_qa_graph(deps), session, args.question)
        print(result["answer"])
        _print_sources(result)
        return 0
    qa_loop(deps, args.arxiv_id)
    return 0


def cmd_list(deps: Deps, args) -> int:
    rows = deps.sessions.list()
    if not rows:
        print("No saved papers yet.")
    for row in rows:
        print(f"{row['arxiv_id']:<14} {row['turns']:>3} Q&A  {row['title']}")
    return 0


def cmd_export(deps: Deps, args) -> int:
    try:
        session = deps.sessions.load(args.arxiv_id)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    lines = [session["briefing_markdown"].rstrip(), "", "## Q&A transcript", ""]
    history = session.get("history", [])
    if not history:
        lines.append("No questions asked yet.")
    for turn in history:
        lines += [f"**Q:** {turn['question']}", "", f"**A:** {turn['answer']}", ""]
        if turn.get("citations"):
            sources = "; ".join(f"[{c['n']}] {c['section']}, p. {c['page']}" for c in turn["citations"])
            lines += [f"_Sources: {sources}_", ""]
        lines += [f"_Status: {turn.get('status')}_", ""]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.arxiv_id.replace('/', '_')}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")
    return 0


def cmd_graph(deps: Deps, args) -> int:
    print(build_ingest_graph(deps).get_graph().draw_mermaid())
    print(build_qa_graph(deps).get_graph().draw_mermaid())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arxiv_agent", description="arXiv paper digest and QA agent")
    sub = parser.add_subparsers(dest="command", required=True)

    brief = sub.add_parser("brief", help="build a briefing from a topic or an arXiv ID, then open QA")
    brief.add_argument("query", help='a topic in quotes, an arXiv ID, or an arXiv URL')
    brief.add_argument("--refresh", action="store_true", help="ignore the saved session and rebuild")
    brief.add_argument("--no-qa", action="store_true", help="print the briefing and exit")
    brief.set_defaults(func=cmd_brief)

    ask = sub.add_parser("ask", help="ask questions about a paper you already briefed")
    ask.add_argument("arxiv_id")
    ask.add_argument("-q", "--question", help="ask one question and exit")
    ask.set_defaults(func=cmd_ask)

    sub.add_parser("list", help="show papers with saved sessions").set_defaults(func=cmd_list)
    export = sub.add_parser("export", help="write the briefing plus the Q&A transcript to a markdown file")
    export.add_argument("arxiv_id")
    export.add_argument("--out", default="examples")
    export.set_defaults(func=cmd_export)

    sub.add_parser("graph", help="print both graphs as Mermaid").set_defaults(func=cmd_graph)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    deps = build_deps(load_settings())
    return args.func(deps, args)
