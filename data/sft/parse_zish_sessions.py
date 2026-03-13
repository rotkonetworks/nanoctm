"""
Parse zish agent sessions into SFT training data.

Each session becomes a conversation: user/assistant turns with tool use context.
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime


MIN_TURNS = 1  # minimum user+assistant turns to keep
MIN_ASSISTANT_CHARS = 20  # skip sessions where assistant barely says anything


def sanitize_text(text):
    """Redact sensitive values."""
    # IPs
    text = re.sub(r'(\d{1,3}\.\d{1,3}\.)\d{1,3}\.\d{1,3}', r'\1x.x', text)
    # SSH key paths
    text = re.sub(r'-i\s+\S*\.ssh\S*', '-i <key>', text)
    # Long hex strings (hashes, tokens)
    text = re.sub(r'[0-9a-f]{40,}', '<hash>', text)
    # Penumbra addresses
    text = re.sub(r'(penumbra1[a-z0-9]{12})[a-z0-9]+', r'\1...', text)
    return text


def parse_session(session_dir):
    """Parse a single session directory into training text."""
    conv_file = session_dir / "conversation.jsonl"
    meta_file = session_dir / "meta.json"

    if not conv_file.exists() or conv_file.stat().st_size == 0:
        return None

    # Read meta
    meta = {}
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except:
            pass

    # Read conversation
    messages = []
    with open(conv_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
            except:
                continue

    if not messages:
        return None

    # Build conversation text
    parts = []
    user_turns = 0
    asst_chars = 0
    tool_context = []

    for msg in messages:
        t = msg.get("t", "")

        if t == "u":
            content = msg.get("content", "").strip()
            if content:
                parts.append(f"user: {content}")
                user_turns += 1

        elif t == "a":
            content = msg.get("content", "").strip()
            if content:
                # Include tool context if any
                if tool_context:
                    tool_summary = " | ".join(tool_context)
                    parts.append(f"[used: {tool_summary}]")
                    tool_context = []
                parts.append(f"assistant: {content}")
                asst_chars += len(content)

        elif t == "tc":
            tool = msg.get("tool", "?")
            tool_context.append(tool)

        elif t == "tr":
            pass  # tool results - skip, too verbose

        elif t == "e":
            # error
            content = msg.get("content", "")
            if content:
                parts.append(f"[error: {content}]")

        elif t == "d":
            # done marker - ignore
            pass

    if user_turns < MIN_TURNS or asst_chars < MIN_ASSISTANT_CHARS:
        return None

    # Add session header
    ts = meta.get("created") or (messages[0].get("ts") if messages else 0)
    cwd = meta.get("cwd", "")
    model = meta.get("model", "")

    header_parts = []
    if ts:
        dt = datetime.fromtimestamp(ts)
        header_parts.append(dt.strftime("%Y-%m-%d %H:%M"))
    if cwd:
        # Shorten home paths
        cwd = cwd.replace("/home/alice", "~").replace("/steam/rotko", "~/r")
        header_parts.append(cwd)
    if model:
        # Shorten model name
        model = model.replace("claude-", "").replace("-20251001", "")
        header_parts.append(model)

    header = " | ".join(header_parts)
    text = f"# zish session {header}\n" + sanitize_text("\n".join(parts))

    return text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path.home() / ".zish/sessions"))
    parser.add_argument("--output", default="data/sft/sanitized/zish_sessions.jsonl")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    sessions_dir = Path(args.input)
    results = []

    for d in sorted(sessions_dir.iterdir()):
        if not d.is_dir():
            continue
        text = parse_session(d)
        if text:
            results.append({"text": text})

    print(f"Parsed {len(list(sessions_dir.iterdir()))} session dirs")
    print(f"Kept {len(results)} sessions with real content")

    if args.preview:
        for r in results[:8]:
            print("=" * 60)
            print(r["text"][:500])
            print()
    else:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
