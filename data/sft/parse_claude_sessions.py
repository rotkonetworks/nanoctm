"""
Parse Claude Code session files into SFT training data.

Extracts user/assistant conversation turns, skipping tool calls and thinking blocks.
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime


MIN_TURNS = 2
MIN_ASSISTANT_CHARS = 50
MAX_TEXT_LEN = 4000  # cap per entry to avoid huge dumps


def sanitize_text(text):
    """Redact sensitive values."""
    text = re.sub(r'(\d{1,3}\.\d{1,3}\.)\d{1,3}\.\d{1,3}', r'\1x.x', text)
    text = re.sub(r'-i\s+\S*\.ssh\S*', '-i <key>', text)
    text = re.sub(r'[0-9a-f]{40,}', '<hash>', text)
    text = re.sub(r'(penumbra1[a-z0-9]{12})[a-z0-9]+', r'\1...', text)
    # Redact API keys/tokens
    text = re.sub(r'(sk-|pk-|key-)[a-zA-Z0-9]{20,}', r'\1<redacted>', text)
    return text


def extract_text_from_content(content):
    """Extract plain text from Claude message content (can be string or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # Skip thinking, tool_use, tool_result blocks
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def parse_session(session_file):
    """Parse a Claude Code session JSONL into conversation entries."""
    messages = []
    with open(session_file, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
            except:
                continue

    # Build conversations - split on long gaps or natural breaks
    conversations = []
    current = []
    user_turns = 0
    asst_chars = 0

    for msg in messages:
        msg_type = msg.get("type", "")

        if msg_type == "user":
            content = extract_text_from_content(msg.get("message", {}).get("content", ""))
            content = content.strip()
            if content and len(content) > 2:
                current.append(f"user: {content}")
                user_turns += 1

        elif msg_type == "assistant":
            content = extract_text_from_content(msg.get("message", {}).get("content", ""))
            content = content.strip()
            if content and len(content) > 5:
                # Truncate very long assistant messages
                if len(content) > 2000:
                    content = content[:2000] + "..."
                current.append(f"assistant: {content}")
                asst_chars += len(content)

        elif msg_type == "summary":
            # Conversation was compressed - natural break point
            if user_turns >= MIN_TURNS and asst_chars >= MIN_ASSISTANT_CHARS:
                conversations.append("\n".join(current))
            current = []
            user_turns = 0
            asst_chars = 0

    # Don't forget last conversation
    if user_turns >= MIN_TURNS and asst_chars >= MIN_ASSISTANT_CHARS:
        conversations.append("\n".join(current))

    return conversations


def get_project_name(path):
    """Extract project name from session path."""
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "projects" and i + 1 < len(parts):
            name = parts[i + 1]
            # Clean up the encoded path
            name = name.replace("-steam-rotko-", "").replace("-", "/")
            return name
    return "unknown"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path.home() / ".claude/projects"))
    parser.add_argument("--output", default="data/sft/sanitized/claude_sessions.jsonl")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    projects_dir = Path(args.input)
    results = []
    total_files = 0

    for session_file in sorted(projects_dir.rglob("*.jsonl")):
        # Skip subagent files
        if "subagents" in str(session_file):
            continue
        total_files += 1

        project = get_project_name(session_file)
        conversations = parse_session(session_file)

        for conv in conversations:
            text = sanitize_text(conv)
            # Cap length
            if len(text) > MAX_TEXT_LEN:
                text = text[:MAX_TEXT_LEN] + "\n..."
            if len(text) > 100:  # skip tiny scraps
                results.append({"text": f"# claude session ({project})\n{text}"})

    print(f"Parsed {total_files} session files")
    print(f"Extracted {len(results)} conversation segments")

    if args.preview:
        for r in results[:5]:
            print("=" * 60)
            print(r["text"][:600])
            print()
    else:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
