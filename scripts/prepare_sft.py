#!/usr/bin/env python3
"""Prepare SFT data for mixing into pretraining.

Converts all sanitized SFT data into a single JSONL file with {"text": "..."} format,
suitable for the SFT mixing dataloader in train_qwen_ctm.py.

Sources:
  - memory_conversations.jsonl (QA about memory/plasticity)
  - plasticity_test.jsonl (plasticity examples)
  - ISIS_IDENTITY.md (identity/values)
  - conversations/*.json (multi-turn Claude conversations)
  - blogs/ (blog posts)
  - neuroscience/ (papers/notes)

Output: data/sft/sanitized/sft_combined.jsonl
"""

import json
from pathlib import Path


def extract_conversation_text(data):
    """Convert conversation JSON to training text chunks.

    Extracts user/assistant turns, skipping tool calls and system noise.
    Returns list of text chunks (each chunk is a coherent exchange).
    """
    chunks = []
    turns = data.get("turns", [])
    system = data.get("system", "")

    # Include system prompt as its own chunk if substantial
    if system and len(system) > 100:
        chunks.append(system)

    # Group into user-assistant pairs
    current_exchange = []
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")

        if not isinstance(content, str):
            continue

        # Skip tool noise, task notifications, system reminders
        if "<task-notification>" in content or "<system-reminder>" in content:
            continue
        if "<function_calls>" in content or "<invoke" in content:
            continue
        if content.startswith("Error:") or content.startswith("Warning:"):
            continue

        # Clean up
        content = content.strip()
        if len(content) < 20:
            continue

        if role == "user":
            # Start new exchange if we have a pending one
            if current_exchange:
                chunks.append("\n\n".join(current_exchange))
                current_exchange = []
            current_exchange.append(content)
        elif role == "assistant":
            current_exchange.append(content)
            # Complete exchange — save it
            if len(current_exchange) >= 2:
                chunks.append("\n\n".join(current_exchange))
                current_exchange = []

    if current_exchange:
        chunks.append("\n\n".join(current_exchange))

    return chunks


def main():
    sft_dir = Path(__file__).parent.parent / "data" / "sft" / "sanitized"
    output = sft_dir / "sft_combined.jsonl"

    all_texts = []

    # 1. JSONL files (already in {"text": "..."} format)
    for jsonl_file in sorted(sft_dir.glob("*.jsonl")):
        if jsonl_file.name == "sft_combined.jsonl":
            continue
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if len(text) > 50:
                        all_texts.append(text)
                except json.JSONDecodeError:
                    continue
        print(f"  {jsonl_file.name}: {len(all_texts)} texts so far")

    # 2. Identity document
    identity_file = sft_dir / "ISIS_IDENTITY.md"
    if identity_file.exists():
        text = identity_file.read_text().strip()
        if text:
            all_texts.append(text)
        print(f"  ISIS_IDENTITY.md: {len(all_texts)} texts so far")

    # 3. Conversations (extract user/assistant exchanges)
    conv_dir = sft_dir / "conversations"
    if conv_dir.exists():
        for f in sorted(conv_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            chunks = extract_conversation_text(data)
            # Filter: keep chunks between 50-4000 chars (skip tiny and huge)
            chunks = [c for c in chunks if 50 < len(c) < 4000]
            all_texts.extend(chunks)
            print(f"  {f.name}: +{len(chunks)} chunks, {len(all_texts)} total")

    # 4. Blogs
    blogs_dir = sft_dir / "blogs"
    if blogs_dir.exists():
        for f in sorted(blogs_dir.rglob("*")):
            if f.is_file():
                text = f.read_text(errors="ignore").strip()
                if len(text) > 100:
                    # Split long blog posts into ~2000 char chunks
                    if len(text) > 3000:
                        paragraphs = text.split("\n\n")
                        chunk = ""
                        for para in paragraphs:
                            if len(chunk) + len(para) > 2000 and chunk:
                                all_texts.append(chunk.strip())
                                chunk = para
                            else:
                                chunk += "\n\n" + para if chunk else para
                        if chunk.strip():
                            all_texts.append(chunk.strip())
                    else:
                        all_texts.append(text)
        print(f"  blogs/: {len(all_texts)} total")

    # 5. Neuroscience
    neuro_dir = sft_dir / "neuroscience"
    if neuro_dir.exists():
        for f in sorted(neuro_dir.rglob("*")):
            if f.is_file():
                text = f.read_text(errors="ignore").strip()
                if len(text) > 100:
                    if len(text) > 3000:
                        paragraphs = text.split("\n\n")
                        chunk = ""
                        for para in paragraphs:
                            if len(chunk) + len(para) > 2000 and chunk:
                                all_texts.append(chunk.strip())
                                chunk = para
                            else:
                                chunk += "\n\n" + para if chunk else para
                        if chunk.strip():
                            all_texts.append(chunk.strip())
                    else:
                        all_texts.append(text)
        print(f"  neuroscience/: {len(all_texts)} total")

    # Write combined output
    with open(output, "w") as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    total_chars = sum(len(t) for t in all_texts)
    print(f"\nDone: {len(all_texts)} texts, ~{total_chars//1024}KB")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
