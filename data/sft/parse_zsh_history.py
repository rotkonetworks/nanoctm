"""
Parse zsh history into SFT training data.

Groups related commands into sessions (by time proximity),
filters noise, and outputs {"text": "..."} JSONL.
"""

import json
import re
import sys
from datetime import datetime

# Commands that are pure noise
SKIP_EXACT = {
    "ls", "cd", "clear", "exit", "pwd", "whoami", "history",
    "-", "\\", "", "#", "ccc", "ccb",  # claude code launcher
}

# Prefixes that are noise unless part of a chain
SKIP_PREFIXES = [
    "clear", "exit",
]

# Minimum session length to keep (commands)
MIN_SESSION_LEN = 3
# Max gap between commands to consider same session (seconds)
SESSION_GAP = 120
# Min interesting commands in a session
MIN_INTERESTING = 2

# Commands that are "interesting" — show real work
INTERESTING_PREFIXES = [
    "git", "ssh", "rsync", "cargo", "python", "pcli", "docker", "podman",
    "curl", "gh", "sudo", "systemctl", "journalctl", "openssl",
    "certbot", "haproxy", "nginx", "ansible", "wasm-pack",
    "rustup", "npm", "yarn", "pip", "uv",
    "iptables", "nft", "ip ", "ss ", "dig", "nslookup",
    "gpg", "age", "openssl", "wireguard", "wg",
    "tmux", "screen",
    "tar", "unzip", "zstd",
    "make", "cmake", "gcc", "clang",
    "ligerito",  # custom tool
]


def parse_zsh_history(path):
    """Parse zsh extended history format: `: timestamp:duration;command`"""
    entries = []
    current_cmd = None
    current_ts = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            # New command entry
            m = re.match(r"^: (\d+):\d+;(.*)$", line)
            if m:
                # Save previous if exists
                if current_cmd is not None:
                    entries.append((current_ts, current_cmd))
                current_ts = int(m.group(1))
                current_cmd = m.group(2)
            elif current_cmd is not None:
                # Continuation line (multiline command)
                current_cmd += "\n" + line

    # Don't forget last entry
    if current_cmd is not None:
        entries.append((current_ts, current_cmd))

    return entries


def is_noise(cmd):
    """Check if command is pure noise."""
    stripped = cmd.strip()
    first_word = stripped.split()[0] if stripped.split() else ""

    if first_word in SKIP_EXACT:
        return True
    if stripped in SKIP_EXACT:
        return True
    # Very short commands with no substance
    if len(stripped) <= 2:
        return True
    return False


def is_interesting(cmd):
    """Check if command shows real work."""
    stripped = cmd.strip()
    for prefix in INTERESTING_PREFIXES:
        if stripped.startswith(prefix):
            return True
    # Multiline commands are usually interesting
    if "\n" in cmd:
        return True
    # Pipe chains
    if "|" in cmd and len(cmd) > 20:
        return True
    return False


def is_sensitive(cmd):
    """Filter out potentially sensitive commands."""
    lower = cmd.lower()
    sensitive_patterns = [
        "password", "passwd", "secret", "token=",
        "api_key", "apikey", "private_key",
        ".kdbx",  # keepass
        "credentials",
    ]
    for pat in sensitive_patterns:
        if pat in lower:
            return True
    return False


def sanitize_cmd(cmd):
    """Redact sensitive values but keep command structure."""
    # Redact SSH key paths (keep the command, redact the key path)
    cmd = re.sub(r'-i\s+\S+', '-i <key>', cmd)
    # Redact long penumbra addresses (keep first 12 chars)
    cmd = re.sub(r'(penumbra1[a-z0-9]{12})[a-z0-9]+', r'\1...', cmd)
    # Redact penumbravalid addresses
    cmd = re.sub(r'(penumbravalid1[a-z0-9]{12})[a-z0-9]+', r'\1...', cmd)
    # Redact IP addresses (keep structure, redact last octets)
    cmd = re.sub(r'(\d{1,3}\.\d{1,3}\.)\d{1,3}\.\d{1,3}', r'\1x.x', cmd)
    # Redact port numbers after -p in ssh
    cmd = re.sub(r'(-p\s+)\d{4,5}', r'\1<port>', cmd)
    # Redact bearer tokens / hex strings > 32 chars
    cmd = re.sub(r'[0-9a-f]{32,}', '<hash>', cmd)
    return cmd


def group_sessions(entries, gap=SESSION_GAP):
    """Group entries into sessions by time proximity."""
    if not entries:
        return []

    sessions = []
    current = [entries[0]]

    for i in range(1, len(entries)):
        ts_prev = entries[i - 1][0]
        ts_curr = entries[i][0]

        if ts_curr - ts_prev > gap:
            sessions.append(current)
            current = [entries[i]]
        else:
            current.append(entries[i])

    sessions.append(current)
    return sessions


def session_to_text(session):
    """Convert a session to training text."""
    # Get timestamp range
    ts_start = session[0][0]
    dt = datetime.fromtimestamp(ts_start)

    # Filter noise from session
    cmds = []
    for ts, cmd in session:
        if not is_noise(cmd) and not is_sensitive(cmd):
            cmds.append(sanitize_cmd(cmd.strip()))

    if not cmds:
        return None

    # Count interesting commands
    n_interesting = sum(1 for c in cmds if is_interesting(c))
    if n_interesting < MIN_INTERESTING:
        return None

    if len(cmds) < MIN_SESSION_LEN:
        return None

    # Format as a shell session
    lines = [f"# shell session {dt.strftime('%Y-%m-%d %H:%M')}"]
    for cmd in cmds:
        if "\n" in cmd:
            # Multiline - indent continuation
            parts = cmd.split("\n")
            lines.append(f"$ {parts[0]}")
            for p in parts[1:]:
                lines.append(f"  {p}")
        else:
            lines.append(f"$ {cmd}")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(__import__("pathlib").Path.home() / ".zsh_history"))
    parser.add_argument("--output", default="data/sft/sanitized/zsh_history.jsonl")
    parser.add_argument("--session-gap", type=int, default=SESSION_GAP)
    parser.add_argument("--preview", action="store_true", help="print instead of write")
    args = parser.parse_args()

    entries = parse_zsh_history(args.input)
    print(f"Parsed {len(entries)} history entries")

    sessions = group_sessions(entries, gap=args.session_gap)
    print(f"Grouped into {len(sessions)} sessions")

    results = []
    for session in sessions:
        text = session_to_text(session)
        if text:
            results.append({"text": text})

    print(f"Kept {len(results)} sessions after filtering")

    if args.preview:
        for r in results[:10]:
            print("=" * 60)
            print(r["text"])
            print()
    else:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
