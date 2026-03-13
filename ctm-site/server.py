#!/usr/bin/env python3
"""
Minimal static file server + JMAP contact form handler for ctm.rotko.net.

Serves static files from the current directory.
Handles POST /api/contact by sending an email via Stalwart JMAP.

Usage:
    python3 server.py                        # reads secrets from env
    JMAP_USER=hq@rotko.net JMAP_PASS=xxx python3 server.py --port 8080

Or with age-encrypted secrets (auto-decrypt):
    python3 server.py --secrets ~/tommidata/secrets/rotko_mail.age --age-key ~/.ssh/id_claude
"""

import http.server
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from html import escape
from pathlib import Path
from datetime import datetime, timezone
import uuid
import argparse

# --- Config ---

JMAP_URL = os.environ.get("JMAP_URL", "https://mail.rotko.net/jmap/")
JMAP_USER = os.environ.get("JMAP_USER", "")
JMAP_PASS = os.environ.get("JMAP_PASS", "")
JMAP_ACCOUNT_ID = os.environ.get("JMAP_ACCOUNT_ID", "")
NOTIFY_TO = os.environ.get("NOTIFY_TO", "hq@rotko.net,tommi@rotko.net")
SEND_FROM = os.environ.get("SEND_FROM", "noreply@rotko.net")
PORT = int(os.environ.get("PORT", "8080"))
BIND = os.environ.get("BIND", "0.0.0.0")


JMAP_SENT_MAILBOX = os.environ.get("JMAP_SENT_MAILBOX", "")
JMAP_IDENTITY_ID = os.environ.get("JMAP_IDENTITY_ID", "")


def jmap_discover(user: str, password: str) -> tuple[str, str]:
    """Discover JMAP apiUrl and accountId from .well-known."""
    req = urllib.request.Request("https://mail.rotko.net/.well-known/jmap")
    _add_basic_auth(req, user, password)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    api_url = data["apiUrl"]
    account_id = list(data["accounts"].keys())[0]
    return api_url, account_id


def jmap_find_sent_and_identity(user: str, password: str):
    """Find the Sent mailbox ID and identity ID for the account."""
    global JMAP_SENT_MAILBOX, JMAP_IDENTITY_ID

    req = urllib.request.Request(
        JMAP_URL,
        data=json.dumps({
            "using": [
                "urn:ietf:params:jmap:core",
                "urn:ietf:params:jmap:mail",
                "urn:ietf:params:jmap:submission",
            ],
            "methodCalls": [
                ["Mailbox/get", {"accountId": JMAP_ACCOUNT_ID, "properties": ["name", "role"]}, "m0"],
                ["Identity/get", {"accountId": JMAP_ACCOUNT_ID}, "i0"],
            ],
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    _add_basic_auth(req, user, password)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    for call in data["methodResponses"]:
        name, result, _tag = call
        if name == "Mailbox/get":
            for mb in result["list"]:
                if mb.get("role") == "sent":
                    JMAP_SENT_MAILBOX = mb["id"]
                    print(f"  Sent mailbox: {JMAP_SENT_MAILBOX}")
        elif name == "Identity/get":
            for ident in result["list"]:
                if ident["email"] == SEND_FROM:
                    JMAP_IDENTITY_ID = ident["id"]
                    print(f"  Identity: {JMAP_IDENTITY_ID} ({ident['email']})")


def _add_basic_auth(req: urllib.request.Request, user: str, password: str):
    import base64
    creds = base64.b64encode(f"{user}:{password}".encode()).decode()
    req.add_header("Authorization", f"Basic {creds}")


def jmap_send_email(
    subject: str,
    body_text: str,
    to_emails: str | list[str],
    from_email: str,
    reply_to: str | None = None,
) -> bool:
    """Send an email via JMAP Email/set + EmailSubmission/set."""
    if isinstance(to_emails, str):
        to_emails = [e.strip() for e in to_emails.split(",") if e.strip()]

    email_id = f"ctm-{uuid.uuid4().hex[:12]}"
    submission_id = f"sub-{uuid.uuid4().hex[:12]}"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    email_create = {
        "mailboxIds": {JMAP_SENT_MAILBOX: True},
        "from": [{"email": from_email}],
        "to": [{"email": addr} for addr in to_emails],
        "subject": subject,
        "sentAt": now,
        "bodyValues": {
            "body": {
                "value": body_text,
                "isEncodingProblem": False,
                "isTruncated": False,
            }
        },
        "textBody": [{"partId": "body", "type": "text/plain"}],
        "keywords": {"$seen": True},
    }

    if reply_to:
        email_create["replyTo"] = [{"email": reply_to}]

    jmap_request = {
        "using": [
            "urn:ietf:params:jmap:core",
            "urn:ietf:params:jmap:mail",
            "urn:ietf:params:jmap:submission",
        ],
        "methodCalls": [
            [
                "Email/set",
                {
                    "accountId": JMAP_ACCOUNT_ID,
                    "create": {email_id: email_create},
                },
                "c0",
            ],
            [
                "EmailSubmission/set",
                {
                    "accountId": JMAP_ACCOUNT_ID,
                    "create": {
                        submission_id: {
                            "emailId": f"#{email_id}",
                            "identityId": JMAP_IDENTITY_ID,
                            "envelope": {
                                "mailFrom": {"email": from_email},
                                "rcptTo": [{"email": addr} for addr in to_emails],
                            },
                        }
                    },
                },
                "c1",
            ],
        ],
    }

    req = urllib.request.Request(
        JMAP_URL,
        data=json.dumps(jmap_request).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    _add_basic_auth(req, JMAP_USER, JMAP_PASS)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            # Check for errors in method responses
            for call in result.get("methodResponses", []):
                method_name, response_data, _tag = call
                if "notCreated" in response_data:
                    print(f"JMAP error in {method_name}: {response_data['notCreated']}", file=sys.stderr)
                    return False
            return True
    except urllib.error.URLError as e:
        print(f"JMAP request failed: {e}", file=sys.stderr)
        return False


def format_contact_email(form: dict) -> tuple[str, str]:
    """Format form data into email subject + body."""
    name = form.get("name", "Unknown")
    email = form.get("email", "Unknown")
    org = form.get("org", "")
    interest = form.get("interest", "")
    message = form.get("message", "")

    subject = f"[ctm.rotko.net] Contact from {name}"

    lines = [
        f"New inquiry from ctm.rotko.net",
        f"",
        f"Name:         {name}",
        f"Email:        {email}",
        f"Organization: {org or '(not provided)'}",
        f"Interest:     {interest}",
        f"",
    ]
    if message:
        lines += [f"Message:", f"{message}", f""]
    lines += [
        f"---",
        f"Submitted at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]

    return subject, "\n".join(lines)


TICKS_FILE = os.environ.get("TICKS_FILE", "/usr/share/nginx/html/ticks.json")
INGEST_KEY = os.environ.get("INGEST_KEY", "")  # simple API key auth

# In-memory buffer for fast append; flushed to disk periodically
_ticks_buffer = []
_ticks_flush_every = 10  # flush to disk every N snapshots


def ingest_snapshot(snapshot: dict) -> bool:
    """Append a training snapshot to ticks.json and broadcast via WebSocket."""
    global _ticks_buffer

    # Validate minimal fields
    if "step" not in snapshot or "loss" not in snapshot:
        return False

    _ticks_buffer.append(snapshot)
    ws_broadcast(snapshot)

    if len(_ticks_buffer) >= _ticks_flush_every:
        flush_ticks()
    return True


def flush_ticks():
    """Write buffered snapshots to ticks.json."""
    global _ticks_buffer
    if not _ticks_buffer:
        return

    # Read existing
    try:
        with open(TICKS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.extend(_ticks_buffer)
    _ticks_buffer = []

    # Atomic write
    tmp = TICKS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, TICKS_FILE)


# --- WebSocket multicast ---

WS_PORT = int(os.environ.get("WS_PORT", "8082"))
_ws_clients: list[socket.socket] = []
_ws_lock = threading.Lock()


def _ws_encode_frame(data: bytes) -> bytes:
    """Encode a WebSocket text frame (opcode 0x81)."""
    length = len(data)
    if length <= 125:
        header = struct.pack("!BB", 0x81, length)
    elif length <= 65535:
        header = struct.pack("!BBH", 0x81, 126, length)
    else:
        header = struct.pack("!BBQ", 0x81, 127, length)
    return header + data


def _ws_decode_frame(data: bytes) -> bytes | None:
    """Decode a masked WebSocket frame. Returns payload or None."""
    if len(data) < 6:
        return None
    payload_len = data[1] & 0x7F
    mask_start = 2
    if payload_len == 126:
        if len(data) < 8:
            return None
        payload_len = struct.unpack("!H", data[2:4])[0]
        mask_start = 4
    elif payload_len == 127:
        if len(data) < 14:
            return None
        payload_len = struct.unpack("!Q", data[2:10])[0]
        mask_start = 10
    mask = data[mask_start:mask_start + 4]
    payload = data[mask_start + 4:mask_start + 4 + payload_len]
    return bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def ws_broadcast(snapshot: dict):
    """Send a snapshot to all connected WebSocket clients."""
    frame = _ws_encode_frame(json.dumps(snapshot).encode())
    with _ws_lock:
        dead = []
        for client in _ws_clients:
            try:
                client.sendall(frame)
            except Exception:
                dead.append(client)
        for client in dead:
            _ws_clients.remove(client)
            try:
                client.close()
            except Exception:
                pass


def _ws_handle_client(client: socket.socket):
    """Handle a single WebSocket client connection."""
    with _ws_lock:
        _ws_clients.append(client)
    print(f"WS client connected ({len(_ws_clients)} total)")
    try:
        while True:
            data = client.recv(4096)
            if not data:
                break
            # Check for close frame (opcode 0x8)
            if data[0] & 0x0F == 0x8:
                break
            # Check for ping (opcode 0x9), respond with pong
            if data[0] & 0x0F == 0x9:
                payload = _ws_decode_frame(data)
                if payload is not None:
                    pong = struct.pack("!BB", 0x8A, len(payload)) + payload
                    client.sendall(pong)
    except Exception:
        pass
    finally:
        with _ws_lock:
            if client in _ws_clients:
                _ws_clients.remove(client)
        try:
            client.close()
        except Exception:
            pass
        print(f"WS client disconnected ({len(_ws_clients)} total)")


def _ws_server_loop():
    """Accept WebSocket connections with HTTP upgrade handshake."""
    import base64
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((BIND, WS_PORT))
    srv.listen(16)
    print(f"WebSocket server on ws://{BIND}:{WS_PORT}")
    while True:
        client, addr = srv.accept()
        try:
            # Read HTTP upgrade request
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = client.recv(4096)
                if not chunk:
                    client.close()
                    continue
                request += chunk
            headers = request.decode(errors="replace")
            # Extract Sec-WebSocket-Key
            key = None
            for line in headers.split("\r\n"):
                if line.lower().startswith("sec-websocket-key:"):
                    key = line.split(":", 1)[1].strip()
                    break
            print(f"WS handshake from {addr}, key={key}", flush=True)
            if not key:
                client.close()
                continue
            # Compute accept hash
            MAGIC = "258EAFA5-E914-47DA-95CA-5AB5DC85B11B"
            accept = base64.b64encode(
                hashlib.sha1((key + MAGIC).encode()).digest()
            ).decode()
            # Send upgrade response
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "\r\n"
            )
            client.sendall(response.encode())
            threading.Thread(target=_ws_handle_client, args=(client,), daemon=True).start()
        except Exception as e:
            print(f"WS handshake error: {e}", file=sys.stderr)
            try:
                client.close()
            except Exception:
                pass


class CTMHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server with /api/contact and /api/ingest handlers."""

    def do_POST(self):
        if self.path == "/api/ingest":
            return self._handle_ingest()
        if self.path != "/api/contact":
            self.send_error(404)
            return

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 10_000:
            self.send_error(413, "Too large")
            return

        body = self.rfile.read(content_length)

        try:
            form = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Validate required fields
        name = form.get("name", "").strip()
        email = form.get("email", "").strip()
        if not name or not email or "@" not in email:
            self._json_response(400, {"error": "Name and valid email are required."})
            return

        # Sanitize
        for key in form:
            if isinstance(form[key], str):
                form[key] = form[key][:1000]  # cap field length

        # Format and send
        subject, body_text = format_contact_email(form)
        ok = jmap_send_email(
            subject=subject,
            body_text=body_text,
            to_emails=NOTIFY_TO,
            from_email=SEND_FROM,
            reply_to=email,
        )

        if ok:
            print(f"Contact form: {name} <{email}> ({form.get('interest', '')})")
            self._json_response(200, {"ok": True})
        else:
            self._json_response(502, {"error": "Failed to send. Try again later."})

    def _handle_ingest(self):
        """Handle POST /api/ingest — receive training snapshots."""
        # Auth check
        if INGEST_KEY:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {INGEST_KEY}":
                self._json_response(401, {"error": "unauthorized"})
                return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 1_000_000:  # 1MB max
            self.send_error(413, "Too large")
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON"})
            return

        # Accept single snapshot or batch
        snapshots = payload if isinstance(payload, list) else [payload]
        count = 0
        for snap in snapshots:
            if ingest_snapshot(snap):
                count += 1

        self._json_response(200, {"ok": True, "ingested": count})

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Quieter logging
        if "/api/" in (args[0] if args else ""):
            super().log_message(format, *args)


def load_secrets_from_age(age_file: str, age_key: str):
    """Decrypt rotko_mail.age and extract hq@rotko.net credentials."""
    global JMAP_USER, JMAP_PASS
    result = subprocess.run(
        ["age", "-d", "-i", age_key, age_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to decrypt {age_file}: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    secrets = json.loads(result.stdout)
    if "hq@rotko.net" in secrets:
        JMAP_USER = "hq@rotko.net"
        JMAP_PASS = secrets["hq@rotko.net"]["password"]
        print(f"Loaded credentials for {JMAP_USER}")
    else:
        print(f"hq@rotko.net not found in secrets", file=sys.stderr)
        sys.exit(1)


def main():
    global JMAP_URL, JMAP_ACCOUNT_ID, PORT

    parser = argparse.ArgumentParser(description="CTM site server with JMAP contact form")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--secrets", help="Path to rotko_mail.age")
    parser.add_argument("--age-key", default=os.path.expanduser("~/.ssh/id_claude"),
                        help="Path to age identity key")
    args = parser.parse_args()
    PORT = args.port

    # Load secrets (optional — server starts without JMAP, ingest still works)
    try:
        if args.secrets:
            load_secrets_from_age(args.secrets, args.age_key)
        elif not JMAP_USER:
            default_secrets = os.path.expanduser("~/tommidata/secrets/rotko_mail.age")
            default_key = os.path.expanduser("~/.ssh/id_claude")
            if os.path.exists(default_secrets) and os.path.exists(default_key):
                load_secrets_from_age(default_secrets, default_key)
            else:
                print("No JMAP credentials — contact form disabled, ingest still works.", file=sys.stderr)

        if JMAP_USER:
            if not JMAP_ACCOUNT_ID:
                print("Discovering JMAP session...")
                JMAP_URL, JMAP_ACCOUNT_ID = jmap_discover(JMAP_USER, JMAP_PASS)
                print(f"  Account: {JMAP_ACCOUNT_ID} api={JMAP_URL}")
            if not JMAP_SENT_MAILBOX or not JMAP_IDENTITY_ID:
                jmap_find_sent_and_identity(JMAP_USER, JMAP_PASS)
            print("JMAP ready.")
    except Exception as e:
        print(f"JMAP setup failed: {e} — contact form disabled, ingest still works.", file=sys.stderr)

    # Serve from ctm-site directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Start WebSocket multicast server
    threading.Thread(target=_ws_server_loop, daemon=True).start()

    server = http.server.HTTPServer((BIND, PORT), CTMHandler)
    print(f"Serving on http://{BIND}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        flush_ticks()
        print("\nShutdown (flushed ticks).")


if __name__ == "__main__":
    main()
