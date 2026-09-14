"""The LLM call, shared by all three stages: the Claude Messages API over raw
HTTP (no SDK, no framework).  One function: claude(messages) -> text.

Anthropic differs from OpenAI: the system prompt is a separate top-level field,
roles are only user/assistant, and the reply is a list of content blocks. We
translate our simple [{"role","content"}] messages into that shape here so the
three stage scripts stay tiny.

    export ANTHROPIC_API_KEY=sk-ant-...              # from console.anthropic.com
    export ANTHROPIC_MODEL=claude-3-5-haiku-latest   # optional
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request


def claude(messages, retries=4):
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY (and optionally ANTHROPIC_MODEL / ANTHROPIC_BASE_URL).")

    # Split our [system, user, assistant, ...] into Anthropic's shape.
    system = "".join(m["content"] for m in messages if m["role"] == "system")
    convo = [{"role": m["role"], "content": m["content"]}
             for m in messages if m["role"] != "system"]
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": convo,
        "stop_sequences": ["Observation:"],   # let OUR loop supply the observation (stage 3)
    }
    if system:
        payload["system"] = system
    req = urllib.request.Request(
        base + "/v1/messages", data=json.dumps(payload).encode(),
        headers={"x-api-key": key,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return "".join(b.get("text", "") for b in data["content"]).strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            # 429 = rate limit / no credit; 500/503/529 = transient. Back off and retry.
            if e.code in (429, 500, 503, 529) and attempt < retries:
                wait = float(e.headers.get("retry-after", 2 ** attempt))
                print(f"  (HTTP {e.code}; retry {attempt + 1}/{retries} in {wait:.0f}s)", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code == 429:
                raise SystemExit(
                    "Claude API returned 429 (rate limit, or your key has no credit).\n"
                    f"  detail: {detail}\n"
                    "  fixes: wait and retry; check credit/limits at console.anthropic.com;\n"
                    "    or try a different model (export ANTHROPIC_MODEL=claude-3-5-haiku-latest).")
            raise SystemExit(f"Claude API error {e.code}: {detail}")
        except urllib.error.URLError as e:
            raise SystemExit(f"Network error reaching {base}: {e.reason}")
