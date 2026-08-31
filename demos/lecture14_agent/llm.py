"""LLM backends for the demo.  A backend is just a function: chat(messages) -> text.

  - offline_chat   : a stand-in (no API key, no network) that is scripted just
    enough to show the three stages.  It is NOT a real model: it answers general
    questions, INVENTS specific QC numbers when asked (the hallucination beat),
    and, in ReAct mode, calls tools and answers from the observations.
  - anthropic_chat : a real LLM via a raw HTTP call to the Claude Messages API
    (https://api.anthropic.com/v1/messages).  No SDK, no framework.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request


def make_llm(backend):
    return anthropic_chat if backend == "claude" else offline_chat


# --------------------------------------------------------------------------- #
# Real LLM: raw HTTP to the Claude Messages API (no SDK, no framework).
# Anthropic differs from OpenAI: the system prompt is a separate top-level
# field, roles are only user/assistant, and the reply is a list of content
# blocks.  We translate our messages accordingly.
# --------------------------------------------------------------------------- #
def anthropic_chat(messages, retries=4):
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
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
        "temperature": 0,
        "stop_sequences": ["Observation:"],   # let OUR loop supply the observation
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
                    "    try a smaller model (export ANTHROPIC_MODEL=claude-3-5-haiku-latest);\n"
                    "    or just run the offline demo:  python3 agent.py 3")
            raise SystemExit(f"Claude API error {e.code}: {detail}")
        except urllib.error.URLError as e:
            raise SystemExit(f"Network error reaching {base}: {e.reason}\n"
                             "  (offline fallback:  python3 agent.py 3)")


# --------------------------------------------------------------------------- #
# Offline stand-in.  Scripted, deterministic, no dependencies.
# --------------------------------------------------------------------------- #
_CONCEPT = ("An out-of-spec QC run means a control sample fell outside its "
            "acceptance limits (mass accuracy, resolution, or signal), so results "
            "from that batch aren't trustworthy until it's investigated and re-run.")

# The dangerous hallucination: fluent, confident, and WRONG in the worst way —
# it PASSES a run that actually fails all three limits. No data, so it guesses.
_FABRICATED = ("QC-04's mass error is about +0.8 ppm — comfortably inside the "
               "\u00b12 ppm limit — so it looks fine and passes QC.")

_MEMORY_ANSWER = "You said we're reviewing QC-04 this morning."


def offline_chat(messages):
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    if "Action:" in system:                       # stage 3 = ReAct system prompt
        return _offline_react(messages)
    q = _last_user(messages)
    kind = _intent(q)
    if kind == "concept":
        return _CONCEPT
    if kind == "recall":
        # THE memory test: with history it recalls; without, it has nothing.
        if _has_memory(messages):
            return _MEMORY_ANSWER
        return ("I don't keep any memory between questions, so I've already lost "
                "what you told me — which run did you mean?")
    if kind == "data":
        return _FABRICATED                        # no tools -> it makes a number up
    # follow-up: with memory it stays consistent with its story; without, it's lost.
    if _has_memory(messages):
        return ("Recalibrate and re-run it, then repeat the QC before releasing "
                "any results.")
    return "I've lost the earlier context — which run do you mean?"


def _offline_react(messages):
    q = _last_user(messages)
    kind = _intent(q)
    if kind == "concept":
        return "Thought: General question; I can answer directly.\nAnswer: " + _CONCEPT
    if kind == "recall":
        return ("Thought: The run id came from our conversation, not the data file.\n"
                "Answer: " + _MEMORY_ANSWER)
    if kind == "followup":
        return ("Thought: I already read the data earlier, so I can answer from that.\n"
                "Answer: Recalibrate and re-run QC-04, then repeat the QC before releasing "
                "results \u2014 it failed mass error, resolution and TIC.")
    # data question -> walk the tools, one per turn, then answer from the last observation.
    plan = [
        ("I need the real data first, so I'll load the QC table.", "load_csv", "qc_runs.csv"),
        ("Now check every run against the spec limits.", "check_limits", "none"),
        ("A run failed; draft a QC summary with its exact values.", "draft_summary", "none"),
    ]
    steps = _obs_since_last_user(messages)
    if steps < len(plan):
        thought, action, arg = plan[steps]
        return f"Thought: {thought}\nAction: {action}\nAction Input: {arg}"
    return "Thought: The tools gave me the real numbers.\nAnswer: " + _last_observation(messages)


# ---- small helpers the stand-in uses to read the running conversation -------
def _last_user(messages):
    for m in reversed(messages):
        if m["role"] == "user" and not m["content"].startswith("Observation:"):
            return m["content"]
    return ""


def _has_memory(messages):
    # More than [system, current-user] means we're carrying history forward.
    return sum(1 for m in messages if m["role"] != "system") > 1


def _intent(q):
    ql = q.lower()
    if any(w in ql for w in ("did i", "which run did", "reviewing today", "what did i say")):
        return "recall"                          # a memory test, not a data lookup
    if any(w in ql for w in ("what does", "definition", "mean", "explain", "why do")):
        return "concept"
    if any(w in ql for w in ("do about", "should we", "recommend", "that run", "next step", "re-run")):
        return "followup"
    return "data"                                # anything asking for a value/verdict


def _obs_since_last_user(messages):
    idx = max((i for i, m in enumerate(messages)
               if m["role"] == "user" and not m["content"].startswith("Observation:")), default=-1)
    return sum(1 for m in messages[idx + 1:]
               if m["role"] == "user" and m["content"].startswith("Observation:"))


def _last_observation(messages):
    for m in reversed(messages):
        if m["role"] == "user" and m["content"].startswith("Observation:"):
            return m["content"][len("Observation:"):].strip()
    return ""
