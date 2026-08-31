"""LLM backends for the demo.  A backend is just a function: chat(messages) -> text.

  - offline_chat : a stand-in (no API key, no network) that is scripted just
    enough to show the three stages.  It is NOT a real model: it answers general
    questions, INVENTS specific QC numbers when asked (the hallucination beat),
    and, in ReAct mode, calls tools and answers from the observations.
  - openai_chat  : a real LLM via a raw HTTP call to any OpenAI-compatible
    /chat/completions endpoint (OpenAI, or a local server like Ollama). No SDK.
"""
import json
import os
import urllib.request


def make_llm(backend):
    return openai_chat if backend == "openai" else offline_chat


# --------------------------------------------------------------------------- #
# Real LLM: raw HTTP to any OpenAI-compatible endpoint (no SDK, no framework).
# --------------------------------------------------------------------------- #
def openai_chat(messages):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL).")
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stop": ["Observation:"],   # let OUR loop supply the observation
    }).encode()
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- #
# Offline stand-in.  Scripted, deterministic, no dependencies.
# --------------------------------------------------------------------------- #
_CONCEPT = ("An out-of-spec QC run means a control sample fell outside its "
            "acceptance limits (mass accuracy, resolution, or signal), so results "
            "from that batch aren't trustworthy until it's investigated and re-run.")

# The hallucination: confident, fluent, and WRONG (wrong run, invented numbers).
_FABRICATED = ("Today QC-02 is the outlier: its mass error is about +3.4 ppm "
               "(just past \u00b12) and its resolution slipped to ~28,500. The rest "
               "look fine \u2014 I'd recalibrate and re-run QC-02.")


def offline_chat(messages):
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    if "Action:" in system:                       # stage 3 = ReAct system prompt
        return _offline_react(messages)
    q = _last_user(messages)
    kind = _intent(q)
    if kind == "concept":
        return _CONCEPT
    if kind == "data":
        return _FABRICATED                        # no tools -> it makes numbers up
    # follow-up: with memory it stays consistent with its story; without, it's lost.
    if _has_memory(messages):
        return ("For QC-02, recalibrate, repeat the QC, and if the mass error "
                "stays high check the ion source before releasing results.")
    return ("I don't keep memory between questions, so I've lost the earlier "
            "context \u2014 which run do you mean, and what were its numbers?")


def _offline_react(messages):
    q = _last_user(messages)
    kind = _intent(q)
    if kind == "concept":
        return "Thought: General question; I can answer directly.\nAnswer: " + _CONCEPT
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
    if any(w in ql for w in ("what does", "what is", "what's", "mean", "explain", "why do")):
        return "concept"
    if any(w in ql for w in ("do about", "should we", "recommend", "that run", "next step", "re-run")):
        return "followup"
    return "data"


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
