#!/usr/bin/env python3
"""
MSACL DS301 · Lecture 14 · live-demo agent  —  the "MS analysis assistant"
==========================================================================

An AGENT = an LLM + TOOLS + a REASON -> ACT loop.  This file is that whole
sentence, written out in plain Python with NO agent framework (no LangChain,
no LlamaIndex).  Everything the agent can do is a normal function you can read
below; the "loop" is ~30 lines at the bottom.

What it does (the five demo beats from the slides):
    READ the QC table  ->  ANALYZE against the spec limits  ->  PLOT the three
    metrics  ->  FLAG the out-of-range run  ->  DRAFT a QC summary for a human.

The key idea to say out loud during the demo:
    The LLM only ever produces TEXT.  It never touches your files.  When it
    wants to act, it writes a line like  `Action: check_limits`.  OUR code
    parses that line, runs the matching Python function, and feeds the result
    back as an `Observation:`.  Tools + the loop are the only new machinery;
    the model is the same chatbot underneath.

Two ways to run
---------------
    # Offline (default) — no API key, no network, never fails live.
    python qc_agent.py

    # With a real LLM (OpenAI-compatible endpoint; raw HTTP, no SDK).
    export OPENAI_API_KEY=sk-...            # or any OpenAI-compatible key
    export OPENAI_MODEL=gpt-4o-mini         # optional
    python qc_agent.py --llm openai
    # Works with a LOCAL model too, e.g. Ollama:
    #   export OPENAI_BASE_URL=http://localhost:11434/v1
    #   export OPENAI_MODEL=llama3.1  ;  export OPENAI_API_KEY=ollama

The offline backend is a tiny SCRIPTED planner — it is not a real model, it
just returns the correct reason->act steps so the demo always runs.  The loop,
the parsing, the tools, and the grounded final answer are all 100% real.
"""

import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# The QC spec limits (what "in control" means for this assay).  A real lab
# would read these from an SOP; we hard-code them so the demo is self-contained.
# ---------------------------------------------------------------------------
MASS_ERROR_ABS_MAX = 2.0        # |mass error| must be <= 2 ppm
RESOLUTION_MIN     = 30_000     # resolution must be > 30,000
TIC_LOW, TIC_HIGH  = 8.0, 12.0  # TIC (x1e6) must be within 8-12


# ===========================================================================
# 1) TOOLS  — each is a plain function.  A tool takes a string argument and a
#    shared `state` dict, does one concrete thing, and returns a short text
#    OBSERVATION.  This is everything the agent is allowed to do.
# ===========================================================================

def load_csv(arg, state):
    """Read the QC table from disk into memory (state['rows'])."""
    path = arg.strip() or "qc_runs.csv"
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "run_id":         r["run_id"],
                "mass_error_ppm": float(r["mass_error_ppm"]),
                "resolution":     float(r["resolution"]),
                "tic_1e6":        float(r["tic_1e6"]),
            })
    state["rows"] = rows
    cols = "run_id, mass_error_ppm, resolution, tic_1e6"
    return f"{len(rows)} runs loaded · columns: {cols}"


def check_limits(arg, state):
    """Apply the three spec limits to every run; record which runs fail."""
    rows = state.get("rows")
    if not rows:
        return "ERROR: no data loaded yet — call load_csv first."
    flagged = []
    for r in rows:
        reasons = []
        if abs(r["mass_error_ppm"]) > MASS_ERROR_ABS_MAX:
            reasons.append(f"mass error {r['mass_error_ppm']:+.1f} ppm (limit ±{MASS_ERROR_ABS_MAX:.0f})")
        if r["resolution"] < RESOLUTION_MIN:
            reasons.append(f"resolution {int(r['resolution']):,} (limit >{RESOLUTION_MIN:,})")
        if not (TIC_LOW <= r["tic_1e6"] <= TIC_HIGH):
            reasons.append(f"TIC {r['tic_1e6']:.1f} (limit {TIC_LOW:.0f}-{TIC_HIGH:.0f})")
        if reasons:
            flagged.append((r["run_id"], reasons))
    state["flagged"] = flagged
    n_pass = len(rows) - len(flagged)
    if not flagged:
        return f"all {len(rows)} runs pass every limit."
    parts = [f"{rid} FAILS — " + "; ".join(rs) for rid, rs in flagged]
    return f"{n_pass}/{len(rows)} runs pass. " + " | ".join(parts)


def plot_qc(arg, state):
    """Plot the three metrics with their limit lines; highlight failed runs.
    Falls back to a text note if matplotlib is unavailable (demo still works)."""
    rows = state.get("rows")
    if not rows:
        return "ERROR: no data loaded yet — call load_csv first."
    out = (arg.strip() or "qc_plot.png")
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)
    flagged_ids = {rid for rid, _ in state.get("flagged", [])}
    try:
        import matplotlib
        matplotlib.use("Agg")               # no display needed
        import matplotlib.pyplot as plt
    except Exception:
        return (f"(matplotlib not installed — skipping the PNG) "
                f"{len(rows)} runs; flagged: {sorted(flagged_ids) or 'none'}")

    ids = [r["run_id"] for r in rows]
    x = range(len(rows))
    panels = [
        ("mass error (ppm)", [r["mass_error_ppm"] for r in rows],
         [(-MASS_ERROR_ABS_MAX, MASS_ERROR_ABS_MAX)]),
        ("resolution", [r["resolution"] for r in rows], [(RESOLUTION_MIN, None)]),
        ("TIC (x1e6)", [r["tic_1e6"] for r in rows], [(TIC_LOW, TIC_HIGH)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (title, vals, limits) in zip(axes, panels):
        colors = ["#C4534F" if i in flagged_ids else "#0E7C7B" for i in ids]
        ax.bar(x, vals, color=colors)
        for lo, hi in limits:
            if lo is not None:
                ax.axhline(lo, color="#8A9099", ls="--", lw=1)
            if hi is not None:
                ax.axhline(hi, color="#8A9099", ls="--", lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(list(x))
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return f"saved {os.path.basename(out)} (3 panels; failed runs in red, limits dashed)"


def draft_summary(arg, state):
    """Compose a QC summary that names the failed run(s) and their EXACT
    out-of-range values — every number comes from the data, none invented —
    and hands off to a human.  Returns the summary text as the observation."""
    rows = state.get("rows")
    flagged = state.get("flagged")
    if rows is None or flagged is None:
        return "ERROR: run load_csv and check_limits first."
    if not flagged:
        return "QC SUMMARY: all runs within spec; no action needed. (Please verify before releasing results.)"
    lines = ["QC SUMMARY (DRAFT — for human review):"]
    for rid, reasons in flagged:
        lines.append(f"  • {rid} is OUT OF SPEC — " + "; ".join(reasons) + ".")
    lines.append("  Recommendation: recalibrate and re-run the flagged run(s); "
                 "hold affected results.")
    lines.append("  ** Please verify these numbers before releasing any results. **")
    return "\n".join(lines)


# The registry the agent is told about.  name -> (function, one-line description)
TOOLS = {
    "load_csv":      (load_csv,      "load the QC table from disk (arg: filename)"),
    "check_limits":  (check_limits,  "flag runs that break any spec limit (no arg)"),
    "plot_qc":       (plot_qc,       "save a 3-panel plot with limits (arg: out.png)"),
    "draft_summary": (draft_summary, "draft a QC summary naming failed runs (no arg)"),
}


def build_system_prompt():
    """The system prompt: role, the tools, the strict ReAct format, guardrails."""
    tool_lines = "\n".join(f"  - {name}: {desc}" for name, (_, desc) in TOOLS.items())
    return f"""You are a mass-spec QC assistant. You help by REASONING about what to do, then
ACTING with one tool at a time, reading the tool's result, and repeating until you can answer.

You may use ONLY these tools:
{tool_lines}

Always reply in EXACTLY this format, one step at a time:

Thought: <your reasoning about the single next step>
Action: <one tool name from the list above>
Action Input: <the argument, or "none">

After each Action the user will reply with an "Observation:" line. Use it, then take the next step.
When you have flagged the bad run(s) and drafted the summary, reply instead with:

Thought: <why you are done>
Answer: <the final QC summary, built ONLY from the observations>

Rules: use one Action per turn. Never invent a number — every value must come from an Observation.
The final answer must hand off to a human for verification before any result is released."""


# ===========================================================================
# 2) LLM BACKENDS  — a backend is just a function  llm(messages) -> text.
# ===========================================================================

# --- (a) OFFLINE scripted planner: not a real model, just returns the right
#     reason->act steps so the live demo always runs.  The loop/parsing/tools
#     are still real; only the "thinking" is canned.
_SCRIPT = [
    ("I need to see the QC runs and columns before I can judge anything, so I'll load the table.",
     "load_csv", "qc_runs.csv"),
    ("Now I'll check every run against the spec limits to find any that fail.",
     "check_limits", "none"),
    ("Let me plot the three metrics with their limit lines so the bad run is obvious.",
     "plot_qc", "qc_plot.png"),
    ("A run failed all its checks; I'll draft a QC summary naming it and its exact values.",
     "draft_summary", "none"),
]

def offline_llm(messages):
    # How many steps have we already taken?  = number of assistant turns so far.
    taken = sum(1 for m in messages if m["role"] == "assistant")
    if taken < len(_SCRIPT):
        thought, action, arg = _SCRIPT[taken]
        return f"Thought: {thought}\nAction: {action}\nAction Input: {arg}"
    # Final turn: build the Answer straight from the last Observation (the draft).
    last_obs = ""
    for m in reversed(messages):
        if m["role"] == "user" and m["content"].startswith("Observation:"):
            last_obs = m["content"][len("Observation:"):].strip()
            break
    return ("Thought: The summary is ready and flags the failing run with its "
            "out-of-range values; a human must verify before release.\n"
            f"Answer: {last_obs}")


# --- (b) REAL LLM via a raw HTTP call to any OpenAI-compatible /chat/completions
#     endpoint.  No SDK, no framework — you can read exactly what is sent.
def openai_llm(messages):
    import json
    import urllib.request
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not key:
        sys.exit("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL).")
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stop": ["Observation:"],   # let OUR code supply the observation
    }).encode()
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


# ===========================================================================
# 3) PARSING + PRETTY PRINTING
# ===========================================================================

class C:  # course-palette-ish ANSI colors (teal / amber / grey / red)
    THOUGHT = "\033[36m"; ACTION = "\033[33m"; OBS = "\033[90m"
    ANSWER = "\033[31m"; BOLD = "\033[1m"; OFF = "\033[0m"

USE_COLOR = True
def paint(s, col):
    return f"{col}{s}{C.OFF}" if USE_COLOR else s


def parse_reply(text):
    """Pull Thought / Action / Action Input / Answer out of the model's text."""
    def grab(label):
        m = re.search(rf"{label}:\s*(.*?)(?:\n[A-Z][a-z]+ ?[A-Za-z]*:|\Z)",
                      text, re.S)
        return m.group(1).strip() if m else None
    return grab("Thought"), grab("Action"), grab("Action Input"), grab("Answer")


def dispatch(action, action_input, state):
    """Run the tool the model asked for, or report an unknown-tool error."""
    if action not in TOOLS:
        return f"ERROR: unknown tool '{action}'. Choose from {list(TOOLS)}."
    fn, _ = TOOLS[action]
    arg = "" if (action_input or "").lower() in ("", "none", "n/a") else action_input
    return fn(arg, state)


# ===========================================================================
# 4) THE ReAct LOOP  — the whole "agent" is right here.
# ===========================================================================

def run_agent(question, llm, max_steps=8):
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": question},
    ]
    state = {}
    print(paint(f"\nUSER: {question}\n", C.BOLD))

    for step in range(1, max_steps + 1):
        reply = llm(messages)                       # 1. the model produces TEXT
        messages.append({"role": "assistant", "content": reply})
        thought, action, action_input, answer = parse_reply(reply)   # 2. parse it

        if thought:
            print(paint(f"[{step}] THOUGHT  ", C.THOUGHT) + thought)

        if answer is not None:                       # 3a. done?
            print(paint("    ANSWER   ", C.ANSWER) + "\n" +
                  "\n".join("      " + ln for ln in answer.splitlines()))
            print()
            return answer

        if not action:                               # model didn't ask for a tool
            print(paint("    (no action parsed — stopping)", C.OBS)); return None

        print(paint("    ACTION   ", C.ACTION) +
              f"{action}(" + (action_input or "") + ")")
        observation = dispatch(action, action_input, state)          # 3b. run tool
        print(paint("    OBSERV   ", C.OBS) +
              observation.replace("\n", "\n             "))
        messages.append({"role": "user", "content": "Observation: " + observation})

    print(paint("(reached max_steps without an Answer)", C.OBS))
    return None


# ===========================================================================
# 5) CLI
# ===========================================================================

def main():
    global USE_COLOR
    ap = argparse.ArgumentParser(description="Lecture 14 live-demo ReAct agent.")
    ap.add_argument("--llm", choices=["offline", "openai"], default="offline",
                    help="offline (default, no key) or openai (real LLM via HTTP)")
    ap.add_argument("--question", default=(
        "Check this week's QC runs against the spec limits, plot them, flag any "
        "out-of-range run, and draft a QC summary for review."))
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = ap.parse_args()
    USE_COLOR = not args.no_color and sys.stdout.isatty()

    llm = offline_llm if args.llm == "offline" else openai_llm
    print(paint("=" * 68, C.OBS))
    print(paint(f"  MS QC assistant  ·  backend = {args.llm}  ·  "
                "reason -> act -> observe -> ... -> answer", C.BOLD))
    print(paint("=" * 68, C.OBS))
    run_agent(args.question, llm)


if __name__ == "__main__":
    main()
