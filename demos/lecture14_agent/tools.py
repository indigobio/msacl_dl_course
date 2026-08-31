"""QC tools for the Lecture 14 agent demo — plain functions over a CSV.

Everything the agent is allowed to DO lives here, plus three tiny helpers the
ReAct loop needs: tell the model which tools exist (REACT_SYSTEM), spot a tool
call in the model's text (has_action / parse_action), and run it (run_tool).
Nothing in this file knows about the LLM.
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Spec limits (a real lab reads these from an SOP; hard-coded here to stay self-contained).
MASS_ERROR_ABS_MAX = 2.0        # |mass error| <= 2 ppm
RESOLUTION_MIN = 30_000         # resolution > 30,000
TIC_LOW, TIC_HIGH = 8.0, 12.0   # TIC (x1e6) within 8-12


def load_csv(arg, state):
    """Read the QC table from disk into memory (state['rows'])."""
    path = arg or "qc_runs.csv"
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(
                run_id=r["run_id"],
                mass_error_ppm=float(r["mass_error_ppm"]),
                resolution=float(r["resolution"]),
                tic_1e6=float(r["tic_1e6"]),
            ))
    state["rows"] = rows
    return f"{len(rows)} runs loaded; columns: run_id, mass_error_ppm, resolution, tic_1e6"


def check_limits(arg, state):
    """Flag every run that breaks any spec limit; record the offenders."""
    rows = state.get("rows")
    if not rows:
        return "ERROR: load the data first."
    flagged = []
    for r in rows:
        why = []
        if abs(r["mass_error_ppm"]) > MASS_ERROR_ABS_MAX:
            why.append(f"mass error {r['mass_error_ppm']:+.1f} ppm (limit \u00b12)")
        if r["resolution"] < RESOLUTION_MIN:
            why.append(f"resolution {int(r['resolution']):,} (limit >30,000)")
        if not (TIC_LOW <= r["tic_1e6"] <= TIC_HIGH):
            why.append(f"TIC {r['tic_1e6']:.1f} (limit 8-12)")
        if why:
            flagged.append((r["run_id"], why))
    state["flagged"] = flagged
    if not flagged:
        return f"all {len(rows)} runs pass."
    return f"{len(rows) - len(flagged)}/{len(rows)} pass; " + " | ".join(
        f"{rid} FAILS: " + "; ".join(w) for rid, w in flagged)


def draft_summary(arg, state):
    """Draft a QC summary naming the failed run(s) and their exact values."""
    flagged = state.get("flagged")
    if flagged is None:
        return "ERROR: check the limits first."
    if not flagged:
        return "QC summary: all runs within spec. (Verify before release.)"
    lines = ["QC summary (DRAFT for human review):"]
    for rid, why in flagged:
        lines.append(f"  - {rid} is OUT OF SPEC: " + "; ".join(why) + ".")
    lines.append("  Recommend recalibrate + re-run; hold affected results. "
                 "Please verify before releasing any results.")
    return "\n".join(lines)


# name -> (function, one-line description shown to the model)
TOOLS = {
    "load_csv":      (load_csv,      "load the QC table from disk (arg: filename)"),
    "check_limits":  (check_limits,  "flag runs that break any spec limit"),
    "draft_summary": (draft_summary, "draft a QC summary naming the failed runs"),
}


def run_tool(name, arg, state):
    if name not in TOOLS:
        return f"ERROR: no tool '{name}'."
    return TOOLS[name][0](arg, state)


def has_action(text):
    """True if the model asked to call a tool (and hasn't already answered)."""
    return "Action:" in text and "Answer:" not in text


def parse_action(text):
    """Pull (tool_name, argument) out of the model's Action / Action Input lines."""
    name = re.search(r"Action:\s*(\w+)", text)
    arg = re.search(r"Action Input:\s*(.*)", text)
    a = arg.group(1).strip() if arg else ""
    if a.lower() in ("none", "n/a", ""):
        a = ""
    return (name.group(1) if name else ""), a


# The system prompt for stage 3: lists the tools and the strict ReAct format.
REACT_SYSTEM = (
    "You are a mass-spec QC assistant that can use tools.\n"
    "Tools:\n" + "\n".join(f"  - {n}: {d}" for n, (_, d) in TOOLS.items()) + "\n\n"
    "Reply in this format, one step at a time:\n"
    "Thought: <your reasoning>\n"
    "Action: <one tool name>\n"
    "Action Input: <argument, or none>\n\n"
    "I will reply with an 'Observation:' line. Repeat until you can finish with:\n"
    "Thought: <why you are done>\n"
    "Answer: <final answer, built ONLY from the observations>\n\n"
    "Never invent a number; every value must come from an Observation."
)
