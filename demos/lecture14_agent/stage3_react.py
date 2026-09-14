#!/usr/bin/env python3
"""
Stage 3 - add TOOLS (ReAct): let the model read the data.

Same memory loop as stage 2, plus ONE idea: when the model writes
`Action: <tool>`, WE run that Python function and feed the result back as an
`Observation:`. It keeps thinking -> acting -> observing until it can answer.
Now Q3 is grounded in the real file - the hallucination is gone and QC-04 is
correctly flagged OUT OF SPEC.

The tools (load_csv, check_limits, draft_summary) and the ReAct system prompt
live in tools.py.

    python3 stage3_react.py
"""
from llm import claude
from tools import REACT_SYSTEM, has_action, parse_action, run_tool
from ui import agent_step, answer, banner, bot, note, obs, user

MAX_STEPS = 6          # a live demo must never spin forever on a confused model

QUESTIONS = [
    "We're starting morning QC and the run in question is QC-04. "
    "First: what does it mean for a run to be 'out of spec'?",
    "Which run did I just say we're reviewing today?",
    "What is QC-04's mass error, and does it pass QC?",
]

messages = [{"role": "system", "content": REACT_SYSTEM}]   # tools + format described here
state = {}                                                 # the tools' scratchpad

banner(3, "+ TOOLS (ReAct)", "think -> act -> observe, until it can answer from real data")

for q in QUESTIONS:
    user(q)
    messages.append({"role": "user", "content": q})
    reply = claude(messages, stop=["Observation:"])
    steps = 0
    capped = False
    while has_action(reply):            # the ReAct loop: think -> act -> observe
        if steps >= MAX_STEPS:
            note(f"stopped after {MAX_STEPS} tool calls without an Answer.")
            capped = True
            break
        steps += 1
        agent_step(reply)
        messages.append({"role": "assistant", "content": reply})
        name, arg = parse_action(reply)
        result = run_tool(name, arg, state)          # never raises: see tools.py
        messages.append({"role": "user", "content": "Observation: " + result})
        obs(result)
        if result.startswith("ERROR:"):
            note("the tool errored — watch it read that and try again")
        reply = claude(messages, stop=["Observation:"])
    messages.append({"role": "assistant", "content": reply})
    if capped:
        bot(f"(no answer — the loop hit the {MAX_STEPS}-tool-call cap)")
    else:
        answer(reply)
