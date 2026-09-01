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

QUESTIONS = [
    "We're starting morning QC and the run in question is QC-04. "
    "First: what does it mean for a run to be 'out of spec'?",
    "Which run did I just say we're reviewing today?",
    "What is QC-04's mass error, and does it pass QC?",
]

messages = [{"role": "system", "content": REACT_SYSTEM}]   # tools + format described here
state = {}                                                 # the tools' scratchpad

for q in QUESTIONS:
    print(f"\nyou > {q}")
    messages.append({"role": "user", "content": q})
    reply = claude(messages)
    while has_action(reply):                # the ReAct loop: think -> act -> observe
        print(reply)
        messages.append({"role": "assistant", "content": reply})
        name, arg = parse_action(reply)
        obs = run_tool(name, arg, state)
        messages.append({"role": "user", "content": "Observation: " + obs})
        print("obs > " + obs.splitlines()[0] + (" ..." if "\n" in obs else ""))
        reply = claude(messages)
    messages.append({"role": "assistant", "content": reply})
    print(f"bot > {reply}")
