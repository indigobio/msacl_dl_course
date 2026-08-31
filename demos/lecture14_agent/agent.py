#!/usr/bin/env python3
"""
MSACL DS301 · Lecture 14 · one agent loop, three stages.

Same loop; flip two switches and watch what each one buys:

  stage 1  chatbot          memory=off  tools=off   forgets; can't see the data
  stage 2  + memory         memory=on   tools=off   fluent, but INVENTS QC numbers
  stage 3  + tools (ReAct)  memory=on   tools=on    reads the data; hallucination gone

  python3 agent.py 1              # or 2, or 3   (offline; no API key needed)
  python3 agent.py 3 --claude     # use the real Claude API instead
"""
import sys

from llm import make_llm
from tools import REACT_SYSTEM, has_action, parse_action, run_tool

SYSTEM_CHAT = (
    "You are a mass-spec QC lab assistant. Answer the user's questions directly "
    "and concisely. Give your best specific answer; don't tell the user to go "
    "look it up in another system."
)

# Three turns chosen so each stage fails at a DIFFERENT, visible point:
#   Q1 plants the run id + asks a concept  -> every stage answers well
#   Q2 tests MEMORY                        -> stage 1 (no memory) forgets
#   Q3 tests GROUNDING                     -> stage 2 invents an in-spec number
#                                             for a run that actually FAILS;
#                                             stage 3 reads the file and flags it
QUESTIONS = [
    "We're starting morning QC and the run in question is QC-04. "
    "First: what does it mean for a run to be 'out of spec'?",
    "Which run did I just say we're reviewing today?",
    "What is QC-04's mass error, and does it pass QC?",
]


def agent(questions, memory, tools, llm):
    """The whole agent. `memory` keeps the chat history; `tools` turns on ReAct."""
    system = REACT_SYSTEM if tools else SYSTEM_CHAT
    history = [{"role": "system", "content": system}]   # the running memory
    state = {}                                          # the tools' scratchpad
    for q in questions:
        # MEMORY: reuse the running history, or start each question from scratch.
        messages = history if memory else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": q})
        print(f"\nyou > {q}")

        reply = llm(messages)
        # TOOLS (ReAct): while the model asks to act, run the tool and let it continue.
        while tools and has_action(reply):
            show(reply)
            messages.append({"role": "assistant", "content": reply})
            name, arg = parse_action(reply)
            obs = run_tool(name, arg, state)
            messages.append({"role": "user", "content": "Observation: " + obs})
            print(f"obs > {obs.splitlines()[0]}" + (" ..." if "\n" in obs else ""))
            reply = llm(messages)

        messages.append({"role": "assistant", "content": reply})
        show(reply)


def show(reply):
    for line in reply.splitlines():
        print(f"bot > {line}" if line.strip() else "")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    backend = "claude" if "--claude" in sys.argv else "offline"
    switches = {"1": (False, False), "2": (True, False), "3": (True, True)}
    if stage not in switches:
        sys.exit("usage: python3 agent.py [1|2|3] [--claude]")
    memory, tools = switches[stage]
    print(f"=== stage {stage}:  memory={memory}  tools={tools}  (backend={backend}) ===")
    agent(QUESTIONS, memory, tools, make_llm(backend))


if __name__ == "__main__":
    main()
