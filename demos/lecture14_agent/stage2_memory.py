#!/usr/bin/env python3
"""
Stage 2 - add MEMORY: keep the whole conversation and send it every turn.

The ONLY change from stage 1: `messages` now persists and we append each reply.
Now Q2 works - it remembers we're on QC-04. But Q3 reveals the next failure: it
still can't SEE the data, so it confidently makes a number up (and may pass a run
that is actually out of spec). Memory is not knowledge.

    python3 stage2_memory.py
"""
from llm import claude
from ui import banner, bot, user

SYSTEM = ("You are a mass-spec QC lab assistant. Answer the user's questions "
          "directly and concisely.")

QUESTIONS = [
    "We're starting morning QC and the run in question is QC-04. "
    "First: what does it mean for a run to be 'out of spec'?",
    "Which run did I just say we're reviewing today?",
    "What is QC-04's mass error, and does it pass QC?",
]

messages = [{"role": "system", "content": SYSTEM}]     # ONE running history

banner(2, "+ MEMORY", "Q2 works now — but Q3 shows memory is not knowledge")

for q in QUESTIONS:
    user(q)
    messages.append({"role": "user", "content": q})
    reply = claude(messages)
    messages.append({"role": "assistant", "content": reply})   # remember it
    bot(reply)
