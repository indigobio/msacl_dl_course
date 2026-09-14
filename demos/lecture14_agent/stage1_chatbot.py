#!/usr/bin/env python3
"""
Stage 1 - a bare chatbot: NO memory, NO tools.

Each question is sent to Claude on its own and then thrown away. Watch Q2: it
can't remember the run we just named, because nothing carries over between turns.

    export ANTHROPIC_API_KEY=sk-ant-...
    python3 stage1_chatbot.py
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

banner(1, "A BARE CHATBOT", "no memory, no tools — watch Q2 forget the run")

for q in QUESTIONS:
    user(q)
    messages = [                                   # rebuilt every turn => NO memory
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
    ]
    bot(claude(messages))
