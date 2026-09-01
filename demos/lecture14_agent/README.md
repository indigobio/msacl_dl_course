# Lecture 14 live demo — building an agent in three steps

Three small scripts, run in order, show what turns a chatbot into a trustworthy
agent. Each file adds **exactly one idea** — open them side by side and diff them.

| file | the one new idea | what the room sees |
|------|------------------|--------------------|
| `stage1_chatbot.py` | a bare LLM call | forgets between turns — **no memory** |
| `stage2_memory.py`  | keep the conversation | remembers, but **invents QC numbers** |
| `stage3_react.py`   | let it call tools (ReAct) | **reads the data** → grounded, correct |

All three ask the **same three questions**; watch where each one breaks:

| turn | asks for… | stage 1 | stage 2 | stage 3 |
|------|-----------|---------|---------|---------|
| Q1 | a concept | ✅ | ✅ | ✅ |
| Q2 | **memory** ("which run did I say?") | ❌ forgets | ✅ "QC-04" | ✅ "QC-04" |
| Q3 | **the data** ("does QC-04 pass?") | makes it up | **"+0.8 ppm — passes"** (confident & WRONG) | reads file → **"+5.6 ppm — OUT OF SPEC"** |

The punchline: the memory-only chatbot (stage 2) **passes a QC run that actually
fails all three limits** — the very run the agent (stage 3) catches. Same QC table
as your follow-along handout; run **QC-04** (`+5.6 ppm / 22,300 / 3.1`) is the
sole failure.

## Run it

```bash
cd demos/lecture14_agent
export ANTHROPIC_API_KEY=sk-ant-...              # from console.anthropic.com
export ANTHROPIC_MODEL=claude-3-5-haiku-latest   # optional (any model you can access)

python3 stage1_chatbot.py     # no memory
python3 stage2_memory.py      # + memory
python3 stage3_react.py       # + tools (ReAct)
```

Only the standard library is needed (plus a Claude API key).

## The diff, step by step

- **stage 1 → stage 2.** In stage 1 the message list is rebuilt from scratch
  every turn, so nothing carries over. In stage 2 there is one running `messages`
  list and we append each reply — that single change is "memory."
- **stage 2 → stage 3.** Same memory loop, but the system prompt now lists tools
  and a `Thought / Action / Action Input` format. When the model writes an
  `Action:`, the script runs that Python function and appends an `Observation:`.
  That inner loop is "ReAct." The model still only ever produces **text** — the
  script is what actually touches the data. That is why "restricted tools" is a
  real guardrail: the model can't do anything the script doesn't run for it.

## What to say at each stage

- **Stage 1 — "the raw LLM is a stateless text function."** Q1 answers fine.
  Then Q2 — "which run did I just say?" — and it has already forgotten. No memory,
  every turn starts from nothing.
- **Stage 2 — "memory ≠ knowledge."** Now Q2 works. But Q3 exposes a scarier
  failure: with no way to *see* the data it invents a clean number and passes a
  run that is actually out of spec. A chatbot here would release a bad batch.
  (Tie to Quiz 14 Q1b / Q4: hallucination.)
- **Stage 3 — "tools ground it."** `load_csv → check_limits → draft_summary`,
  each result fed back as an `Observation:`. It answers with the real QC-04
  numbers, flags it OUT OF SPEC, and hands off to a human. Every number now
  traces to an observation. (Tie to Q3.)

## Files

| file | what it is |
|------|------------|
| `stage1_chatbot.py` | stage 1 — one LLM call per turn, no memory |
| `stage2_memory.py`  | stage 2 — a running conversation history |
| `stage3_react.py`   | stage 3 — the ReAct loop over the tools |
| `llm.py`   | the shared Claude Messages API call (raw HTTP, no SDK) |
| `tools.py` | the QC tools + ReAct helpers + the ReAct system prompt |
| `qc_runs.csv` | 6 QC runs; QC-04 fails all three limits (matches the handout) |

Notes on the live model: stage 3 relies on Claude emitting the
`Action: / Action Input:` lines; capable models (Haiku/Sonnet) follow this
reliably. Stages 1–2 will hallucinate the Q3 number, but the exact value varies
run to run — run it once before class so you know what you'll get, and keep a
saved transcript as your fallback if the network misbehaves.

### Spec limits (hard-coded from the QC SOP)

mass error within **±2 ppm** · resolution **> 30,000** · TIC within **8–12** (×10⁶).
