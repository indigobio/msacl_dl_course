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

Output is colour-coded for the projector — a reverse-video chip and a coloured
bar mark every turn: **YOU** (blue), **BOT** (green), **OBS** (amber, a real tool
result), **THINK** / **ACT** (grey / purple, the agent's ReAct steps). Colour
switches itself off when the output is not a terminal; `NO_COLOR=1` forces it off
and `FORCE_COLOR=1` forces it on (useful when piping to `tee` for a transcript).

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
| `tools.py` | the QC tools + ReAct helpers + the system prompt (tools **and** the data inventory) |
| `ui.py`    | terminal colour/layout so the room can tell YOU from BOT from OBS |
| `qc_runs.csv` | 6 QC runs; QC-04 fails all three limits (matches the handout) |

Notes on the live model: stage 3 relies on Claude emitting the
`Action: / Action Input:` lines; capable models (Haiku/Sonnet) follow this
reliably. Stages 1–2 will hallucinate the Q3 number, but the exact value varies
run to run — run it once before class so you know what you'll get, and keep a
saved transcript as your fallback if the network misbehaves.

### An agent only knows what its context says exists

Worth saying out loud during stage 3, because it is the single most common way a
real agent goes wrong. The system prompt (see `REACT_SYSTEM` in `tools.py`) does
not just list *tools* — it lists the **data**:

```
Data files available to you (this is the complete list; never guess a filename
that is not here):
  - qc_runs.csv (6 rows) columns: run_id, mass_error_ppm, resolution, tic_1e6
```

That block is generated by reading the directory (`data_inventory()`), so it can
never drift from what is actually on disk — drop another CSV next to the scripts
and the agent is told about it on the next run.

Without it the demo used to crash, and the reason is worth a minute of class
time: the prompt listed a `load_csv` tool but never named a single file, so when
asked about QC-04 the model had nothing to put in `Action Input` except the run
id — and the tool dutifully tried to open a file called `QC-04`. The model was
not being stupid; **we never told it what existed.** Tool descriptions and an
inventory of the environment are part of the prompt, and an agent that has to
guess will guess.

### If a tool call goes sideways

Belt and braces on top of that, so nothing the model does can kill the demo
mid-sentence in front of the room:

- **A guessed argument.** `load_csv` ignores any argument that is not an existing
  file, falls back to `qc_runs.csv`, and says so in the observation.
- **A tool that raises, or a tool that does not exist.** `run_tool` catches
  everything and returns an `ERROR: ...` string as the observation instead. The
  model reads the error and tries something else — narrate it when it happens,
  because recovering from a failed tool call is exactly what a real agent does.
- **A model that never finishes.** The ReAct loop stops after `MAX_STEPS` (6)
  tool calls rather than looping forever on your API bill.

### Spec limits (hard-coded from the QC SOP)

mass error within **±2 ppm** · resolution **> 30,000** · TIC within **8–12** (×10⁶).
