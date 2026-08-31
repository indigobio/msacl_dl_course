# Lecture 14 live demo — building an agent in three stages

A framework-free demo (no LangChain) that shows, step by step, what turns a
chatbot into a trustworthy agent. **One loop, two switches** — flip them and
watch the behaviour change.

| stage | `memory` | `tools` | what the room sees |
|-------|----------|---------|--------------------|
| 1 · chatbot        | off | off | forgets between turns; **can't see the data** |
| 2 · + memory       | on  | off | stays coherent, but **invents QC numbers** (hallucination) |
| 3 · + tools (ReAct)| on  | on  | **reads the data** → grounded answer, hallucination gone |

Same QC table as your follow-along handout: run **QC-04** (`+5.6 ppm / 22,300 /
3.1`) breaks all three limits.

## Run it

```bash
cd demos/lecture14_agent
python3 agent.py 1      # chatbot        (offline; no API key)
python3 agent.py 2      # + memory
python3 agent.py 3      # + tools (ReAct)

# Use the real Claude API instead of the offline stand-in:
export ANTHROPIC_API_KEY=sk-ant-...              # from console.anthropic.com
export ANTHROPIC_MODEL=claude-3-5-haiku-latest   # optional (any model you can access)
python3 agent.py 3 --claude
```

Pure standard library — nothing to install for the offline path.

**`HTTP 429: Too Many Requests`?** That's the `--claude` path only. 429 means
rate-limited or (most often with a new key) **no credit** — check credit/limits at
console.anthropic.com. The code already retries with backoff; you can also try a
smaller model (`export ANTHROPIC_MODEL=claude-3-5-haiku-latest`), or just run the
offline demo (`python3 agent.py 3`) — it needs no key and never hits the network.

## What to say at each stage

- **Stage 1 — "the raw LLM is a stateless text function."** Ask the concept
  question: it answers fine. Ask *which* run is out of spec: it makes up a run
  and numbers (it has never seen your file). Ask a follow-up: it has already
  forgotten — no memory.
- **Stage 2 — "memory ≠ knowledge."** Now the follow-up works — it remembers the
  conversation. But it *still* can't see the data, so it confidently repeats and
  builds on the **fabricated** QC-02 numbers. Fluent and consistent is not the
  same as correct. (Tie to Quiz 14 Q1b / Q4: hallucination.)
- **Stage 3 — "tools ground it."** Same loop, but now the model can act: it
  `load_csv` → `check_limits` → `draft_summary`, each result fed back as an
  `Observation:`, and answers with the **real** QC-04 values. The hallucination
  disappears because every number now comes from an observation, and it hands
  off to a human. (Tie to Q3.)

The one thing to point at on screen: **the model only ever produces text.** When
it wants to act it writes `Action: check_limits`; *our* loop parses that line and
runs the Python function. The model never touches your files — that is what makes
"restricted tools" a real guardrail.

## Files

| file | what it is |
|------|------------|
| `agent.py` | the demo — one loop, the two switches, the three stages, CLI |
| `llm.py`   | backends: an offline stand-in + a real Claude Messages API call |
| `tools.py` | the QC tools (`load_csv`, `check_limits`, `draft_summary`) + ReAct helpers |
| `qc_runs.csv` | 6 QC runs; QC-04 fails all three limits (matches the handout) |

The offline backend in `llm.py` is a **scripted stand-in**, not a real model — it
is just enough to make the three stages reproducible with no key or network. The
loop, the tool calls, and the grounded final answer are all real. Run it once
before class and the printed transcript *is* your fallback.

With the real Claude backend (`--claude`), stage 3 relies on the model emitting
the `Action: / Action Input:` lines; capable models (Haiku/Sonnet) follow this
reliably. Stages 1–2 will still hallucinate, but the exact fabricated run/number
will vary run to run — the offline stand-in is what keeps the demo identical.

### Spec limits (hard-coded from the QC SOP)

mass error within **±2 ppm** · resolution **> 30,000** · TIC within **8–12** (×10⁶).
