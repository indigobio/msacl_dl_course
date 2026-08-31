# Lecture 14 live demo — building an agent in three stages

A framework-free demo (no LangChain) that shows, step by step, what turns a
chatbot into a trustworthy agent. **One loop, two switches** — flip them and
watch the behaviour change.

The demo asks the **same three turns** at each stage; watch where each one breaks:

| turn | asks for… | stage 1 (chatbot) | stage 2 (+memory) | stage 3 (+tools) |
|------|-----------|-------------------|-------------------|------------------|
| Q1 | a concept | ✅ good answer | ✅ good answer | ✅ good answer |
| Q2 | **memory** ("which run did I say?") | ❌ **forgets** | ✅ "QC-04" | ✅ "QC-04" |
| Q3 | **the data** ("does QC-04 pass?") | makes it up | **"+0.8 ppm — passes"** (confident & WRONG) | reads file → **"+5.6 ppm — OUT OF SPEC"** |

The punchline: the memory-only chatbot (stage 2) **passes a QC run that actually
fails all three limits** — the very run the agent (stage 3) catches. Same QC table
as your follow-along handout; run **QC-04** (`+5.6 ppm / 22,300 / 3.1`) is the
sole failure. Each step up the ladder adds exactly one thing: Q2 shows what
*memory* buys, Q3 shows what *tools* buy.

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

- **Stage 1 — "the raw LLM is a stateless text function."** Q1 answers fine. Then
  Q2 — "which run did I just say?" — and it has already **forgotten**: no memory,
  every turn starts from nothing. It'll still cheerfully guess at Q3.
- **Stage 2 — "memory ≠ knowledge."** Now Q2 works — it remembers we're on QC-04.
  But Q3 exposes a scarier failure: with no way to *see* the data it invents a
  clean number and **passes a run that is actually out of spec**. Fluent and
  confident is not the same as correct. A chatbot here would have released a bad
  batch. (Tie to Quiz 14 Q1b / Q4: hallucination.)
- **Stage 3 — "tools ground it."** Same loop, but now the model can act:
  `load_csv` → `check_limits` → `draft_summary`, each result fed back as an
  `Observation:`. It answers with the **real** QC-04 values and correctly flags
  it OUT OF SPEC — then hands off to a human. Every number now traces to an
  observation. (Tie to Q3.)

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
