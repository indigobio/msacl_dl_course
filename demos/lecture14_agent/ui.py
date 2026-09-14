"""Terminal decoration for the Lecture 14 live demo.

This runs on a projector in front of a room, so the point is CONTRAST: who is
speaking has to be obvious from the back row. Each turn gets a reverse-video
chip (` YOU `, ` BOT `, ` OBS `) in its own colour, a coloured bar down the left
of the body text, and wrapping that keeps everything off the screen edges.

Colour is switched off automatically when the output is not a terminal (piping
to a file or a pager), and honours the usual NO_COLOR / FORCE_COLOR env vars.

    from ui import banner, user, bot, obs, agent_step, note
"""
import os
import shutil
import sys
import textwrap

# ---------------------------------------------------------------- colour -----
def _colour_enabled():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


COLOUR = _colour_enabled()
WIDTH = min(shutil.get_terminal_size((100, 24)).columns, 100)

# 256-colour codes, chosen to stay legible on both light and dark terminals.
BLUE, GREEN, AMBER, PURPLE, GREY, RED = 39, 41, 214, 177, 245, 203
PAPER = 231  # near-white, for chip text


def _c(text, *codes):
    """Wrap text in the given SGR codes (no-op when colour is off)."""
    if not COLOUR or not codes:
        return text
    return "\033[" + ";".join(str(c) for c in codes) + "m" + text + "\033[0m"


LABEL_W = 5          # every chip is the same width so the gutters line up


def _chip(label, colour):
    """A reverse-video label: white bold text on a solid colour block."""
    return _c(f" {label:^{LABEL_W}} ", 1, 48, 5, colour, 38, 5, PAPER)


def _block(label, colour, text, body_codes=(), bar="┃"):
    """A chip, then the body indented behind a coloured vertical bar."""
    pad = LABEL_W + 5                          # chip + space + bar + space
    body = str(text).rstrip() or "(empty)"
    lines = []
    for para in body.split("\n"):
        lines.extend(textwrap.wrap(para, max(20, WIDTH - pad)) or [""])
    gutter = _c(bar, 1, 38, 5, colour)
    out = [f"{_chip(label, colour)} {gutter} {_c(lines[0], *body_codes)}"]
    for line in lines[1:]:
        out.append(f"{' ' * (LABEL_W + 2)} {gutter} {_c(line, *body_codes)}")
    print("\n".join(out))


# ---------------------------------------------------------------- public -----
def banner(stage, title, subtitle=""):
    """The stage header: a solid bar the room can see from the back."""
    print()
    print(_c("━" * WIDTH, 1, 38, 5, BLUE))
    head = f"  STAGE {stage}  ·  {title}  "
    print(_c(head.ljust(WIDTH), 1, 48, 5, BLUE, 38, 5, PAPER))
    if subtitle:
        print(_c("  " + subtitle, 3, 38, 5, GREY))
    print(_c("━" * WIDTH, 1, 38, 5, BLUE))


def user(text):
    print()
    _block("YOU", BLUE, text, (1,))


def bot(text):
    _block("BOT", RED, text, (1,))


def obs(text):
    """A tool result. Long results are trimmed — the point is that it is REAL."""
    first, *rest = str(text).split("\n")
    _block("OBS", AMBER, first + (f"   (+{len(rest)} more lines)" if rest else ""))


def agent_step(text):
    """The model's reasoning turn: its Thought, then the tool call as one line.

    Action and Action Input are merged into `tool(argument)` — on a projector,
    one line the room can read beats two lines of protocol.
    """
    thought, action, arg, other = [], "", "", []
    for line in str(text).strip().split("\n"):
        if not line.strip():
            continue
        key, sep, rest = line.partition(":")
        k = key.strip().lower()
        if k == "thought" and sep:
            thought.append(rest.strip())
        elif k == "action" and sep:
            action = rest.strip()
        elif k == "action input" and sep:
            arg = rest.strip()
        else:
            other.append(line.strip())
    for t in thought + other:
        _block("THINK", GREY, t, (3,))
    if action:
        if arg.lower() in ("none", "n/a", ""):
            arg = ""
        _block("ACT", PURPLE, f"{action}({arg})", (1,))


def answer(text):
    """Stage 3's final turn: strip the ReAct protocol and show the answer.

    The model signs off with 'Thought: ... / Answer: ...'. The room should see
    the reasoning as reasoning and the answer as the answer, not the raw format.
    """
    body = str(text).strip()
    thoughts = []
    for line in body.split("\n"):
        key, sep, rest = line.partition(":")
        if sep and key.strip().lower() == "thought":
            thoughts.append(rest.strip())
    for t in thoughts:
        _block("THINK", GREY, t, (3,))
    marker = "Answer:"
    if marker in body:
        body = body.split(marker, 1)[1].strip()
    elif thoughts:
        body = "\n".join(l for l in body.split("\n")
                          if not l.strip().lower().startswith("thought:")).strip()
    bot(body)


def note(text):
    """An aside from the script itself (errors, retries, teaching asides)."""
    print(_c("  → " + str(text), 3, 38, 5, PURPLE))
