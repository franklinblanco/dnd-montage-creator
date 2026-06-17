#!/usr/bin/env python3
"""
judge.py — the model that decides whether a candidate window is a real PvP fight
and how montage-worthy it is.

`Verdict` is the schema shared by every backend AND by the label store — it is both
the VLM's structured output and the training label, so the seed teacher and the
distilled student speak the same language. `Judge` is the interface.

Phase 1 ships `ClaudeJudge` (the one-time paid seed teacher, Opus vision).
`LocalVLMJudge` (free, Qwen2.5-VL-7B on the PC) and `TrainedHeadJudge` (the
distilled student) are stubs that slot into the same interface in later phases.

Only `ClaudeJudge` imports the `anthropic` SDK, and it does so lazily — this module
imports fine on a machine that only runs the local pieces.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from frames import Frame

# The four highlight categories the rubric scores for (your picks).
CATEGORIES = ["kill_win", "clutch_lowhp", "funny_fail", "flashy_play"]
SEED_MODEL = "claude-opus-4-8"


@dataclass
class WindowContext:
    clip: str
    start: float
    end: float
    duration: float
    voice: str = ""          # narration overlapping the window (helps the judge)
    player_class: str = ""   # the recording player's class, if known


@dataclass
class Verdict:
    is_pvp: bool
    categories: list[str]
    montage_score: int       # 0-10 overall keep-worthiness
    tight_start: float       # refined window bounds (sec, clip-relative)
    tight_end: float
    confidence: float        # 0-1
    reason: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def output_schema():
    """JSON Schema for structured outputs / the label record. Note: structured
    outputs don't support numeric min/max, so montage_score is constrained with an
    enum and confidence is range-checked client-side."""
    return {
        "type": "object",
        "properties": {
            "is_pvp": {"type": "boolean"},
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": CATEGORIES},
            },
            "montage_score": {"type": "integer", "enum": list(range(11))},
            "tight_start": {"type": "number"},
            "tight_end": {"type": "number"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["is_pvp", "categories", "montage_score",
                     "tight_start", "tight_end", "confidence", "reason"],
        "additionalProperties": False,
    }


# The rubric. Kept as a stable system prompt so it can be prompt-cached across every
# window in the seed batch (the per-window frames go in the user turn).
SYSTEM = """\
You label short windows of Dark and Darker gameplay, recorded from the player's own
point of view, to find montage-worthy PvP moments.

Dark and Darker has NO kill feed and NO enemy nameplates, so judge from what you see
across the frames. They are in time order, each tagged with its [t=..s] timestamp.

PvP vs PvE — the key call:
- PvP = another human PLAYER is involved. Players wear varied, mismatched craftable
  armor and weapons, use class abilities and consumables, and move erratically
  (strafing, jukes, crouch-spam, kiting). Health and stamina swing fast.
- PvE = only AI monsters (skeletons, zombies, goblins, cultists, spiders, ghosts,
  wraiths, etc.) and traps. Mobs look uniform and move in repetitive, predictable
  patterns.
- The recording player's OWN abilities, arrows, spells, and pets are not an
  opponent — look for a separate enemy player. If you only see mobs or an empty
  room, is_pvp is false.

categories (choose any that apply; [] if none):
- kill_win: the player kills an enemy player or clearly wins the fight.
- clutch_lowhp: outnumbered (1vX), very low HP, or a comeback.
- funny_fail: death to a trap, a big whiff, trolling, or otherwise funny/meme.
- flashy_play: a big hit, clean shot, or slick ability combo, even without a kill.

montage_score (0-10) — overall keep-worthiness for a highlight reel:
- 0-2: no PvP / boring / just running or looting.
- 3-5: minor PvP poke or an unremarkable fight.
- 6-8: a solid PvP kill or a good back-and-forth fight.
- 9-10: exceptional — multi-kill, clutch 1vX, or genuinely hilarious.

Also return tight_start and tight_end (seconds, clip-relative) trimmed to just the
action within the window, confidence (0-1) in your call, and a one-line reason.
Be decisive; when the frames are ambiguous, lower the confidence rather than
inflating the score."""


class Judge(Protocol):
    """A judge scores one candidate window from its sampled frames."""
    def score(self, frames: list[Frame], ctx: WindowContext) -> Verdict: ...


class ClaudeJudge:
    """Seed teacher (Opus vision). For the batch seed use `build_request` +
    `parse_message`; `score` does a single synchronous call (handy for testing one
    window). This is the ONLY backend that costs money."""

    def __init__(self, model=SEED_MODEL, effort="medium"):
        self.model = model
        self.effort = effort

    def _user_content(self, frames, ctx):
        head = (f"Clip: {ctx.clip}\n"
                f"Window: {ctx.start:.1f}s-{ctx.end:.1f}s of a {ctx.duration:.0f}s clip.\n")
        if ctx.player_class:
            head += f"The recording player is a {ctx.player_class}.\n"
        if ctx.voice.strip():
            head += f"Player voice during the window: \"{ctx.voice.strip()}\"\n"
        head += f"{len(frames)} frames follow, in time order."
        content = [{"type": "text", "text": head}]
        for fr in frames:
            content.append({"type": "text", "text": f"[t={fr.t:.1f}s]"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(fr.jpeg).decode()}})
        content.append({"type": "text", "text": "Return the JSON verdict for this window."})
        return content

    def _params(self, frames, ctx):
        return {
            "model": self.model,
            "max_tokens": 4000,
            "system": [{"type": "text", "text": SYSTEM,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": self._user_content(frames, ctx)}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": output_schema()}},
        }

    def score(self, frames, ctx) -> Verdict:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(**self._params(frames, ctx))
        return self.parse_message(msg)

    def build_request(self, custom_id, frames, ctx):
        """A Batches API request for one window (−50% vs synchronous)."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        return Request(custom_id=custom_id,
                       params=MessageCreateParamsNonStreaming(**self._params(frames, ctx)))

    @staticmethod
    def parse_message(message) -> Verdict:
        text = next((b.text for b in message.content if b.type == "text"), None)
        if not text:
            raise ValueError("no text block in response")
        v = Verdict.from_dict(json.loads(text))
        v.confidence = max(0.0, min(1.0, float(v.confidence)))  # schema can't clamp
        return v


class LocalVLMJudge:
    """Free local teacher — Qwen2.5-VL-7B via Ollama on the PC. Phase 2."""

    def __init__(self, model="qwen2.5vl:7b", host="http://localhost:11434"):
        self.model, self.host = model, host

    def score(self, frames, ctx) -> Verdict:
        raise NotImplementedError("LocalVLMJudge lands in Phase 2 (free local teacher).")


class TrainedHeadJudge:
    """The distilled student — CLIP embeddings + sklearn heads. Trained on the
    label store; runs free and real-time on either machine. Phase 'Student'."""

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir

    def score(self, frames, ctx) -> Verdict:
        raise NotImplementedError("TrainedHeadJudge lands after the student is trained.")
