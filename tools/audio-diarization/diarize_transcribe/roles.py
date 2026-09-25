"""Guess speaker roles ("Manager", "Client", ...) from what they say, using Claude.

Needs ANTHROPIC_API_KEY (or another credential the Anthropic SDK can find).
Returns None on any failure so callers can fall back to "Speaker N".
"""

from __future__ import annotations

import json
import logging

from .alignment import Turn, speaker_label
from .formatters import fmt_clock

log = logging.getLogger(__name__)

DEFAULT_ROLES_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You label speakers in a diarized transcript of a conversation.

Speakers are numbered by the order in which they first speak. For each speaker, give a
short role label (1-3 words) in the same language as the conversation, based on what they
say and how others address them. Examples: "Manager", "Client", "Interviewer",
"Candidate", "Doctor", "Patient", "Host", "Guest". If someone is addressed by name and their
role is also clear, use "Name (Role)", e.g. "Ana (Client)". If a role really can't be
inferred, keep "Speaker N"."""

SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker_number": {"type": "integer"},
                    "role": {"type": "string"},
                },
                "required": ["speaker_number", "role"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["speakers"],
    "additionalProperties": False,
}


def _transcript_for_prompt(turns: list[Turn]) -> str:
    return "\n".join(f"[{fmt_clock(t.start)}] {speaker_label(t.speaker)}: {t.text}" for t in turns)


def guess_roles(
    turns: list[Turn],
    context: str | None = None,
    model: str = DEFAULT_ROLES_MODEL,
) -> list[str] | None:
    """Return role names indexed by speaker (index 0 = first speaker), or None."""
    if not turns:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed; skipping role detection")
        return None

    user_msg = f"<transcript>\n{_transcript_for_prompt(turns)}\n</transcript>"
    if context:
        user_msg = f"Context about this recording: {context}\n\n{user_msg}"

    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            thinking={"type": "adaptive"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            # Server-side fallback: if the model declines, the API retries on a recommended model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        log.warning("Anthropic credentials missing or invalid; skipping role detection")
        return None
    except anthropic.RateLimitError:
        log.warning("Anthropic rate limit hit; skipping role detection")
        return None
    except anthropic.APIStatusError as exc:
        log.warning("Anthropic API error %s; skipping role detection", exc.status_code)
        return None
    except anthropic.APIConnectionError:
        log.warning("Could not reach the Anthropic API; skipping role detection")
        return None

    if response.stop_reason in ("refusal", "max_tokens"):
        log.warning("Role detection stopped early (%s)", response.stop_reason)
        return None

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Role detection returned invalid JSON")
        return None

    n_speakers = max(t.speaker for t in turns) + 1
    roles = [speaker_label(i) for i in range(n_speakers)]
    for item in data.get("speakers", []):
        idx = int(item["speaker_number"]) - 1  # prompt shows 1-based "Speaker N"
        role = str(item["role"]).strip()
        if 0 <= idx < n_speakers and role:
            roles[idx] = role
    return roles
