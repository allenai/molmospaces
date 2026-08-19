"""Gemini pointing for the object-permanence (mug/ball) task.

Given a chronological sequence of exo-camera frames -- the first showing the mugs still
in mid-air with the ball visible -- ask a Gemini model to point, *in the last frame*, at
the mug the ball ended up hidden under. Answering correctly requires remembering where
the ball went after it stopped being visible.

The prompts, response schema and point convention here are lifted from
``scripts/object_permanence_point_eval.py`` so the online policy and the offline eval
ask the model exactly the same question. NOTE: that script still carries its own copies;
it should be switched over to import from here once it settles down, otherwise the two
will drift.

Points are returned as ``(x, y)`` normalized to ``[0, 1]``, which is the convention
``CAP_Policy``'s back-projection expects.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"

SCENE_DESCRIPTION = """You are looking at {n} frames, in chronological order, from a fixed camera watching a tabletop scene in a simulated house. Two mugs are placed upside-down on the table, and a small yellow ball is on the table under one of them.
"""

MIDAIR_DESCRIPTION = """In the FIRST frame the two mugs are frozen in mid-air above the table, and the small yellow ball is visible on the table surface below them. The ball lies directly beneath exactly one of the two mugs.

Over the following frames both mugs fall straight down and land on the table. Neither mug moves sideways while falling. One mug lands on top of the yellow ball and hides it completely; the other lands on bare table and hides nothing. After landing, neither mug is moved.
"""

TASK_INSTRUCTION = """Your task: in the LAST frame provided (frame {n} of {n}), point to the mug that the yellow ball is hidden under.

Reason about it like this: find the ball in the first frame, work out which of the two mugs comes down on top of it, then track that same mug through the remaining frames. The two mugs look very similar, so identify the correct one by its position, not by its appearance.

Report a single point that lands on the correct mug AS IT APPEARS IN THE LAST FRAME, in [y, x] format normalized to 0-1000 (y measured from the top of the image, x from the left)."""

# Deliberately does NOT mention the ball. The shared SCENE_DESCRIPTION does, and leaking
# it would let the control condition reason about the hidden object anyway -- which is
# exactly the ability the control is meant to remove.
RANDOM_SCENE_DESCRIPTION = """You are looking at {n} frames, in chronological order, from a fixed camera watching a tabletop scene in a simulated house. Two mugs are placed upside-down on the table.
"""

RANDOM_MUG_INSTRUCTION = """Your task: in the LAST frame provided (frame {n} of {n}), pick ONE of the two mugs at random and point to it.

Do not try to work out which mug is more interesting or more likely to be the right answer -- there is no right answer here. Choose between the two mugs arbitrarily, as if flipping a coin, and commit to that choice.

Report a single point that lands on the mug you chose AS IT APPEARS IN THE LAST FRAME, in [y, x] format normalized to 0-1000 (y measured from the top of the image, x from the left)."""

PROMPT_MODES = ("object_permanence", "random_mug")

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": (
                "Where was the ball in the first frame, which mug fell onto it, and where "
                "is that mug in the last frame? Two or three sentences."
            ),
        },
        "mug_description": {
            "type": "STRING",
            "description": (
                "Short description of the chosen mug in the last frame, e.g. 'the left "
                "mug, closer to the camera'."
            ),
        },
        "point": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
            "min_items": 2,
            "max_items": 2,
            "description": "[y, x] of the chosen mug in the LAST frame, normalized 0-1000.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence in [0, 1] that the ball is under the mug pointed at.",
        },
    },
    "property_ordering": ["reasoning", "mug_description", "point", "confidence"],
    "required": ["reasoning", "mug_description", "point", "confidence"],
}


# Schema for the random_mug control. The field descriptions in RESPONSE_SCHEMA name the
# ball explicitly ("where was the ball in the first frame", "confidence that the ball is
# under the mug pointed at"); those descriptions are sent to the model as part of the
# structured-output config and steer it just as the prompt does. Reusing them made the
# control reproduce the object-permanence answer exactly -- same mug, point identical to
# three decimals -- while its reasoning still discussed tracking the ball.
RANDOM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": "One sentence on which of the two mugs you picked. Do not analyse the scene.",
        },
        "mug_description": {
            "type": "STRING",
            "description": (
                "Short description of the chosen mug in the last frame, e.g. 'the left "
                "mug, closer to the camera'."
            ),
        },
        "point": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
            "min_items": 2,
            "max_items": 2,
            "description": "[y, x] of the chosen mug in the LAST frame, normalized 0-1000.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Always report 0.5; this choice is arbitrary.",
        },
    },
    "property_ordering": ["reasoning", "mug_description", "point", "confidence"],
    "required": ["reasoning", "mug_description", "point", "confidence"],
}


class PointingError(RuntimeError):
    """Raised when a point could not be obtained.

    Deliberately fatal rather than falling back to the image centre. RUMClient.infer_point
    returns [0.5, 0.5] on any failure, which for this task lands between the two mugs and
    is indistinguishable from a real answer -- a silent wrong answer is worse than a loud
    failure when the whole point of the run is to measure pointing accuracy.
    """


@dataclass
class PointingResult:
    x: float  # normalized [0, 1], from image left
    y: float  # normalized [0, 1], from image top
    reasoning: str = ""
    mug_description: str = ""
    confidence: float = float("nan")
    raw: dict = field(default_factory=dict)
    # Token accounting from the API response. thoughts_tokens matters most: this is a
    # reasoning model and thinking tokens bill at the OUTPUT rate (5x input), so leaving
    # thinking enabled can dominate the bill.
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0

    def est_cost_usd(self, in_per_m: float = 1.0, out_per_m: float = 5.0) -> float:
        billed_out = self.output_tokens + self.thoughts_tokens
        return self.prompt_tokens / 1e6 * in_per_m + billed_out / 1e6 * out_per_m

    def as_xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


def build_prompt(num_shown: int, prompt_mode: str = "object_permanence") -> str:
    """Assemble the prompt.

    object_permanence -- describe the mid-air frame and the falling mugs, then ask which
                         mug the ball ended up under. Requires remembering the ball.
    random_mug        -- chance-level control: deliberately omits any mention of the ball
                         or the mugs falling, and asks for an arbitrary choice between the
                         two mugs. Isolates "points at the RIGHT mug" from "points at A
                         mug", with the depth back-projection path held identical.
    """
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}")
    if prompt_mode == "random_mug":
        return (
            RANDOM_SCENE_DESCRIPTION.format(n=num_shown)
            + "\n"
            + RANDOM_MUG_INSTRUCTION.format(n=num_shown)
        )
    return (
        SCENE_DESCRIPTION.format(n=num_shown)
        + "\n"
        + MIDAIR_DESCRIPTION
        + "\n"
        + TASK_INSTRUCTION.format(n=num_shown)
    )


def frame_caption(position: int, total: int, prompt_mode: str = "object_permanence") -> str:
    tags = []
    if position == 0 and prompt_mode == "object_permanence":
        tags.append("mugs in mid-air, ball visible")
    if position == total - 1:
        tags.append("LAST FRAME - give your answer in this image")
    suffix = f" ({'; '.join(tags)})" if tags else ""
    return f"Frame {position + 1} of {total}{suffix}:"


def parse_point(raw: object) -> tuple[float, float]:
    """Gemini's [y, x] in 0-1000 -> (x, y) normalized to [0, 1]."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise PointingError(f"Unparseable point: {raw!r}")
    y, x = float(raw[0]), float(raw[1])
    scale = 1.0 if max(abs(x), abs(y)) <= 1.5 else 1000.0  # tolerate 0-1 fractions
    x, y = x / scale, y / scale
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise PointingError(f"Point outside image after normalization: ({x}, {y})")
    return x, y


def _to_part(img: np.ndarray | Image.Image, jpeg_quality: int = 92, max_dim: int | None = None):
    from google.genai import types  # noqa: PLC0415

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    if max_dim is not None and max(img.size) > max_dim:
        # Gemini tiles images into 768px crops at 258 tokens each, so a 960x720 frame
        # costs 2 tiles. Fitting the long edge to <=768 makes it a single tile and halves
        # the image tokens.
        scale = max_dim / max(img.size)
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def make_client(api_key: str | None = None):
    """Build a genai client. Raises PointingError with actionable text if unusable."""
    import os  # noqa: PLC0415

    try:
        from google import genai  # noqa: PLC0415
    except ImportError as e:
        raise PointingError("google-genai is not installed. `pip install google-genai`.") from e

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise PointingError(
            "GEMINI_API_KEY is not set. Export it locally, or supply it as a Beaker "
            "secret mapped to the GEMINI_API_KEY env var."
        )
    return genai.Client(api_key=key)


def point_from_frames(
    frames: list[np.ndarray],
    client=None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
    prompt_mode: str = "object_permanence",
    thinking_budget: int | None = None,
    max_image_dim: int | None = None,
) -> PointingResult:
    """Ask Gemini which mug hides the ball; point in the LAST frame of ``frames``.

    ``frames`` must be chronological, RGB uint8 (H, W, 3), and the first should show the
    mugs still in mid-air -- that frame is what makes the question answerable.
    """
    from google.genai import types  # noqa: PLC0415

    if len(frames) < 2:
        raise PointingError(f"Need at least 2 frames (got {len(frames)})")

    client = client or make_client()

    parts: list = []
    for position, img in enumerate(frames):
        parts.append(types.Part.from_text(text=frame_caption(position, len(frames), prompt_mode)))
        parts.append(_to_part(img, max_dim=max_image_dim))
    parts.append(types.Part.from_text(text=build_prompt(len(frames), prompt_mode)))

    config_kwargs = dict(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=(
            RANDOM_RESPONSE_SCHEMA if prompt_mode == "random_mug" else RESPONSE_SCHEMA
        ),
    )
    if thinking_budget is not None:
        # rum_client.py sets thinking_budget=0 for its Gemini call; this module originally
        # omitted it, leaving reasoning on by default on a reasoning model.
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    config = types.GenerateContentConfig(**config_kwargs)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=parts, config=config)
            payload = json.loads((response.text or "").strip())
            x, y = parse_point(payload.get("point"))
            um = getattr(response, "usage_metadata", None)
            return PointingResult(
                x=x,
                y=y,
                reasoning=str(payload.get("reasoning", "")),
                mug_description=str(payload.get("mug_description", "")),
                confidence=float(payload.get("confidence", float("nan"))),
                raw=payload,
                prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                thoughts_tokens=getattr(um, "thoughts_token_count", 0) or 0,
            )
        except PointingError:
            raise  # a malformed point will not fix itself on retry
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            msg = str(e)
            # 429 is the one 4xx worth retrying: it is ordinarily a per-minute rate limit
            # that clears on its own. It is also what a hard quota exhaustion looks like
            # ("RESOURCE_EXHAUSTED ... prepayment credits"), which will NOT clear -- those
            # simply burn the retries and then fail, which is the right trade. Treating all
            # 4xx as fatal previously meant one transient rate limit killed an episode
            # outright; a run lost 931 episodes that way.
            is_rate_limit = status == 429 or "RESOURCE_EXHAUSTED" in msg or "429" in msg
            is_client_error = not is_rate_limit and (
                (isinstance(status, int) and 400 <= status < 500)
                or any(
                    tok in msg
                    for tok in ("404", "400", "401", "403", "PERMISSION_DENIED", "INVALID_ARGUMENT")
                )
            )
            if is_client_error:
                raise PointingError(f"Gemini client error (not retrying): {e}") from e
            last_err = e
            # Rate limits need a longer wait than a transient server error.
            backoff = (5 * 2**attempt) if is_rate_limit else (2**attempt)
            log.warning(
                "Gemini call failed (attempt %d/%d): %s; retrying in %ds",
                attempt + 1,
                max_retries,
                e,
                backoff,
            )
            time.sleep(backoff)

    raise PointingError(f"Gemini failed after {max_retries} attempts: {last_err}")
