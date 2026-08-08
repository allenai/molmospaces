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

    def as_xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


def build_prompt(num_shown: int) -> str:
    return (
        SCENE_DESCRIPTION.format(n=num_shown)
        + "\n"
        + MIDAIR_DESCRIPTION
        + "\n"
        + TASK_INSTRUCTION.format(n=num_shown)
    )


def frame_caption(position: int, total: int) -> str:
    tags = []
    if position == 0:
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


def _to_part(img: np.ndarray | Image.Image, jpeg_quality: int = 92):
    from google.genai import types  # noqa: PLC0415

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def make_client(api_key: str | None = None):
    """Build a genai client. Raises PointingError with actionable text if unusable."""
    import os  # noqa: PLC0415

    try:
        from google import genai  # noqa: PLC0415
    except ImportError as e:
        raise PointingError(
            "google-genai is not installed. `pip install google-genai`."
        ) from e

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise PointingError(
            "GEMINI_API_KEY is not set. Export it locally, or supply it as a Beaker "
            "secret (envVars: - name: GEMINI_API_KEY / secret: ARJUNG_GEMINI_API_KEY)."
        )
    return genai.Client(api_key=key)


def point_from_frames(
    frames: list[np.ndarray],
    client=None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
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
        parts.append(types.Part.from_text(text=frame_caption(position, len(frames))))
        parts.append(_to_part(img))
    parts.append(types.Part.from_text(text=build_prompt(len(frames))))

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=parts, config=config)
            payload = json.loads((response.text or "").strip())
            x, y = parse_point(payload.get("point"))
            return PointingResult(
                x=x,
                y=y,
                reasoning=str(payload.get("reasoning", "")),
                mug_description=str(payload.get("mug_description", "")),
                confidence=float(payload.get("confidence", float("nan"))),
                raw=payload,
            )
        except PointingError:
            raise  # a malformed point will not fix itself on retry
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            msg = str(e)
            is_client_error = (isinstance(status, int) and 400 <= status < 500) or any(
                tok in msg
                for tok in ("404", "400", "401", "403", "PERMISSION_DENIED", "INVALID_ARGUMENT")
            )
            if is_client_error:
                raise PointingError(f"Gemini client error (not retrying): {e}") from e
            last_err = e
            backoff = 2**attempt
            log.warning(
                "Gemini call failed (attempt %d/%d): %s; retrying in %ds",
                attempt + 1,
                max_retries,
                e,
                backoff,
            )
            time.sleep(backoff)

    raise PointingError(f"Gemini failed after {max_retries} attempts: {last_err}")
