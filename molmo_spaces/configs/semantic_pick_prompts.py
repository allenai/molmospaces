"""Per-object prompt config for the semantic_grasp_pick prompt-level ablation.

Three prompt levels:
  1. Basic pick:        "pick up the {object}."  (matches the standard pick task)
  2. Existing semantic: today's production behavior in PromptSampler.
                        Pan: "pick up the hot {object}." All others:
                        "pick up the {object} to give to someone."
  3. Semantic + part:   "pick up the {object} by the {part}."

Keys are the raw `pickup_obj_name` prefix from the benchmark (the substring
before the first underscore), so the mapping is deterministic from the
benchmark and does not depend on the ObjectMeta description DB. The
``{object}`` placeholder is substituted at runtime with the rendered short
description (e.g. ``cookingpan`` -> ``pan``, ``boiler`` -> ``kettle``), which
preserves today's surface text for Level 2.

Categories covered match those present in the semantic_grasp_pick benchmark.
"""

# {object} is filled in at format time with the rendered object name.
# The Level 3 template additionally needs a `part` value supplied per-category.
LEVEL_TEMPLATES: dict[int, str] = {
    1: "pick up the {object}.",
    2: "pick up the {object} to give to someone.",
    3: "pick up the {object} by the {part}.",
}

# Per-object overrides. Any key omitted from a given level falls back to
# LEVEL_TEMPLATES[level].format(object=..., part=...).
SEMANTIC_PICK_PROMPTS: dict[str, dict[int, str]] = {
    "mug": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
    "knife": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
    "butterknife": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
    "ladle": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
    "fork": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
    "cookingpan": {
        1: "pick up the {object}.",
        2: "pick up the hot {object}.",
        3: "pick up the {object} by the handle.",
    },
    "spatula": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
    "boiler": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
    "spoon": {
        1: "pick up the {object}.",
        2: "pick up the {object} to give to someone.",
        3: "pick up the {object} by the handle.",
    },
}


VALID_PROMPT_LEVELS = (1, 2, 3)


def get_semantic_pick_prompt(category: str, object_name: str, level: int) -> str:
    """Render the prompt for a given object category and ablation level.

    Args:
        category: ``pickup_obj_name`` prefix (e.g. ``"cookingpan"``).
        object_name: The rendered object name to splice into ``{object}``
            (typically ``ObjectMeta.short_descriptions(uid)[0].lower()``).
        level: 1, 2, or 3.
    """
    if level not in VALID_PROMPT_LEVELS:
        raise ValueError(f"prompt_level must be one of {VALID_PROMPT_LEVELS}, got {level!r}")

    per_object = SEMANTIC_PICK_PROMPTS.get(category, {})
    template = per_object.get(level, LEVEL_TEMPLATES[level])
    return template.format(object=object_name, part="handle")
