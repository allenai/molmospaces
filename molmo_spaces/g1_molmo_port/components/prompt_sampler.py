import gzip
import json

import ml_collections

from molmo_spaces.g1_molmo_port import ASSETS_DIR

TEMPLATES = [
    "pick up the {object}.",
    "grab the {object}.",
    "grasp the {object}.",
    "lift the {object}.",
    "take the {object}.",
    "get the {object}.",
]

OPEN_TEMPLATES = [
    "open the {object}.",
    "pull open the {object}.",
    "slide open the {object}.",
    "open up the {object}.",
]

CLOSE_TEMPLATES = [
    "close the {object}.",
    "push closed the {object}.",
    "slide closed the {object}.",
    "shut the {object}.",
]

_TEMPLATE_SETS = {"pick": TEMPLATES, "open": OPEN_TEMPLATES, "close": CLOSE_TEMPLATES}

_WORD_KEYS = ["one_word", "two_words", "three_words", "four_words", "five_words"]
_METADATA = None


def _load_metadata():
    global _METADATA
    if _METADATA is None:
        path = ASSETS_DIR / "objects" / "objathor_metadata" / "objects_metadata.json.gz"
        with gzip.open(path) as f:
            raw = json.load(f)
        # older versions: dict keyed by assetId; 20251117+: list of entries
        _METADATA = raw if isinstance(raw, dict) else {e["assetId"]: e for e in raw}
    return _METADATA


def get_object_name(asset_id, num_words=1):
    entry = _load_metadata().get(asset_id)
    if entry is None:
        return asset_id.split("_")[0].lower()
    key = _WORD_KEYS[min(num_words, len(_WORD_KEYS)) - 1]
    return entry.get("description_short", {}).get(key, asset_id).lower()


class PromptSampler:
    def __init__(self, config=None):
        config = config or get_config()
        mode = config.get("mode", "pick")
        templates = _TEMPLATE_SETS.get(mode, TEMPLATES)
        self.templates = templates if config.randomize else templates[:1]
        self.num_words = config.num_words

    def sample(self, asset_id, rng):
        name = get_object_name(asset_id, self.num_words)
        idx = int(rng.integers(len(self.templates)))
        return self.templates[idx].format(object=name)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            randomize=False,
            num_words=1,
            mode="pick",  # "pick" | "open" | "close" — selects template set
        )
    )
