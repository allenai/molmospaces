import hashlib

import numpy as np
from PIL import Image

# PromptSampler moved to molmo_spaces.utils.prompt_sampler.PromptSamplerLearnedPolicy
# (renamed to sit next to PromptSamplerSimple, gold's own template-based
# sampler, without the two names colliding -- see that module for what
# distinguishes this one).


def resize_with_pad(images, height, width):
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(Image.fromarray(im), height, width, method=Image.BILINEAR)
            for im in images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image, height, width, method):
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def generate_object_hash(asset_id: str) -> str:
    hasher = hashlib.md5()
    hasher.update(asset_id.encode())
    return hasher.hexdigest()
