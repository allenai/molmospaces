import numpy as np
from PIL import Image, ImageDraw


def speckle_texture(
    base_color,
    size=256,
    noise_strength=0.1,
    num_blobs=80,
    blob_size_range=(5, 25),
    blob_variation=0.1,
) -> Image.Image:
    img = np.ones((size, size, 3)) * np.array(base_color)
    noise = np.random.normal(0, noise_strength, (size, size, 1))
    img += noise
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw chunky rectangular or elliptical blobs
    for _ in range(num_blobs):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        w = np.random.randint(*blob_size_range)
        h = np.random.randint(*blob_size_range)

        variation = np.random.uniform(-blob_variation, blob_variation)
        blob_color = tuple(int(np.clip((c + variation) * 255, 0, 255)) for c in base_color)

        if np.random.random() > 0.5:
            draw.ellipse((x, y, x + w, y + h), fill=blob_color)
        else:
            draw.rectangle((x, y, x + w, y + h), fill=blob_color)

    return pil_img
