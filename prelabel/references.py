"""Conservative preparation of catalog canvases; originals remain untouched."""
from collections import deque

import numpy as np
from PIL import Image, ImageOps

PREPARATION_VERSION = 'white_canvas_components_v3'


def prepare_reference(original):
    rgba = ImageOps.exif_transpose(original).convert('RGBA')
    image = Image.new('RGB', rgba.size, 'white')
    image.paste(rgba, mask=rgba.getchannel('A'))
    small = image.copy()
    small.thumbnail((384, 384))
    pixels = np.asarray(small)
    white = np.all(pixels >= 245, axis=2)
    # Catalog headings often touch the top edge; assess the canvas on the other edges.
    border = np.concatenate((white[-1], white[:, 0], white[:, -1]))
    bounds = (0, 0, image.width, image.height)
    if border.mean() >= .9:
        edge_pixels = np.concatenate((pixels[-1], pixels[:, 0], pixels[:, -1]))
        background = np.median(edge_pixels, axis=0)
        mask = np.max(np.abs(pixels.astype(float) - background), axis=2) > 4
        components = []
        height, width = mask.shape
        for y, x in zip(*np.where(mask)):
            if not mask[y, x]:
                continue
            queue = deque([(int(x), int(y))])
            mask[y, x] = False
            count, left, top, right, bottom = 0, x, y, x, y
            while queue:
                a, b = queue.popleft()
                count += 1
                left, top, right, bottom = min(left, a), min(top, b), max(right, a), max(bottom, b)
                for nx, ny in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx]:
                        mask[ny, nx] = False
                        queue.append((nx, ny))
            # Wide, thin headings on a catalog canvas are not product views.
            if not (top < height * .15 and bottom-top < height * .08 and right-left > width * .5):
                components.append((count, int(left), int(top), int(right+1), int(bottom+1)))
        if components:
            largest = max(c[0] for c in components)
            kept = [c for c in components if c[0] >= largest * .15]
            if largest >= width * height * .01:
                sx, sy = image.width / width, image.height / height
                pad = max(2, round(min(image.size) * .02))
                bounds = (max(0, int(min(c[1] for c in kept)*sx)-pad),
                          max(0, int(min(c[2] for c in kept)*sy)-pad),
                          min(image.width, int(np.ceil(max(c[3] for c in kept)*sx))+pad),
                          min(image.height, int(np.ceil(max(c[4] for c in kept)*sy))+pad))
                # A lone colored cap/logo cannot establish a light product's silhouette.
                fractions = ((bounds[2]-bounds[0])/image.width, (bounds[3]-bounds[1])/image.height)
                coverage = fractions[1] if image.height > image.width*1.25 else fractions[0] if image.width > image.height*1.25 else max(fractions)
                if coverage < .35:
                    bounds = (0, 0, image.width, image.height)
    cleaned = image.crop(bounds)
    return cleaned, {'version': PREPARATION_VERSION, 'original_size': list(image.size),
                     'prepared_size': list(cleaned.size), 'crop_xyxy': list(bounds),
                     'cropped': bounds != (0, 0, image.width, image.height),
                     'low_resolution': min(cleaned.size) < 64,
                     'association_verified': False}
