import math
import os
from pathlib import Path
import zipfile

YAML = 'path: .\ntrain: images\n\nnames:\n  0: product\n'


def yolo_row(box, width, height):
    if width <= 0 or height <= 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Invalid box or image dimensions')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Box must have positive area within image boundaries')
    return '0 ' + ' '.join(f'{v:.8f}' for v in ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height))


def export_zip(images: list[dict], destination: Path):
    destination = Path(destination)
    if not images:
        raise ValueError('Cannot export a batch with no successful images')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.zip.tmp')
    names = set()
    try:
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('data.yaml', YAML)
            for im in images:
                name = im['name']
                if Path(name).name != name or Path(name).stem.casefold() in names:
                    raise ValueError('Export names must be unique simple filenames')
                names.add(Path(name).stem.casefold())
                rows = [yolo_row(d['bbox_xyxy_pixels'], im['width'], im['height'])
                        for d in im['detections'] if d['decision'] in ('retain', 'retain_uncertain')]
                z.write(im['path'], f'images/{name}')
                z.writestr(f'labels/{Path(name).stem}.txt', '\n'.join(rows) + ('\n' if rows else ''))
        os.replace(temp, destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return destination
