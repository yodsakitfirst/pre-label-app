import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from .files import checked_members, digest, is_image


def prepare_image(source: Path, destination: Path) -> dict:
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Source and destination must be different')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as original:
        orientation = original.getexif().get(274, 1)
        image = ImageOps.exif_transpose(original)
        image.load()
        if orientation not in (None, 1):
            # Export normalized pixels; strip orientation so importers cannot rotate twice.
            if image.mode not in ('RGB', 'L') and destination.suffix.lower() in ('.jpg', '.jpeg'):
                image = image.convert('RGB')
            image.save(destination)
        else:
            shutil.copyfile(source, destination)
        return {'path': str(destination.resolve()), 'width': image.width, 'height': image.height,
                'orientation_normalized': orientation not in (None, 1), 'source_sha256': digest(source)}


def import_images(source: Path, destination: Path) -> list[dict]:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.exists():
        raise ValueError('Image source does not exist')
    if source == destination:
        raise ValueError('Staging destination must differ from source')
    if source.is_dir():
        entries = [(str(p.relative_to(source)), p) for p in sorted(source.rglob('*'))
                   if p.is_file() and is_image(p.name) and destination not in p.resolve().parents]
    elif source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as z:
            members = checked_members(z)
            entries = [(m.filename, None) for m in members if not m.is_dir() and is_image(m.filename)]
    elif is_image(source.name):
        entries = [(source.name, source)]
    else:
        raise ValueError('Select an image folder, image ZIP, or image file')
    if not entries:
        raise ValueError('No supported images found')
    destination.mkdir(parents=True, exist_ok=True)
    used, results = set(), []
    z = zipfile.ZipFile(source) if source.suffix.lower() == '.zip' else None
    scratch = tempfile.TemporaryDirectory(prefix='prelabel-input-', dir=destination.parent) if z else None
    try:
        for relative, path in entries:
            raw_name = Path(relative.replace('\\', '/')).name
            stem = Path(raw_name).stem
            if stem.casefold() in used:
                stem += '_' + hashlib.sha256(relative.encode()).hexdigest()[:10]
                base, counter = stem, 2
                while stem.casefold() in used:
                    stem = f'{base}_{counter}'
                    counter += 1
            used.add(stem.casefold())
            name = stem + Path(raw_name).suffix.lower()
            raw = Path(scratch.name) / name if scratch else None
            if z:
                with z.open(relative) as src, raw.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                path = raw
            # Decoding failures stay visible in the batch rather than disappearing.
            try:
                result = prepare_image(path, destination / name)
            except Exception as exc:
                shutil.copyfile(path, destination / name)
                result = {'path': str(destination / name), 'input_error': str(exc), 'source_sha256': digest(path)}
            result.update({'name': name, 'source_name': relative})
            results.append(result)
            if z:
                raw.unlink()
    finally:
        if z:
            z.close()
            scratch.cleanup()
    return results
