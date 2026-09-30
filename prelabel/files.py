import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def checked_members(z: zipfile.ZipFile):
    members = z.infolist()
    if len(members) > 100000 or sum(m.file_size for m in members) > 20 * 1024**3:
        raise ValueError('Archive exceeds 100000 entries or 20 GiB expanded limit')
    seen = set()
    for m in members:
        name = m.filename.replace('\\', '/')
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or ':' in name or '\x00' in name:
            raise ValueError(f'Unsafe archive path: {name}')
        if (m.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f'Unsafe archive symlink: {name}')
        if name.casefold() in seen:
            raise ValueError(f'Duplicate archive entry: {name}')
        seen.add(name.casefold())
    return members


def is_image(name):
    p = PurePosixPath(str(name).replace('\\', '/'))
    return p.suffix.lower() in IMAGE_EXTENSIONS and '__MACOSX' not in p.parts and not p.name.startswith('._')
