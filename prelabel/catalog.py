import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

from .files import checked_members, digest, is_image, write_json


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate YAML key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def ingest_catalog(source: Path, destination: Path, target=True, **workbook_options) -> dict:
    source, destination = Path(source), Path(destination)
    source_hash = digest(source)
    selection = {'sheet': workbook_options.get('sheet'), 'header_row': workbook_options.get('header_row', 3),
                 'barcode_column': workbook_options.get('barcode_column', 'Barcode'),
                 'name_column': workbook_options.get('name_column', 'Product Name'),
                 'image_column': workbook_options.get('image_column', 'Product ImageRef')}
    version = hashlib.sha256(json.dumps([source_hash, target, selection], sort_keys=True).encode()).hexdigest() if source.suffix.lower() == '.xlsx' else (source_hash if target else hashlib.sha256((source_hash + ':excluded').encode()).hexdigest())
    destination = destination / version
    destination.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == '.xlsx':
        from .workbook import workbook_records
        raw_records = workbook_records(source, **workbook_options)
    else:
        with zipfile.ZipFile(source) as z:
            members = checked_members(z)
            manifests = [m.filename for m in members if Path(m.filename).name == 'catalog.yaml']
            if len(manifests) != 1:
                raise ValueError('Catalog ZIP needs exactly one catalog.yaml')
            manifest = manifests[0]
            root = str(Path(manifest).parent).replace('\\', '/')
            root = '' if root == '.' else root + '/'
            data = yaml.load(z.read(manifest).decode('utf-8-sig'), Loader=UniqueLoader)
            names = data.get('names') if isinstance(data, dict) else None
            if isinstance(names, list):
                names = dict(enumerate(names))
            if not isinstance(names, dict) or not names:
                raise ValueError('catalog.yaml must contain non-empty names')
            raw_records = []
            canonical_ids = set()
            for key, value in names.items():
                if isinstance(key, bool) or not re.fullmatch(r'\d+', str(key)) or not isinstance(value, str):
                    raise ValueError('Catalog names require numeric IDs and string labels')
                if str(key) in canonical_ids:
                    raise ValueError(f'Duplicate canonical catalog ID: {key}')
                canonical_ids.add(str(key))
                match = re.fullmatch(r'(\d+)\s+-\s+(.+)', value.strip())
                if not match:
                    raise ValueError(f'Catalog label {key} needs barcode - product name')
                refs = [m.filename for m in members if m.filename.startswith(root + 'references/')
                        and is_image(m.filename) and Path(m.filename).stem == str(key)]
                if not refs:
                    raise ValueError(f'Missing reference for catalog ID {key}')
                raw_records.append({'source_id': str(key), 'barcode': match[1], 'name': match[2],
                                    'images': [(n, z.read(n)) for n in sorted(refs)],
                                    'mapping_method': 'catalog_yaml_id_to_reference_stem'})
    records, warnings = [], []
    for raw in raw_records:
        sku_id = hashlib.sha256((version + ':' + raw['source_id']).encode()).hexdigest()[:24]
        refs = []
        for index, (original_name, pixels) in enumerate(raw.pop('images')):
            with Image.open(io.BytesIO(pixels)) as im:
                im.verify()
            name = f'{sku_id}_{index}' + Path(original_name).suffix.lower()
            path = destination / name
            path.write_bytes(pixels)
            refs.append({'path': str(path.resolve()), 'source_name': original_name, 'sha256': hashlib.sha256(pixels).hexdigest()})
        if not refs:
            warnings.append(f"No mapped image for {raw['source_id']} ({raw['barcode']}); excluded from retrieval")
        records.append({**raw, 'sku_id': sku_id, 'target': bool(target), 'references': refs})
    duplicates = [code for code, count in Counter(r['barcode'] for r in records).items() if count > 1]
    warnings.extend(f'Duplicate barcode {code}: records preserved separately' for code in duplicates)
    result = {'catalog_id': version, 'catalog_version': version, 'source_sha256': source_hash, 'source': source.name, 'target': target,
              'records': records, 'warnings': warnings, 'mapping_visually_verified': False}
    write_json(destination / 'catalog.json', result)
    return result
