"""Human-reviewed examples and diagnostics; never silently relabel old exports."""
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from .files import digest, write_json
from .policy import decide


def find_box(job, detection_id):
    for result in job['results'].values():
        for box in result['detections']:
            if box['detection_id'] == detection_id:
                return result, box
    raise KeyError(detection_id)


def review_path(root, job_id, detection_id):
    key = hashlib.sha256(f'{job_id}:{detection_id}'.encode()).hexdigest()
    return Path(root) / 'reviews' / f'{key}.json'


def load_review(root, job_id, detection_id):
    path = review_path(root, job_id, detection_id)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def crop_box(result, box):
    if result.get('staged_sha256') and digest(Path(result['path'])) != result['staged_sha256']:
        raise ValueError('Photo pixels changed since processing; create a new batch')
    with Image.open(result['path']) as image:
        return image.convert('RGB').crop(tuple(box['bbox_xyxy_pixels']))


def save_review(root, job, detection_id, label, sku_id=None):
    result, box = find_box(job, detection_id)
    record, catalog_id = None, None
    catalogs = [c for c in job['config'].get('catalogs', []) if not c.get('reviewed_examples')]
    if label == 'correct':
        for cat in catalogs:
            for r in cat['records']:
                if r['sku_id'] == sku_id and r['target']:
                    record, catalog_id = r, cat['catalog_id']
        if record is None:
            raise ValueError('Select the correct SKU from this batch’s target catalogs')
    path = review_path(root, job['id'], detection_id)
    crop = crop_box(result, box)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = io.BytesIO()
    crop.save(data, format='PNG')
    crop_hash = hashlib.sha256(data.getvalue()).hexdigest()
    crop_path = path.parent / 'images' / f'{crop_hash}.png'
    crop_path.parent.mkdir(exist_ok=True)
    if not crop_path.exists():
        crop_path.write_bytes(data.getvalue())
    value = {'job_id': job['id'], 'detection_id': detection_id, 'label': label, 'sku_id': sku_id if record else None,
             'catalog_id': catalog_id, 'catalog_scope': sorted(c['catalog_id'] for c in catalogs),
             'barcode': record['barcode'] if record else None, 'name': record['name'] if record else None,
             'path': str(crop_path.resolve()), 'sha256': crop_hash,
             'image_sha256': digest(Path(result['path']))}
    write_json(path, value)
    return value


def reviewed_catalog(root, selected_ids):
    selected = sorted(selected_ids)
    records, versions = [], []
    for path in sorted((Path(root) / 'reviews').glob('*.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        positive = value['label'] == 'correct' and value['catalog_id'] in selected
        negative = value['label'] == 'outside_catalog' and value['catalog_scope'] == selected
        if not (positive or negative):
            continue
        crop_path = Path(value['path'])
        if not crop_path.is_file() or digest(crop_path) != value['sha256']:
            raise ValueError('A reviewed example is missing or changed; clear or re-save that review')
        versions.append(digest(path))
        records.append({'sku_id': value['sku_id'] if positive else 'outside_' + path.stem,
                        'barcode': value['barcode'] if positive else 'outside_' + path.stem,
                        'name': value['name'] if positive else 'Reviewed product outside selected catalogs',
                        'target': positive, 'references': [{'path': value['path'], 'sha256': value['sha256'],
                                                           'source_name': f"review:{value['job_id']}:{value['detection_id']}",
                                                           'shelf_example': True}]})
    if not records:
        return None
    version = hashlib.sha256(json.dumps(versions).encode()).hexdigest()
    return {'catalog_id': 'reviewed_examples', 'catalog_version': version, 'reviewed_examples': True,
            'source': 'Human-reviewed shelf examples', 'records': records}


def validation_report(root, job):
    reviewed = []
    for result in job['results'].values():
        for box in result['detections']:
            review = load_review(root, job['id'], box['detection_id'])
            if review:
                reviewed.append((result, box, review))
    rows = []
    thresholds = sorted({.5, .6, .65, .7, .75, .8, .85, .9, job['config'].get('reference_min_similarity', .75)})
    for threshold in thresholds:
        row = {'minimum_similarity': threshold, 'true_accepts': 0, 'false_accepts': 0, 'missed_targets': 0,
               'correct_rejections': 0, 'wrong_sku_accepts': 0}
        for result, box, review in reviewed:
            config = {**job['config'], 'mode': 'reference_filter', 'reference_min_similarity': threshold}
            decision = decide(box, config=config, image_size=(result['width'], result['height']))
            accepted = decision['decision'] == 'retain'
            target = review['label'] == 'correct'
            key = ('true_accepts' if target else 'false_accepts') if accepted else ('missed_targets' if target else 'correct_rejections')
            row[key] += 1
            best = decision.get('reference_gate', {}).get('best_target')
            row['wrong_sku_accepts'] += bool(accepted and target and best and best['barcode'] != review['barcode'])
        accepted = row['true_accepts'] + row['false_accepts']
        targets = row['true_accepts'] + row['missed_targets']
        row['membership_precision'] = row['true_accepts'] / accepted if accepted else None
        row['target_recall'] = row['true_accepts'] / targets if targets else None
        rows.append(row)
    return {'job_id': job['id'], 'reviewed': len(reviewed), 'thresholds': rows, 'calibrated': False,
            'note': 'Replays this batch’s saved visual scores against your reviews. Selection bias and reuse of reference examples can inflate results. Validate on separate photos before choosing defaults.'}
