"""Compare policies on saved evidence without rerunning YOLO or OCR."""
import json
import sqlite3
from pathlib import Path

from .fusion import fuse
from .policy import decide
from .review import load_review


def compare(root, job_id, engine, overrides=None):
    root = Path(root).resolve()
    with sqlite3.connect((root / 'jobs.sqlite3').as_uri() + '?mode=ro', uri=True) as conn:
        row = conn.execute('SELECT data FROM jobs WHERE id=?', (job_id,)).fetchone()
    if row is None:
        raise ValueError('Unknown job ID')
    job = json.loads(row[0])
    config = {**job['config'], **(overrides or {}), 'mode': 'reference_filter'}
    summary = {name: {'retained': 0, 'false_accepts': 0, 'true_accepts': 0,
                      'missed_targets': 0, 'correct_rejections': 0} for name in ('visual', 'rules', 'laya')}
    boxes, reviewed = [], 0
    for image in job['results'].values():
        for d in image['detections']:
            visual = decide(d, config=config, image_size=(image['width'], image['height']))
            decisions = {'visual': visual,
                         'rules': fuse(d, visual, {**config, 'fusion_backend': 'rules'}),
                         'laya': fuse(d, visual, {**config, 'fusion_backend': 'laya'}, engine)}
            review = load_review(root, job_id, d['detection_id'])
            reviewed += bool(review)
            for name, result in decisions.items():
                accepted = result['decision'] in ('retain', 'retain_uncertain')
                summary[name]['retained'] += accepted
                if review:
                    target = review['label'] == 'correct'
                    key = ('true_accepts' if target else 'false_accepts') if accepted else ('missed_targets' if target else 'correct_rejections')
                    summary[name][key] += 1
            boxes.append({'detection_id': d['detection_id'], 'image': image['name'],
                          'decisions': decisions, 'review': review})
    return {'job_id': job_id, 'boxes': boxes, 'summary': summary, 'reviewed': reviewed,
            'note': 'Replays saved OCR and visual scores with current policies. Accuracy counts use saved human reviews only; unreviewed boxes have no accuracy measurement. This is not a held-out benchmark.'}
