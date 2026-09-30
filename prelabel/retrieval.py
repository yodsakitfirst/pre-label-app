import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from .references import prepare_reference, PREPARATION_VERSION


def normalize(vectors):
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2 or not np.isfinite(v).all() or np.any(np.linalg.norm(v, axis=1) == 0):
        raise ValueError('Encoder returned invalid embeddings')
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class CatalogIndex:
    def __init__(self, catalogs, encoder, cache_dir, batch_size=16, on_progress=None):
        self.encoder = encoder
        self.records, self.references = [], []
        for cat in catalogs:
            for record in cat['records']:
                if record['references']:
                    idx = len(self.records)
                    self.records.append(record)
                    self.references.extend((idx, ref) for ref in record['references'])
        if not self.references:
            raise ValueError('Catalogs contain no mapped reference images')
        fingerprint = {'encoder': encoder.version, 'catalogs': [c['catalog_version'] for c in catalogs],
                       'references': [(self.records[i]['sku_id'], self.records[i]['target'], r['sha256'], r.get('shelf_example', False)) for i, r in self.references],
                       'crop_preparation': PREPARATION_VERSION}
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        self.cache_path = Path(cache_dir) / f'{key}.npy'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the exact prepared view for inspection, including on cache hits.
        prepared = []
        for idx, ref in self.references:
            style = 'shelf_v1' if ref.get('shelf_example') else PREPARATION_VERSION
            view_path = self.cache_path.parent / 'references' / f"{ref['sha256']}_{style}.png"
            audit_path = view_path.with_suffix('.json')
            if not view_path.exists() or not audit_path.exists():
                with Image.open(ref['path']) as original:
                    if ref.get('shelf_example'):
                        view = ImageOps.exif_transpose(original).convert('RGB')
                        audit = {'version': 'reviewed_shelf_crop_v1', 'prepared_size': list(view.size),
                                 'low_resolution': min(view.size) < 64, 'cropped': False}
                    else:
                        view, audit = prepare_reference(original)
                view_path.parent.mkdir(parents=True, exist_ok=True)
                view.save(view_path)
                audit_path.write_text(json.dumps(audit), encoding='utf-8')
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            prepared.append((idx, {**ref, 'prepared_path': str(view_path.resolve()), 'preparation': audit}))
        self.references = prepared
        self.cache_hit = self.cache_path.exists()
        if self.cache_hit:
            self.vectors = normalize(np.load(self.cache_path, allow_pickle=False))
            if len(self.vectors) != len(self.references):
                raise ValueError('Invalid catalog cache row count')
        else:
            batches = []
            if on_progress:
                on_progress(0, len(self.references))
            for offset in range(0, len(self.references), batch_size):
                images = []
                for _, ref in self.references[offset:offset + batch_size]:
                    with Image.open(ref['prepared_path']) as image:
                        images.append(ImageOps.exif_transpose(image).convert('RGB'))
                batches.append(normalize(encoder.encode(images)))
                if on_progress:
                    on_progress(min(offset + batch_size, len(self.references)), len(self.references))
            self.vectors = np.concatenate(batches)
            temp = self.cache_path.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.save(f, self.vectors)
            os.replace(temp, self.cache_path)

    def search(self, images, top_k=5, include_membership=False):
        if not images:
            return []
        similarity = normalize(self.encoder.encode(images)) @ self.vectors.T
        results = []
        for scores in similarity:
            # Max over views, then shortlist distinct canonical source records.
            aggregate = {}
            for (idx, ref), score in zip(self.references, scores):
                if idx not in aggregate or score > aggregate[idx][0]:
                    aggregate[idx] = (float(score), ref)
            ranked = sorted(aggregate.items(), key=lambda pair: pair[1][0], reverse=True)
            shortlist = ranked[:top_k]
            if include_membership:
                # A strong excluded reference must remain visible to the gate even outside Top-K.
                selected = {idx for idx, _ in shortlist}
                for role in (True, False):
                    best = next((pair for pair in ranked if self.records[pair[0]]['target'] is role), None)
                    if best and best[0] not in selected:
                        shortlist.append(best)
                        selected.add(best[0])
                best_target = next((pair for pair in ranked if self.records[pair[0]]['target']), None)
                if best_target:
                    code = self.records[best_target[0]]['barcode']
                    rival = next((pair for pair in ranked if self.records[pair[0]]['target'] and self.records[pair[0]]['barcode'] != code), None)
                    if rival and rival[0] not in selected:
                        shortlist.append(rival)
            results.append([{'sku_id': self.records[idx]['sku_id'], 'barcode': self.records[idx]['barcode'],
                             'name': self.records[idx]['name'], 'target': self.records[idx]['target'],
                             'score': score, 'score_type': 'cosine_similarity',
                             'reference': ref.get('source_name', ref['path']),
                             'reference_path': ref['path'], 'prepared_reference_path': ref['prepared_path'],
                             'reference_preparation': ref['preparation']}
                            for idx, (score, ref) in shortlist])
        return results
