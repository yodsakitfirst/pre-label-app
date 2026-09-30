import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


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
                       'references': [(self.records[i]['sku_id'], self.records[i]['target'], r['sha256']) for i, r in self.references],
                       'crop_preparation': 'rgb_exif_v1'}
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        self.cache_path = Path(cache_dir) / f'{key}.npy'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
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
                    with Image.open(ref['path']) as image:
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
                    aggregate[idx] = (float(score), ref['source_name'] if 'source_name' in ref else ref['path'])
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
            results.append([{'sku_id': self.records[idx]['sku_id'], 'barcode': self.records[idx]['barcode'],
                             'name': self.records[idx]['name'], 'target': self.records[idx]['target'],
                             'score': score, 'score_type': 'cosine_similarity', 'reference': ref}
                            for idx, (score, ref) in shortlist])
        return results
