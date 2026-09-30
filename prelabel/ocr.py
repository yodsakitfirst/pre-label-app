"""Local packaging text extraction and inspectable candidate suggestions.

Text ranking is separate from visual membership policy. No text means no evidence,
not proof that a product is absent from the catalog.
"""
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import re
import unicodedata

import numpy as np

from .files import digest


EASY_MODEL_FILES = ('craft_mlt_25k.pth', 'thai.pth')


def backend_name(config):
    if config.get('ocr_backend'):
        return config['ocr_backend']
    return 'rapidocr' if config.get('ocr_detection_model') or config.get('ocr_recognition_model') else 'easyocr'


def validate_ocr(config):
    if backend_name(config) == 'easyocr':
        directory = Path(config.get('ocr_model_dir') or '__missing_ocr_models__')
        if not all((directory / name).is_file() for name in EASY_MODEL_FILES):
            raise ValueError('Local Thai/English OCR models missing; run python -m prelabel download-ocr')
    else:
        if not all(Path(config.get(key) or '__missing_ocr_model__').is_file()
                   for key in ('ocr_detection_model', 'ocr_recognition_model', 'ocr_dictionary')):
            raise ValueError('Select local OCR detection/recognition models and dictionary')


class EasyOCREngine:
    def __init__(self, model_dir):
        validate_ocr({'ocr_backend': 'easyocr', 'ocr_model_dir': str(model_dir)})
        from easyocr import Reader
        directory = Path(model_dir).resolve()
        # CPU is deliberate: portable and independent of the detector's GPU setting.
        # Never auto-download during batch inference.
        self.reader = Reader(['th', 'en'], gpu=False, model_storage_directory=str(directory),
                             user_network_directory=str(directory / 'networks'),
                             download_enabled=False, verbose=False)
        self.version = {'backend': 'easyocr', 'languages': ['th', 'en'], 'device': 'cpu',
                        'runtime': importlib.metadata.version('easyocr'),
                        'models': {name: digest(directory / name) for name in EASY_MODEL_FILES},
                        'preprocessing': 'pil_rgb_to_bgr_crop_v1'}

    def read(self, image):
        pixels = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
        rows = self.reader.readtext(pixels, detail=1, paragraph=False,
                                    batch_size=1, workers=0, canvas_size=1024, mag_ratio=1.5)
        return [{'text': str(text), 'recognition_score': float(score),
                 'bbox_crop_pixels': [[float(x), float(y)] for x, y in box]}
                for box, text, score in rows]


def create_ocr(config):
    validate_ocr(config)
    if backend_name(config) == 'easyocr':
        return EasyOCREngine(config['ocr_model_dir'])
    from .models import LocalOCR
    return LocalOCR(config['ocr_detection_model'], config['ocr_recognition_model'], config['ocr_dictionary'])


def normalize_text(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    # Fold accented Latin brand names; keep Thai combining marks intact.
    text = ''.join(''.join(c for c in unicodedata.normalize('NFKD', char)
                           if unicodedata.category(c) != 'Mn')
                   if 'LATIN' in unicodedata.name(char, '') else char for char in text)
    return ''.join(c if unicodedata.category(c)[0] in 'LMN' else ' ' for c in text)


def rank_candidates(candidates, evidence, min_score=0.7):
    readable = [normalize_text(row['text']) for row in evidence
                if row.get('recognition_score', 0) >= min_score]
    text = ' '.join(readable)
    tokens = {token for token in text.split() if len(token) >= 3 and not token.isdecimal()}
    tokens -= {'shampoo', 'conditioner', 'product', 'bottle', 'pack', 'new', 'the', 'and', 'for'}
    numbers = set(re.findall(r'\d+', text))
    result = deepcopy(candidates)
    for candidate in result:
        name = normalize_text(candidate.get('name', ''))
        matched = sorted(token for token in tokens if token in name.split())
        barcode = str(candidate.get('barcode', ''))
        candidate['ocr_match'] = {'matched_tokens': matched,
                                  'barcode_match': len(barcode) >= 8 and barcode in numbers,
                                  'readable_lines': len(readable), 'minimum_recognition_score': min_score,
                                  'verified_sku': False}
    return sorted(result, key=lambda c: (c['ocr_match']['barcode_match'],
                                        len(c['ocr_match']['matched_tokens']), c['score']), reverse=True)


def make_report(results):
    return {'text_ranking_changes_export': False,
            'fusion_changes_export': any(d.get('fusion_evidence', {}).get('applied') for r in results for d in r['detections']),
            'images': [
        {'name': r['name'], 'boxes': [
            {key: d.get(key) for key in ('detection_id', 'bbox_xyxy_pixels', 'decision', 'decision_reason',
                                       'ocr_status', 'ocr_error', 'ocr_evidence', 'ocr_candidate_ranking', 'fusion_evidence')}
            for d in r['detections']]} for r in results]}
