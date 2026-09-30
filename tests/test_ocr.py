import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from prelabel.policy import decide


def test_text_ranking_preserves_visual_scores_and_handles_accents_and_low_scores():
    from prelabel.ocr import rank_candidates
    candidates = [{'barcode': '00123', 'name': 'TRESemmé Keratin Smooth 450ml', 'score': 0.66},
                  {'barcode': '00456', 'name': 'Sunsilk Smooth 450ml', 'score': 0.8}]
    ranking = rank_candidates(candidates, [{'text': 'TRESemme KERATIN', 'recognition_score': 0.95}], 0.7)
    assert ranking[0]['barcode'] == '00123'
    assert ranking[0]['score'] == 0.66
    assert ranking[0]['ocr_match']['matched_tokens'] == ['keratin', 'tresemme']
    assert candidates[0].get('ocr_match') is None
    assert rank_candidates(candidates, [{'text': 'TRESemme', 'recognition_score': 0.1}], 0.7)[0]['barcode'] == '00456'


def test_barcode_exact_match_is_not_substring_or_numeric_size_match():
    from prelabel.ocr import rank_candidates
    candidates = [{'barcode': '0012345678901', 'name': 'Product 450ml', 'score': 0.6}]
    assert rank_candidates(candidates, [{'text': '0012345678901', 'recognition_score': 0.99}], 0.7)[0]['ocr_match']['barcode_match']
    assert not rank_candidates(candidates, [{'text': '900123456789010', 'recognition_score': 0.99}], 0.7)[0]['ocr_match']['barcode_match']


def test_empty_ocr_does_not_change_export_policy():
    from prelabel.ocr import rank_candidates
    candidates = [{'barcode': '00123', 'name': 'Product', 'score': 0.6, 'target': True}]
    evidence = {'candidates': candidates, 'ocr_evidence': [], 'ocr_candidate_ranking': rank_candidates(candidates, [], 0.7)}
    assert decide(evidence, config={'mode': 'reference_filter', 'reference_min_similarity': 0.75})['decision'] == 'defer_review'


def test_strong_text_suggestion_does_not_bypass_visual_cutoff_or_assign_sku():
    from prelabel.ocr import rank_candidates
    candidates = [{'barcode': '0012345678901', 'name': 'Sunsilk Smooth', 'score': 0.6, 'target': True}]
    text = [{'text': '0012345678901 Sunsilk Smooth', 'recognition_score': 0.99}]
    ranking = rank_candidates(candidates, text, 0.7)
    assert ranking[0]['ocr_match']['barcode_match']
    result = decide({'candidates': candidates, 'ocr_candidate_ranking': ranking, 'ocr_evidence': text},
                    config={'mode': 'reference_filter', 'reference_min_similarity': 0.75})
    assert result['decision'] == 'defer_review'
    assert result['assigned_sku'] is None


def test_thai_marks_survive_text_normalization():
    from prelabel.ocr import normalize_text, rank_candidates
    text = 'สีเขียว'
    assert normalize_text(text) == text
    ranking = rank_candidates([{'barcode': '00123', 'name': 'ซันซิล สีเขียว', 'score': 0.6}],
                              [{'text': text, 'recognition_score': 0.9}])
    assert ranking[0]['ocr_match']['matched_tokens'] == [text]


def test_easyocr_adapter_uses_offline_thai_english_and_bgr_crop_coordinates(tmp_path, monkeypatch):
    from prelabel.ocr import EasyOCREngine
    for name in ['craft_mlt_25k.pth', 'thai.pth']:
        (tmp_path / name).write_bytes(b'fixture')
    def reader(languages, **kwargs):
        assert languages == ['th', 'en']
        assert kwargs['download_enabled'] is False
        assert kwargs['gpu'] is False
        def readtext(pixels, **opts):
            assert pixels[0, 0].tolist() == [0, 0, 255]
            return [([[0, 0], [10, 0], [10, 5], [0, 5]], 'แชมพู', 0.9)]
        return SimpleNamespace(readtext=readtext)
    monkeypatch.setitem(sys.modules, 'easyocr', SimpleNamespace(Reader=reader))
    monkeypatch.setattr('importlib.metadata.version', lambda name: 'fixture')
    result = EasyOCREngine(tmp_path).read(Image.new('RGB', (10, 5), 'red'))
    assert result[0]['text'] == 'แชมพู'
    assert result[0]['bbox_crop_pixels'][2] == [10.0, 5.0]


def test_easyocr_adapter_requires_local_models_before_loading(tmp_path):
    from prelabel.ocr import EasyOCREngine
    with pytest.raises(ValueError, match='download-ocr'):
        EasyOCREngine(tmp_path)


def test_worker_ocr_reads_all_boxes_up_to_limit_and_isolates_crop_failure(tmp_path, monkeypatch):
    from prelabel.jobs import JobStore, JobRunner
    path = tmp_path / 'photo.png'
    Image.new('RGB', (100, 100), 'red').save(path)
    class Detector:
        metadata = {'fixture': True}
        def detect(self, image):
            return [{'bbox_xyxy_pixels': [n * 20 + 1, 5, n * 20 + 15, 80], 'detection_score': 0.9,
                     'detector_class_id': 0} for n in range(3)]
    class OCR:
        version = {'fixture': True}
        calls = 0
        def read(self, image):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('bad crop')
            return [{'text': 'Sunsilk', 'recognition_score': 0.9}]
    monkeypatch.setattr('prelabel.ocr.create_ocr', lambda config: OCR())
    store = JobStore(tmp_path / 'runtime')
    job = store.create([{'path': str(path), 'name': path.name, 'source_sha256': 'fixture'}],
                       {'mode': 'baseline', 'ocr_enabled': True, 'ocr_max_crops': 2})
    JobRunner(store, detector_factory=lambda config: Detector()).run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'completed'
    detections = result['results']['0']['detections']
    assert [d['ocr_status'] for d in detections] == ['failed', 'completed', 'skipped_limit']
    assert 'bad crop' in detections[0]['ocr_error']
    assert result['ocr_counts'] == {'attempted': 2, 'with_text': 1, 'failed': 1, 'skipped': 1}
    report = json.loads(open(result['ocr_report_path'], encoding='utf-8').read())
    assert report['images'][0]['boxes'][1]['ocr_evidence'][0]['text'] == 'Sunsilk'


def test_api_ocr_report_and_missing_models_validation(tmp_path):
    from fastapi.testclient import TestClient
    from prelabel.app import create_app
    from prelabel.files import write_json
    root = tmp_path / 'runtime'
    app = create_app(root, start_worker=False)
    path = tmp_path / 'model.pt'
    path.write_bytes(b'model')
    with TestClient(app) as client:
        headers = {'X-Prelabel-Token': client.get('/api/system').json()['token']}
        response = client.put('/api/settings', headers=headers,
                              json={'checkpoint': str(path), 'ocr_enabled': True, 'ocr_model_dir': str(tmp_path / 'missing')})
        assert response.status_code == 422
        assert 'download-ocr' in response.json()['detail']
        job = app.state.store.create([{'name': 'photo.jpg'}], {'mode': 'baseline'})
        assert client.get(f"/api/jobs/{job['id']}/ocr").status_code == 404
        report_path = root / 'report.json'
        write_json(report_path, {'images': [{'name': 'photo.jpg', 'boxes': []}]})
        app.state.store.update(job['id'], ocr_report_path=str(report_path))
        response = client.get(f"/api/jobs/{job['id']}/ocr")
        assert response.status_code == 200
        assert response.json()['images'][0]['name'] == 'photo.jpg'
        response = client.get(f"/api/jobs/{job['id']}/download/ocr")
        assert 'ocr.json' in response.headers['content-disposition']


def test_existing_onnx_settings_and_jobs_keep_rapidocr_backend(tmp_path, monkeypatch):
    from prelabel.app import Settings
    from prelabel.ocr import create_ocr
    path = tmp_path / 'model'
    path.write_bytes(b'fixture')
    old_config = {'ocr_enabled': True, 'ocr_detection_model': str(path),
                  'ocr_recognition_model': str(path), 'ocr_dictionary': str(path)}
    assert Settings(**old_config).ocr_backend == 'rapidocr'
    monkeypatch.setattr('prelabel.models.LocalOCR', lambda *args: 'legacy reader')
    assert create_ocr(old_config) == 'legacy reader'
