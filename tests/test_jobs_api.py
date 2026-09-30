import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from prelabel.jobs import JobStore, JobRunner
from prelabel.app import create_app


class Detector:
    metadata = {'fixture': 'coordinate detector'}

    def detect(self, image):
        if image.getpixel((0, 0))[0] > 150:
            raise RuntimeError('Model failed on image')
        return [{'bbox_xyxy_pixels': [1, 2, 5, 8], 'detection_score': 0.7, 'detector_class_id': 0}]


def stage(tmp_path, count=2):
    images = []
    for n in range(count):
        path = tmp_path / f'{n}.png'
        Image.new('RGB', (10, 20), 'red' if n else 'blue').save(path)
        images.append({'path': str(path), 'name': path.name, 'width': 10, 'height': 20, 'source_sha256': str(n)})
    return images


def test_failures_are_reported_not_empty_predictions_and_resume(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path), {'mode': 'baseline', 'checkpoint': 'fixture'})
    runner = JobRunner(store, detector_factory=lambda config: Detector())
    runner.run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'completed_with_errors'
    assert result['completed'] == 1 and len(result['errors']) == 1
    with zipfile.ZipFile(result['export_path']) as z:
        assert 'labels/0.txt' in z.namelist()
        assert 'labels/1.txt' not in z.namelist()
    evidence = Path(result['evidence_path']).read_text()
    assert 'Model failed on image' in evidence and 'bbox_xyxy_pixels' in evidence
    assert store.resume(job['id'])['status'] == 'queued'
    runner.run_job(job['id'])
    assert store.get(job['id'])['completed'] == 1


def test_cancel_before_start_and_restart_interruption(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path, 1), {'mode': 'baseline'})
    store.cancel(job['id'])
    JobRunner(store, detector_factory=lambda config: Detector()).run_job(job['id'])
    assert store.get(job['id'])['status'] == 'cancelled'
    store.resume(job['id'])
    store.update(job['id'], status='running')
    second = JobStore(tmp_path)
    assert second.get(job['id'])['status'] == 'interrupted'
    assert second.resume(job['id'])['status'] == 'queued'


def test_missing_model_fails_explicitly(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path, 1), {'checkpoint': 'missing.pt', 'mode': 'baseline'})
    def failure(config):
        raise ValueError('checkpoint missing')
    JobRunner(store, detector_factory=failure).run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'failed'
    assert result['export_path'] is None
    assert 'checkpoint missing' in result['errors'][0]['error']


def test_api_local_mutation_guard_and_settings(tmp_path):
    app = create_app(tmp_path, start_worker=False)
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/api/system').status_code == 200
        assert client.put('/api/settings', json={'checkpoint': 'missing'}).status_code == 403
        token = client.get('/api/system').json()['token']
        headers = {'X-Prelabel-Token': token}
        assert client.put('/api/settings', json={'checkpoint': 'missing'}, headers=headers).status_code == 422
        assert client.get('/api/jobs').json() == []
        assert client.post('/api/jobs', json={'source': 'missing'}, headers=headers).status_code == 422


def test_bounded_queue(tmp_path):
    store = JobStore(tmp_path)
    for _ in range(8):
        store.create(stage(tmp_path, 1), {})
    with pytest.raises(ValueError, match='queue'):
        store.create(stage(tmp_path, 1), {})


def test_cancel_during_last_photo_reaches_resumable_state(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path, 1), {'mode': 'baseline'})
    class CancelDetector(Detector):
        def detect(self, image):
            store.cancel(job['id'])
            return super().detect(image)
    JobRunner(store, detector_factory=lambda config: CancelDetector()).run_job(job['id'])
    assert store.get(job['id'])['status'] == 'cancelled'
    assert store.resume(job['id'])['status'] == 'queued'


def test_bad_catalog_archive_is_user_input_error(tmp_path):
    bad = tmp_path / 'bad.zip'
    bad.write_bytes(b'not a zip')
    with TestClient(create_app(tmp_path / 'app', start_worker=False), raise_server_exceptions=False) as client:
        token = client.get('/api/system').json()['token']
        response = client.post('/api/catalogs', json={'source': str(bad)}, headers={'X-Prelabel-Token': token})
        assert response.status_code == 422
        assert 'archive' in response.json()['detail'].lower()


def test_resume_rejects_changed_model_version(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path, 1), {'mode': 'baseline'})
    JobRunner(store, detector_factory=lambda config: Detector()).run_job(job['id'])
    store.update(job['id'], status='interrupted')
    store.resume(job['id'])
    class ChangedDetector(Detector):
        metadata = {'fixture': 'new detector'}
    JobRunner(store, detector_factory=lambda config: ChangedDetector()).run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'failed'
    assert 'changed' in result['errors'][0]['error'].lower()


def test_resume_rejects_changed_completed_image(tmp_path):
    store = JobStore(tmp_path)
    images = stage(tmp_path, 1)
    job = store.create(images, {'mode': 'baseline'})
    JobRunner(store, detector_factory=lambda config: Detector()).run_job(job['id'])
    Image.new('RGB', (100, 200), 'blue').save(images[0]['path'])
    store.update(job['id'], status='interrupted')
    store.resume(job['id'])
    JobRunner(store, detector_factory=lambda config: Detector()).run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'failed'
    assert 'changed' in result['errors'][0]['error'].lower()


def test_worker_exports_edge_risk_in_separate_review_zip(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path, 1), {'mode': 'baseline', 'full_product_only': True, 'edge_margin_fraction': 0.01})
    class EdgeDetector(Detector):
        def detect(self, image):
            return [{'bbox_xyxy_pixels': [0, 2, 5, 8], 'detection_score': 0.7, 'detector_class_id': 0}]
    JobRunner(store, detector_factory=lambda config: EdgeDetector()).run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'completed'
    assert result['box_counts'] == {'retained': 0, 'deferred': 1, 'edge_risk': 1, 'removed': 0}
    with zipfile.ZipFile(result['export_path']) as z:
        assert z.read('labels/0.txt') == b''
    with zipfile.ZipFile(result['review_export_path']) as z:
        assert len(z.read('labels/0.txt').splitlines()) == 1
    assert Path(result['preview_path']).is_file()


def test_resume_clears_all_export_artifacts(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path, 1), {'mode': 'baseline'})
    store.update(job['id'], status='completed_with_errors', export_path='old.zip',
                 review_export_path='old_review.zip', preview_path='old.jpg', box_counts={'retained': 5})
    resumed = store.resume(job['id'])
    assert resumed['export_path'] is None
    assert resumed['review_export_path'] is None
    assert resumed['preview_path'] is None
    assert resumed['box_counts'] is None


def test_optional_preview_failure_preserves_completed_exports(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = store.create(stage(tmp_path, 1), {'mode': 'baseline'})
    def fail_preview(*args):
        raise OSError('preview unavailable')
    monkeypatch.setattr('prelabel.preview.save_preview', fail_preview)
    JobRunner(store, detector_factory=lambda config: Detector()).run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'completed'
    assert Path(result['export_path']).is_file()
    assert result['preview_path'] is None
    assert 'preview unavailable' in result['preview_warning']
