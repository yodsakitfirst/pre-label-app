import zipfile

from PIL import Image
import pytest

from prelabel.policy import decide
from prelabel.exporter import export_zip
from prelabel.app import Settings, JobRequest, create_app
from fastapi.testclient import TestClient


def candidate(score, target=True):
    return {'sku_id': 'sku', 'barcode': '00123', 'score': score, 'target': target}


def test_strict_export_uses_target_evidence_not_detector_score():
    config = {'mode': 'reference_filter', 'reference_min_similarity': 0.65, 'excluded_margin': 0.03}
    good = {'detection_score': 0.1, 'candidates': [candidate(0.8)]}
    weak = {'detection_score': 0.99, 'candidates': [candidate(0.4)]}
    assert decide(good, config=config)['decision'] == 'retain'
    assert decide(weak, config=config)['decision'] == 'defer_review'
    assert decide(weak, config=config)['decision_reason'] == 'below_reference_export_threshold'


def test_missing_or_excluded_reference_is_deferred_not_called_non_target():
    config = {'mode': 'reference_filter', 'reference_min_similarity': 0.65, 'excluded_margin': 0.03}
    for evidence in ({'candidates': []}, {'candidates': [candidate(0.9, False)]},
                     {'candidates': [candidate(0.75), candidate(0.74, False)]}):
        assert decide(evidence, config=config)['decision'] == 'defer_review'
    assert decide({'candidates': [candidate(0.8), candidate(0.6, False)]}, config=config)['decision'] == 'retain'


def test_image_edge_filter_excludes_cut_products_but_not_adjacent_interiors():
    config = {'mode': 'baseline', 'full_product_only': True, 'edge_margin_fraction': 0.01}
    for box in ([0, 30, 10, 70], [30, 0, 50, 15], [95, 30, 100, 70], [30, 80, 50, 100]):
        result = decide({'bbox_xyxy_pixels': box}, baseline=True, config=config, image_size=(100, 100))
        assert result['decision'] == 'defer_review'
        assert result['decision_reason'] == 'possible_image_edge_truncation'
    for box in ([20, 30, 40, 80], [40, 30, 60, 80]):
        assert decide({'bbox_xyxy_pixels': box}, baseline=True, config=config, image_size=(100, 100))['decision'] == 'retain_uncertain'


def test_interior_box_is_not_claimed_to_prove_full_visibility():
    result = decide({'bbox_xyxy_pixels': [20, 30, 40, 80]}, baseline=True,
                    config={'full_product_only': True, 'edge_margin_fraction': 0.01}, image_size=(100, 100))
    assert result['completeness_evidence']['status'] == 'no_image_edge_risk'
    assert result['completeness_evidence']['occlusion_checked'] is False


def test_review_boxes_are_separate_from_main_export(tmp_path):
    path = tmp_path / 'shelf.png'
    Image.new('RGB', (100, 100)).save(path)
    detections = [{'bbox_xyxy_pixels': [10, 10, 30, 80], 'decision': 'retain'},
                  {'bbox_xyxy_pixels': [50, 10, 70, 80], 'decision': 'defer_review'}]
    images = [{'path': str(path), 'name': 'shelf.png', 'width': 100, 'height': 100, 'detections': detections}]
    main = export_zip(images, tmp_path / 'main.zip')
    review = export_zip(images, tmp_path / 'review.zip', stream='review')
    with zipfile.ZipFile(main) as z:
        assert z.read('labels/shelf.txt').decode().splitlines() == ['0 0.20000000 0.45000000 0.20000000 0.70000000']
    with zipfile.ZipFile(review) as z:
        assert z.read('labels/shelf.txt').decode().splitlines() == ['0 0.60000000 0.45000000 0.20000000 0.70000000']


def test_batch_defaults_to_reference_filter_with_complete_frame_preference():
    assert JobRequest(source='photos.zip').mode == 'reference_filter'
    assert Settings().full_product_only is True


def test_reference_filter_cannot_start_without_target_references(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'checkpoint')
    with TestClient(create_app(tmp_path / 'app', start_worker=False)) as client:
        token = client.get('/api/system').json()['token']
        headers = {'X-Prelabel-Token': token}
        assert client.put('/api/settings', json={'checkpoint': str(checkpoint)}, headers=headers).status_code == 200
        response = client.post('/api/jobs', json={'source': 'photos.zip', 'mode': 'reference_filter'}, headers=headers)
        assert response.status_code == 422
        assert 'target' in response.json()['detail'].lower()
