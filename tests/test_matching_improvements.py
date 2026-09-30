import io

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from prelabel.app import create_app
from prelabel.jobs import JobStore
from prelabel.policy import decide
from prelabel.retrieval import CatalogIndex


def test_reference_cleanup_removes_disconnected_heading_preserves_product():
    from prelabel.references import prepare_reference
    image = Image.new('RGB', (120, 180), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 119, 5), fill='black')
    draw.rectangle((45, 30, 75, 165), fill='magenta')
    cleaned, info = prepare_reference(image)
    assert cleaned.width < 45 and cleaned.height > 135
    assert info['crop_xyxy'][1] > 5
    assert np.asarray(cleaned)[:, :, 1].min() == 0
    # A shelf photograph with a dark background must not be component-cropped.
    photo = Image.new('RGB', (120, 180), 'gray')
    assert prepare_reference(photo)[0].size == photo.size


def candidates():
    return [{'sku_id': 'a', 'barcode': '001', 'target': True, 'score': .81},
            {'sku_id': 'b', 'barcode': '002', 'target': True, 'score': .80}]


def test_reference_cleanup_keeps_light_bottle_body_and_falls_back_for_invisible_body():
    from prelabel.references import prepare_reference
    image = Image.new('RGB', (200, 300), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 140, 280), fill=(250, 250, 250), outline=(247, 247, 247))
    draw.rectangle((80, 40, 120, 70), fill='red')
    cleaned, audit = prepare_reference(image)
    assert cleaned.height >= 240 and audit['crop_xyxy'][3] >= 280
    invisible = Image.new('RGB', (200, 300), 'white')
    ImageDraw.Draw(invisible).rectangle((80, 40, 120, 70), fill='red')
    assert prepare_reference(invisible)[0].size == (200, 300)


def test_similar_target_skus_retain_membership_without_assigning_exact_sku():
    evidence = {'candidates': candidates()}
    result = decide(evidence, config={'mode': 'reference_filter'})
    assert result['decision'] == 'retain'
    assert result['assigned_sku'] is None
    assert result['reference_gate']['sku_identity_ambiguous'] is True
    evidence['candidates'][1]['barcode'] = '001'
    assert decide(evidence, config={'mode': 'reference_filter'})['decision'] == 'retain'
    assert decide(evidence, config={'mode': 'reference_filter'})['reference_gate']['sku_identity_ambiguous'] is False


def test_membership_still_defers_low_scores_and_target_excluded_ambiguity():
    evidence = {'candidates': candidates()}
    assert decide(evidence, config={'mode': 'reference_filter', 'reference_min_similarity': .9})['decision'] == 'defer_review'
    evidence['candidates'][1]['target'] = False
    result = decide(evidence, config={'mode': 'reference_filter'})
    assert result['decision'] == 'defer_review'
    assert result['decision_reason'] == 'target_excluded_reference_ambiguity'


def test_runner_up_is_available_even_with_top_k_one(tmp_path):
    path = tmp_path / 'ref.png'
    Image.new('RGB', (10, 10), 'red').save(path)
    records = [{**c, 'name': c['sku_id'], 'references': [{'path': str(path), 'sha256': c['sku_id']}]} for c in candidates()]
    class Encoder:
        version = 'fixture'
        def encode(self, images):
            return np.array([[1., 0.] for _ in images])
    index = CatalogIndex([{'catalog_version': 'c', 'records': records}], Encoder(), tmp_path / 'cache')
    found = index.search([Image.new('RGB', (10, 10))], top_k=1, include_membership=True)[0]
    assert {c['barcode'] for c in found} == {'001', '002'}


def review_fixture(tmp_path):
    root = tmp_path / 'app'
    store = JobStore(root)
    path = tmp_path / 'photo.png'
    Image.new('RGB', (100, 100), 'red').save(path)
    record = {'sku_id': 'a', 'barcode': '001', 'name': 'Red product', 'target': True, 'references': []}
    cat = {'catalog_id': 'catalog', 'catalog_version': 'v1', 'source': 'fixture', 'target': True,
           'records': [record], 'warnings': []}
    from prelabel.files import write_json, digest
    write_json(root / 'catalogs' / 'catalog' / 'catalog.json', cat)
    config = {'mode': 'reference_filter', 'catalogs': [cat], 'reference_min_similarity': .75}
    job = store.create([{'path': str(path)}], config)
    detection = {'detection_id': 'box', 'bbox_xyxy_pixels': [10, 20, 40, 80],
                 'candidates': [candidates()[0]], 'decision': 'retain', 'decision_reason': 'fixture'}
    store.update(job['id'], status='completed', results={'0': {'path': str(path), 'name': 'photo.png',
                 'staged_sha256': digest(path), 'width': 100, 'height': 100, 'detections': [detection]}})
    return root, job['id']


def test_review_crops_feedback_negative_scope_and_validation(tmp_path):
    root, job_id = review_fixture(tmp_path)
    with TestClient(create_app(root, start_worker=False)) as client:
        headers = {'X-Prelabel-Token': client.get('/api/system').json()['token']}
        response = client.get(f'/api/jobs/{job_id}/matches')
        assert response.status_code == 200
        assert response.json()['boxes'][0]['detection_id'] == 'box'
        crop = client.get(f'/api/jobs/{job_id}/matches/box/crop')
        assert Image.open(io.BytesIO(crop.content)).size == (30, 60)
        endpoint = f'/api/jobs/{job_id}/matches/box/review'
        assert client.put(endpoint, json={'label': 'outside_catalog'}, headers=headers).status_code == 200
        from prelabel.review import reviewed_catalog
        cat = reviewed_catalog(root, ['catalog'])
        assert cat['records'][0]['target'] is False
        assert reviewed_catalog(root, ['another_catalog']) is None
        # Corrections must enter new job snapshots, not only the review report.
        checkpoint = root / 'model.pt'
        checkpoint.write_bytes(b'fixture')
        assert client.put('/api/settings', json={'checkpoint': str(checkpoint), 'encoder_path': 'fixture'}, headers=headers).status_code == 200
        source = str(tmp_path / 'photo.png')
        new_job = client.post('/api/jobs', json={'source': source, 'mode': 'retrieval', 'catalog_ids': ['catalog']}, headers=headers).json()
        saved = client.app.state.store.get(new_job['id'])
        assert saved['config']['catalogs'][-1]['records'][0]['target'] is False
        example_path = saved['config']['catalogs'][-1]['records'][0]['references'][0]['path']
        report = client.get(f'/api/jobs/{job_id}/validation').json()
        assert report['reviewed'] == 1
        row = next(r for r in report['thresholds'] if r['minimum_similarity'] == .75)
        assert row['false_accepts'] == 1 and row['true_accepts'] == 0
        assert client.put(endpoint, json={'label': 'correct', 'sku_id': 'missing'}, headers=headers).status_code == 422
        assert client.put(endpoint, json={'label': 'correct', 'sku_id': 'a'}, headers=headers).status_code == 200
        cat = reviewed_catalog(root, ['catalog'])
        assert cat['records'][0]['target'] is True and cat['records'][0]['barcode'] == '001'
        repaired = client.post('/api/jobs', json={'source': source, 'mode': 'reference_filter',
                                 'catalog_ids': ['catalog']}, headers=headers)
        assert repaired.status_code == 200
        assert client.delete(endpoint, headers=headers).status_code == 200
        assert reviewed_catalog(root, ['catalog']) is None
        # Clearing reviews cannot invalidate a previously queued job's references.
        assert Image.open(example_path).size == (30, 60)
        assert client.put('/api/settings', json={'checkpoint': str(checkpoint), 'encoder_path': 'fixture',
                          'use_reviewed_examples': False}, headers=headers).status_code == 200
        no_examples = client.post('/api/jobs', json={'source': source, 'mode': 'retrieval', 'catalog_ids': ['catalog']}, headers=headers).json()
        assert len(client.app.state.store.get(no_examples['id'])['config']['catalogs']) == 1
