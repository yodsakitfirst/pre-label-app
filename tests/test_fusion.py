from prelabel.fusion import fuse


class Engine:
    def __init__(self, choice='exclude', probability=.98):
        self.choice, self.probability, self.calls = choice, probability, 0

    def predict(self, state):
        self.calls += 1
        return {'choice': self.choice, 'probabilities': {self.choice: self.probability}}


def box(text='Pantene', score=.8):
    return {'candidates': [{'name': 'ซันซิล แชมพู', 'barcode': '8851234567890',
                            'score': score, 'target': True}],
            'ocr_evidence': [{'text': text, 'recognition_score': .95}]}


def test_conflicting_brand_can_reject_but_weak_text_cannot():
    engine = Engine()
    visual = {'decision': 'retain', 'decision_reason': 'meets_reference_export_threshold'}
    config = {'fusion_backend': 'laya'}
    result = fuse(box(), visual, config, engine)
    assert result['decision'] == 'defer_review'
    assert result['fusion_evidence']['simple_rule'] == 'exclude'
    result = fuse(box('ca +1 99'), visual, config, engine)
    assert result['decision'] == 'retain'
    assert engine.calls == 1


def test_no_ocr_and_model_failure_fall_back_without_removing_boxes():
    visual = {'decision': 'retain', 'decision_reason': 'meets_reference_export_threshold'}
    assert fuse(box(), visual, {'fusion_backend': 'laya'}, None)['decision'] == 'retain'
    class Broken:
        def predict(self, state):
            raise RuntimeError('offline failure')
    result = fuse(box(), visual, {'fusion_backend': 'laya'}, Broken())
    assert result['decision'] == 'retain'
    assert result['fusion_evidence']['status'] == 'failed'


def test_barcode_rescue_and_hard_filters():
    visual = {'decision': 'defer_review', 'decision_reason': 'below_reference_export_threshold'}
    result = fuse(box('8851234567890', .72), visual,
                  {'fusion_backend': 'laya', 'reference_min_similarity': .75}, Engine('include'))
    assert result['decision'] == 'retain'
    for reason in ('possible_image_edge_truncation', 'validated_excluded_product'):
        decision = {'decision': 'defer_review', 'decision_reason': reason}
        assert fuse(box('8851234567890'), decision, {'fusion_backend': 'laya'}, Engine('include')) == decision


def test_ambiguous_target_skus_and_uncertain_laya_keep_visual():
    visual = {'decision': 'retain', 'decision_reason': 'meets_reference_export_threshold'}
    result = fuse(box('Pantene'), visual, {'fusion_backend': 'laya'}, Engine('uncertain'))
    assert result['decision'] == 'retain'
    assert result['fusion_evidence']['applied'] is False


def test_target_brand_elsewhere_in_catalog_is_not_rejected():
    visual = {'decision': 'retain', 'decision_reason': 'meets_reference_export_threshold'}
    config = {'fusion_backend': 'rules', 'catalogs': [{'records': [
        {'name': 'Pantene shampoo', 'target': True}, {'name': 'Sunsilk shampoo', 'target': True}]}]}
    assert fuse(box(), visual, config)['decision'] == 'retain'


def test_unknown_brand_in_target_catalog_prevents_brand_only_exclusion():
    visual = {'decision': 'retain', 'decision_reason': 'meets_reference_export_threshold'}
    config = {'fusion_backend': 'rules', 'catalogs': [{'records': [
        {'name': 'Sunsilk shampoo', 'target': True}, {'name': 'แชมพู สูตรบำรุง', 'target': True}]}]}
    result = fuse(box(), visual, config)
    assert result['decision'] == 'retain'
    assert result['fusion_evidence']['simple_rule'] == 'uncertain'


def test_brand_alone_cannot_rescue_low_similarity_or_weak_answer():
    visual = {'decision': 'defer_review', 'decision_reason': 'below_reference_export_threshold'}
    for text in ('Sunsilk', 'ซันซิล', 'ซันซิล แชมพู'):
        result = fuse(box(text, .72), visual, {'fusion_backend': 'rules'})
        assert result['decision'] == 'defer_review'
    result = fuse(box('8851234567890', .72), visual, {'fusion_backend': 'laya'}, Engine('include', .6))
    assert result['decision'] == 'defer_review'
    for text in ('Head and Shoulders', 'Herbal Essences'):
        evidence = box(text, .72)
        evidence['candidates'][0]['name'] = text + ' anti dandruff 400ml'
        assert fuse(evidence, visual, {'fusion_backend': 'rules'})['decision'] == 'defer_review'


def test_worker_fusion_changes_export_and_keeps_one_zip(tmp_path, monkeypatch):
    import zipfile
    from PIL import Image
    from prelabel.jobs import JobRunner, JobStore
    class Detector:
        metadata = {'fixture': True}
        def detect(self, image):
            return [{'bbox_xyxy_pixels': [10, 10, 40, 90], 'detection_score': .9}]
    class Encoder:
        version = {'fixture': True}
        def __init__(self, *args): pass
    class Index:
        cache_path = tmp_path / 'index.json'
        def __init__(self, *args, **kwargs): pass
        def search(self, crops, *args, **kwargs): return [box()['candidates'] for _ in crops]
    class OCR:
        version = {'fixture': True}
        def read(self, image): return box()['ocr_evidence']
    class Laya(Engine):
        version = {'fixture': True}
        closed = False
        def close(self): self.closed = True
    engine = Laya()
    monkeypatch.setattr('prelabel.models.SiglipEncoder', Encoder)
    monkeypatch.setattr('prelabel.retrieval.CatalogIndex', Index)
    monkeypatch.setattr('prelabel.ocr.create_ocr', lambda config: OCR())
    monkeypatch.setattr('prelabel.laya.LayaEngine', lambda config: engine)
    path = tmp_path / 'photo.png'
    Image.new('RGB', (100, 100)).save(path)
    store = JobStore(tmp_path / 'runtime')
    job = store.create([{'name': 'photo.png', 'path': str(path), 'source_sha256': 'fixture'}],
                       {'mode': 'reference_filter', 'encoder_path': 'fixture', 'ocr_enabled': True,
                        'fusion_backend': 'laya', 'catalogs': [{'catalog_version': 'fixture', 'records': [
                            {'name': 'Sunsilk shampoo', 'target': True}]}]})
    JobRunner(store, detector_factory=lambda config: Detector()).run_job(job['id'])
    result = store.get(job['id'])
    assert result['status'] == 'completed'
    assert result['box_counts']['retained'] == 0
    assert result['fusion_counts']['changed'] == 1
    assert engine.closed
    assert len(list(__import__('pathlib').Path(result['directory']).glob('*.zip'))) == 1
    with zipfile.ZipFile(result['export_path']) as archive:
        assert archive.read('labels/photo.txt') == b''
