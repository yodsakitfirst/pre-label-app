import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from .exporter import export_zip, yolo_row
from .files import write_json, digest
from .policy import decide, resolve_packs


class JobCancelled(Exception):
    pass


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'jobs.sqlite3'
        self.lock = threading.RLock()
        with self.connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            for row in conn.execute('SELECT id,data FROM jobs').fetchall():
                data = json.loads(row[1])
                if data['status'] in ('running', 'cancelling'):
                    data['status'] = 'interrupted'
                    conn.execute('UPDATE jobs SET data=? WHERE id=?', (json.dumps(data), row[0]))

    def connect(self):
        return sqlite3.connect(self.db, timeout=30)

    def list(self):
        with self.connect() as conn:
            return [json.loads(row[0]) for row in conn.execute('SELECT data FROM jobs ORDER BY rowid DESC')]

    def get(self, job_id):
        with self.connect() as conn:
            row = conn.execute('SELECT data FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def create(self, images, config):
        with self.lock:
            if sum(j['status'] in ('queued', 'running', 'cancelling') for j in self.list()) >= 8:
                raise ValueError('The bounded job queue is full (8 jobs)')
            if not images:
                raise ValueError('Job requires images')
            job_id = uuid.uuid4().hex
            directory = self.root / 'jobs' / job_id
            directory.mkdir(parents=True)
            job = {'id': job_id, 'status': 'queued', 'created_at': time.time(), 'updated_at': time.time(),
                   'config': config, 'images': images, 'total': len(images), 'completed': 0, 'processed': 0,
                   'errors': [], 'results': {}, 'export_path': None, 'evidence_path': None,
                   'directory': str(directory), 'elapsed_seconds': 0}
            with self.connect() as conn:
                conn.execute('INSERT INTO jobs VALUES (?,?)', (job_id, json.dumps(job)))
            return job

    def update(self, job_id, **changes):
        with self.lock:
            data = self.get(job_id)
            data.update(changes, updated_at=time.time())
            with self.connect() as conn:
                conn.execute('UPDATE jobs SET data=? WHERE id=?', (json.dumps(data), job_id))
            return data

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] == 'queued':
                return self.update(job_id, status='cancelled')
            if job['status'] == 'running':
                return self.update(job_id, status='cancelling')
            return job

    def resume(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] not in ('cancelled', 'interrupted', 'failed', 'completed_with_errors'):
                raise ValueError('Only cancelled, interrupted, failed or incomplete jobs can resume')
            if sum(j['status'] in ('queued', 'running', 'cancelling') for j in self.list()) >= 8:
                raise ValueError('The bounded job queue is full')
            return self.update(job_id, status='queued', errors=[], processed=job['completed'], export_path=None,
                               review_export_path=None, preview_path=None, preview_warning=None, box_counts=None,
                               ocr_report_path=None, ocr_counts=None, fusion_counts=None, fusion_warning=None)


def default_detector(config):
    from .models import YoloDetector
    path = Path(config['checkpoint'])
    if config.get('checkpoint_sha256') and digest(path) != config['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since job creation; start a new job')
    return YoloDetector(path, config.get('device', 'auto'), config.get('confidence', 0.1),
                        config.get('iou', 0.7), config.get('image_size', 1280), config.get('max_det', 3000))


class JobRunner:
    def __init__(self, store, detector_factory=default_detector):
        self.store, self.detector_factory = store, detector_factory
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.loop, name='prelabel-worker', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def loop(self):
        while not self.stop_event.is_set():
            jobs = [j for j in reversed(self.store.list()) if j['status'] == 'queued']
            if jobs:
                self.run_job(jobs[0]['id'])
            else:
                self.stop_event.wait(0.5)

    def run_job(self, job_id):
        with self.store.lock:
            job = self.store.get(job_id)
            if job['status'] != 'queued':
                return
            self.store.update(job_id, status='running', started_at=time.time(), stage='loading_detector')
        start = time.perf_counter()
        results, errors = dict(job['results']), []
        directory = Path(job['directory'])
        config = job['config']
        self.store.update(job_id, fusion_warning=None, fusion_counts=None)
        metadata = {}
        fusion = None
        try:
            detector = self.detector_factory(config)
            metadata['detector'] = detector.metadata
            index, ocr = None, None
            if config.get('mode') in ('retrieval', 'reference_filter'):
                from .models import SiglipEncoder
                from .retrieval import CatalogIndex
                catalogs = config.get('catalogs', [])
                if not catalogs:
                    raise ValueError('Retrieval requires at least one catalog')
                self.store.update(job_id, stage='loading_retrieval_model')
                encoder = SiglipEncoder(config['encoder_path'], config.get('device', 'auto'))
                def index_progress(done, total):
                    if self.stop_event.is_set() or self.store.get(job_id)['status'] == 'cancelling':
                        raise JobCancelled()
                    self.store.update(job_id, stage=f'indexing_references_{done}_of_{total}')
                index = CatalogIndex(catalogs, encoder, self.store.root / 'indexes', config.get('crop_batch_size', 16), on_progress=index_progress)
                metadata['encoder'] = encoder.version
                from .references import PREPARATION_VERSION
                metadata['reference_preparation'] = {'version': PREPARATION_VERSION,
                                                     'index': index.cache_path.stem}
            if config.get('ocr_enabled'):
                from .ocr import create_ocr
                self.store.update(job_id, stage='loading_ocr')
                ocr = create_ocr(config)
                metadata['ocr'] = ocr.version
            if config.get('fusion_backend', 'visual') != 'visual' and config.get('mode') == 'reference_filter':
                from .fusion import VERSION
                metadata['fusion'] = {'policy': VERSION, 'backend': config['fusion_backend']}
                if config['fusion_backend'] == 'laya' and ocr:
                    self.store.update(job_id, stage='loading_laya')
                    try:
                        from .laya import LayaEngine
                        fusion = LayaEngine(config)
                        metadata['fusion']['model'] = fusion.version
                    except Exception as exc:
                        metadata['fusion']['fallback'] = f'{type(exc).__name__}: {exc}'
                        self.store.update(job_id, fusion_warning=metadata['fusion']['fallback'])
            for saved in results.values():
                previous_models = saved.get('model_versions') or (saved['detections'][0].get('model_versions') if saved.get('detections') else None)
                if previous_models and json.dumps(previous_models, sort_keys=True) != json.dumps(metadata, sort_keys=True):
                    raise ValueError('Model versions changed since completed image results; create a new job')
                if saved.get('staged_sha256') and digest(Path(saved['path'])) != saved['staged_sha256']:
                    raise ValueError('Completed image pixels changed; create a new job')
            for number, image_record in enumerate(job['images']):
                if self.stop_event.is_set() or self.store.get(job_id)['status'] == 'cancelling':
                    self.store.update(job_id, status='interrupted' if self.stop_event.is_set() else 'cancelled')
                    break
                if str(number) in results:
                    continue
                try:
                    if image_record.get('input_error'):
                        raise ValueError(image_record['input_error'])
                    path = Path(image_record['path'])
                    image_start = time.perf_counter()
                    self.store.update(job_id, current_image=image_record['name'], stage='detecting')
                    with Image.open(path) as original:
                        image = original.convert('RGB')
                    detect_start = time.perf_counter()
                    detections = detector.detect(image)
                    detect_time = time.perf_counter() - detect_start
                    for idx, d in enumerate(detections):
                        # Invalid model outputs fail this image; no quiet clipping or deletion.
                        yolo_row(d['bbox_xyxy_pixels'], image.width, image.height)
                        d.update({'image_id': Path(image_record['name']).stem,
                                  'detection_id': hashlib.sha256(f"{image_record['source_sha256']}:{idx}:{d['bbox_xyxy_pixels']}".encode()).hexdigest()[:24],
                                  'candidates': [], 'ocr_evidence': [], 'pack_evidence': {}, 'model_versions': metadata,
                                  'catalog_version': [c['catalog_version'] for c in config.get('catalogs', [])]})
                    retrieval_start = time.perf_counter()
                    batch_size = config.get('crop_batch_size', 16)
                    if index:
                        for offset in range(0, len(detections), batch_size):
                            if self.stop_event.is_set() or self.store.get(job_id)['status'] == 'cancelling':
                                raise JobCancelled()
                            self.store.update(job_id, stage=f'retrieving_crops_{offset}_of_{len(detections)}')
                            subset = detections[offset:offset + batch_size]
                            crops = [image.crop(tuple(d['bbox_xyxy_pixels'])) for d in subset]
                            for d, candidates in zip(subset, index.search(crops, config.get('top_k', 5),
                                                                        include_membership=config.get('mode') == 'reference_filter')):
                                d['candidates'] = candidates
                    retrieval_time = time.perf_counter() - retrieval_start
                    ocr_start = time.perf_counter()
                    if ocr:
                        from .ocr import rank_candidates
                        for number_ocr, d in enumerate(detections):
                            if self.stop_event.is_set() or self.store.get(job_id)['status'] == 'cancelling':
                                raise JobCancelled()
                            if number_ocr >= config.get('ocr_max_crops', 300):
                                d['ocr_status'] = 'skipped_limit'
                                continue
                            self.store.update(job_id, stage=f'ocr_crop_{number_ocr + 1}_of_{len(detections)}')
                            try:
                                d['ocr_evidence'] = ocr.read(image.crop(tuple(d['bbox_xyxy_pixels'])))
                                d['ocr_status'] = 'completed' if d['ocr_evidence'] else 'no_text'
                                d['ocr_candidate_ranking'] = rank_candidates(d['candidates'], d['ocr_evidence'],
                                                                           config.get('ocr_min_score', 0.7))
                            except Exception as exc:
                                d.update(ocr_status='failed', ocr_error=f'{type(exc).__name__}: {exc}')
                    else:
                        for d in detections:
                            d['ocr_status'] = 'disabled'
                    ocr_time = time.perf_counter() - ocr_start
                    fusion_start = time.perf_counter()
                    for fusion_number, d in enumerate(detections):
                        if self.stop_event.is_set() or self.store.get(job_id)['status'] == 'cancelling':
                            raise JobCancelled()
                        d.update(decide(d, baseline=config.get('mode') == 'baseline', config=config,
                                        image_size=(image.width, image.height)))
                        if config.get('mode') == 'reference_filter' and config.get('fusion_backend', 'visual') != 'visual':
                            from .fusion import fuse
                            self.store.update(job_id, stage=f'fusion_crop_{fusion_number + 1}_of_{len(detections)}')
                            d.update(fuse(d, {k: d[k] for k in ('decision', 'decision_reason')}, config, fusion))
                    resolved = resolve_packs(detections)
                    result = {**image_record, 'width': image.width, 'height': image.height, 'detections': resolved,
                              'model_versions': metadata, 'staged_sha256': digest(path),
                              'timings': {'detection': detect_time, 'retrieval': retrieval_time,
                                          'ocr': ocr_time, 'fusion': time.perf_counter() - fusion_start,
                                          'total': time.perf_counter() - image_start}}
                    write_json(directory / 'results' / f'{number}.json', result)
                    results[str(number)] = result
                except JobCancelled:
                    self.store.update(job_id, status='interrupted' if self.stop_event.is_set() else 'cancelled')
                    break
                except Exception as exc:
                    errors.append({'image': image_record['name'], 'error': f'{type(exc).__name__}: {exc}'})
                self.store.update(job_id, results=results, errors=errors, completed=len(results),
                                  processed=len(results) + len(errors), current_image=image_record['name'])
            status = self.store.get(job_id)['status']
            if status == 'cancelling':
                status = 'cancelled'
            if status == 'running':
                status = 'completed_with_errors' if errors else 'completed'
            export_path = None
            preview_path = None
            preview_warning = None
            counts = {'retained': 0, 'deferred': 0, 'edge_risk': 0, 'removed': 0}
            fusion_counts = {'attempted': 0, 'changed': 0, 'failed': 0, 'fallback': 0}
            for result in results.values():
                for detection in result['detections']:
                    decision = detection['decision']
                    counts['deferred' if decision == 'defer_review' else 'removed' if decision == 'remove' else 'retained'] += 1
                    counts['edge_risk'] += detection.get('decision_reason') == 'possible_image_edge_truncation'
                    audit = detection.get('fusion_evidence', {})
                    if audit:
                        fusion_counts['attempted'] += audit['status'] in ('completed', 'failed')
                        fusion_counts['changed'] += bool(audit['applied'])
                        fusion_counts['failed'] += audit['status'] == 'failed'
                        fusion_counts['fallback'] += not audit['applied']
            if status in ('completed', 'completed_with_errors') and results:
                self.store.update(job_id, stage='exporting')
                export_path = str(export_zip([results[k] for k in sorted(results, key=int)], directory / 'set_001.zip'))
                from .preview import save_preview
                try:
                    preview_path = save_preview(results[min(results, key=int)], directory / 'preview.jpg')
                except Exception as exc:
                    preview_warning = f'{type(exc).__name__}: {exc}'
            if status == 'completed_with_errors' and not results:
                status = 'failed'
            self.store.update(job_id, status=status, export_path=export_path, review_export_path=None, box_counts=counts,
                              preview_path=preview_path, preview_warning=preview_warning,
                              fusion_counts=fusion_counts if config.get('fusion_backend', 'visual') != 'visual' else None)
        except JobCancelled:
            self.store.update(job_id, status='interrupted' if self.stop_event.is_set() else 'cancelled')
        except Exception as exc:
            errors.append({'stage': 'job', 'error': f'{type(exc).__name__}: {exc}'})
            self.store.update(job_id, status='failed', errors=errors, export_path=None,
                              review_export_path=None, preview_path=None)
        finally:
            if fusion is not None:
                fusion.close()
            import psutil
            ocr_report_path = None
            ocr_counts = None
            if config.get('ocr_enabled'):
                from .ocr import make_report
                ocr_counts = {'attempted': 0, 'with_text': 0, 'failed': 0, 'skipped': 0}
                for image_result in results.values():
                    for d in image_result['detections']:
                        state = d.get('ocr_status')
                        ocr_counts['attempted'] += state in ('completed', 'no_text', 'failed')
                        ocr_counts['with_text'] += bool(d.get('ocr_evidence'))
                        ocr_counts['failed'] += state == 'failed'
                        ocr_counts['skipped'] += state == 'skipped_limit'
                ocr_report_path = str(directory / 'ocr.json')
                write_json(Path(ocr_report_path), make_report([results[k] for k in sorted(results, key=int)]))
            evidence = {'job_id': job_id, 'model_versions': metadata, 'config': config,
                        'results': list(results.values()), 'errors': errors,
                        'elapsed_seconds': time.perf_counter() - start + job['elapsed_seconds'],
                        'worker_memory_rss_bytes': psutil.Process().memory_info().rss,
                        'preview_warning': self.store.get(job_id).get('preview_warning'),
                        'policy': {'mode': config.get('mode', 'baseline'), 'full_product_only': config.get('full_product_only', False),
                                   'reference_min_similarity': config.get('reference_min_similarity'),
                                   'reference_sku_margin': config.get('reference_sku_margin', 0.03),
                                   'sku_margin_filters_export': False,
                                   'threshold_calibrated': False, 'deferred_boxes_in_evidence': True,
                                   'export_zip_count': int(bool(self.store.get(job_id).get('export_path'))),
                                   'occlusion_checked': False, 'automatic_pack_detection': False}}
            write_json(directory / 'evidence.json', evidence)
            self.store.update(job_id, evidence_path=str(directory / 'evidence.json'), elapsed_seconds=evidence['elapsed_seconds'],
                              ocr_report_path=ocr_report_path, ocr_counts=ocr_counts, stage='finished')
