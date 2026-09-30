import json
import os
import platform
import secrets
import shutil
import uuid
import zipfile
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import UnidentifiedImageError
import yaml

from .catalog import ingest_catalog
from .files import digest, write_json
from .images import import_images
from .jobs import JobStore, JobRunner

STATIC = Path(__file__).parent / 'static'


class Settings(BaseModel):
    checkpoint: str = ''
    encoder_path: str = ''
    device: str = 'auto'
    confidence: float = Field(default=0.1, ge=0.001, le=1)
    iou: float = Field(default=0.7, ge=0.01, le=1)
    image_size: int = Field(default=1280, ge=320, le=4096, multiple_of=32)
    max_det: int = Field(default=3000, ge=1, le=10000)
    crop_batch_size: int = Field(default=16, ge=1, le=128)
    top_k: int = Field(default=5, ge=1, le=100)
    ocr_enabled: bool = False
    ocr_detection_model: str = ''
    ocr_recognition_model: str = ''
    ocr_dictionary: str = ''
    ocr_margin: float = Field(default=0.05, ge=0, le=2)
    ocr_max_crops: int = Field(default=20, ge=1, le=100)


class JobRequest(BaseModel):
    source: str
    mode: str = 'baseline'
    catalog_ids: list[str] = []


class CatalogRequest(BaseModel):
    source: str
    target: bool = True
    sheet: str | None = None
    header_row: int = Field(default=3, ge=1, le=100)
    barcode_column: str = 'Barcode'
    name_column: str = 'Product Name'
    image_column: str = 'Product ImageRef'


def create_app(root=None, start_worker=True):
    root = Path(root or os.environ.get('PRELABEL_HOME', 'runtime')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = JobStore(root)
    runner = JobRunner(store)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            runner.start()
        yield
        runner.stop()

    app = FastAPI(title='Shelf pre-label', lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner

    @app.middleware('http')
    async def local_guard(request: Request, call_next):
        hostname = request.url.hostname
        if hostname not in ('127.0.0.1', 'localhost', '::1', 'testserver'):
            return JSONResponse({'detail': 'Localhost access only'}, status_code=403)
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH') and request.headers.get('X-Prelabel-Token') != token:
            return JSONResponse({'detail': 'Missing local session token; reload the page'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing_error(request, exc):
        return JSONResponse({'detail': 'Item not found'}, status_code=404)

    async def bad_input(request, exc):
        return JSONResponse({'detail': f'Invalid archive, workbook, or image input: {exc}'}, status_code=422)

    for exception in (zipfile.BadZipFile, ET.ParseError, UnidentifiedImageError, yaml.YAMLError, OSError):
        app.add_exception_handler(exception, bad_input)

    def settings():
        path = root / 'settings.json'
        return Settings(**json.loads(path.read_text(encoding='utf-8'))) if path.exists() else Settings()

    def catalogs():
        result = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((root / 'catalogs').glob('*/catalog.json'))]
        from collections import Counter
        counts = Counter(r['barcode'] for c in result for r in c['records'])
        for catalog in result:
            repeated = {r['barcode'] for r in catalog['records'] if counts[r['barcode']] > 1}
            catalog['warnings'].extend(f'Barcode {code} appears in multiple catalog records; identities are preserved' for code in sorted(repeated))
        return result

    def job_summary(job):
        return {k: v for k, v in job.items() if k not in ('images', 'results', 'config')}

    @app.get('/')
    def home():
        return FileResponse(STATIC / 'index.html')

    @app.get('/api/system')
    def system():
        import psutil
        return {'token': token, 'platform': platform.platform(), 'python': platform.python_version(),
                'memory_gb': round(psutil.virtual_memory().total / 1024**3, 1),
                'settings': settings().model_dump(), 'runtime': str(root),
                'capabilities': {'automatic_exclusion': False, 'automatic_pack_detection': False, 'exact_sku_export': False}}

    @app.put('/api/settings')
    def save_settings(value: Settings):
        if value.device not in ('auto', 'cpu', 'mps', 'cuda', 'cuda:0'):
            raise ValueError('Device must be auto, cpu, mps, or cuda:0')
        if not Path(value.checkpoint).is_file() or Path(value.checkpoint).suffix.lower() != '.pt':
            raise ValueError('Select an existing .pt detector checkpoint')
        if value.ocr_enabled and not all(Path(p).is_file() for p in (value.ocr_detection_model, value.ocr_recognition_model, value.ocr_dictionary)):
            raise ValueError('Select local OCR detection/recognition models and dictionary')
        write_json(root / 'settings.json', value.model_dump())
        return value

    @app.post('/api/uploads')
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in ('.zip', '.xlsx', '.pt', '.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'):
            raise ValueError('Unsupported upload type')
        directory = root / 'uploads'
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + suffix)
        size = 0
        try:
            with path.open('wb') as f:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024**3:
                        raise ValueError('Upload exceeds 2 GiB; use a local folder or file path')
                    f.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return {'path': str(path), 'filename': file.filename, 'size': size}

    @app.get('/api/catalogs')
    def list_catalogs():
        return catalogs()

    @app.post('/api/catalogs')
    def add_catalog(value: CatalogRequest):
        if not Path(value.source).is_file():
            raise ValueError('Catalog source does not exist')
        return ingest_catalog(Path(value.source), root / 'catalogs', **value.model_dump(exclude={'source'}))

    @app.get('/api/catalogs/{catalog_id}/references/{sku_id}/{index}')
    def reference(catalog_id: str, sku_id: str, index: int):
        catalog = next((c for c in catalogs() if c['catalog_id'] == catalog_id), None)
        record = next((r for r in catalog['records'] if r['sku_id'] == sku_id), None) if catalog else None
        if record is None or index < 0 or index >= len(record['references']):
            raise HTTPException(404, 'Reference not found')
        return FileResponse(record['references'][index]['path'])

    @app.get('/api/workbook-sheets')
    def workbook_sheets(source: str):
        from openpyxl import load_workbook
        if Path(source).suffix.lower() != '.xlsx' or not Path(source).is_file():
            raise ValueError('Choose an existing .xlsx file')
        wb = load_workbook(source, read_only=True)
        try:
            return {'sheets': wb.sheetnames}
        finally:
            wb.close()

    @app.get('/api/jobs')
    def list_jobs():
        return [job_summary(j) for j in store.list()]

    @app.post('/api/jobs')
    def add_job(value: JobRequest):
        if value.mode not in ('baseline', 'retrieval'):
            raise ValueError('Mode must be baseline or retrieval')
        config = settings().model_dump()
        checkpoint = Path(config['checkpoint'])
        if not checkpoint.is_file():
            raise ValueError('Configure a detector checkpoint before starting')
        config['checkpoint_sha256'] = digest(checkpoint)
        config['mode'] = value.mode
        available = {c['catalog_id']: c for c in catalogs()}
        if any(i not in available for i in value.catalog_ids):
            raise ValueError('Unknown catalog selection')
        config['catalogs'] = [available[i] for i in value.catalog_ids]
        if value.mode == 'retrieval' and (not config['encoder_path'] or not config['catalogs']):
            raise ValueError('Retrieval requires a local SigLIP model and selected catalogs')
        if sum(j['status'] in ('queued', 'running', 'cancelling') for j in store.list()) >= 8:
            raise ValueError('The bounded job queue is full')
        destination = root / 'inputs' / uuid.uuid4().hex
        images = import_images(Path(value.source), destination)
        return job_summary(store.create(images, config))

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: str):
        return job_summary(store.get(job_id))

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        return job_summary(store.cancel(job_id))

    @app.post('/api/jobs/{job_id}/resume')
    def resume_job(job_id: str):
        return job_summary(store.resume(job_id))

    @app.get('/api/jobs/{job_id}/download/{artifact}')
    def download(job_id: str, artifact: str):
        key = {'export': 'export_path', 'evidence': 'evidence_path'}.get(artifact)
        job = store.get(job_id)
        if key is None or not job.get(key) or not Path(job[key]).is_file():
            raise HTTPException(404, 'Artifact is not ready')
        return FileResponse(job[key], filename=f"{job_id[:8]}_{'set_001.zip' if artifact == 'export' else 'evidence.json'}")

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app
