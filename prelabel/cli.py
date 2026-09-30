import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

from .files import digest, write_json


def main():
    parser = argparse.ArgumentParser(description='Local retail shelf pre-labelling')
    parser.add_argument('--home', default=os.environ.get('PRELABEL_HOME', 'runtime'), help='Local runtime and artifact folder')
    commands = parser.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve', help='Start the local browser interface')
    serve.add_argument('--port', type=int, default=8765)
    commands.add_parser('doctor', help='Report hardware and installed model dependencies')
    download = commands.add_parser('download-model', help='Explicitly download SigLIP 2 weights for subsequent offline processing')
    download.add_argument('--revision', default='main')
    catalog = commands.add_parser('catalog', help='Ingest a catalog ZIP or explicitly mapped workbook')
    catalog.add_argument('source')
    catalog.add_argument('--excluded', action='store_true')
    catalog.add_argument('--sheet')
    catalog.add_argument('--header-row', type=int, default=3)
    catalog.add_argument('--barcode-column', default='Barcode')
    catalog.add_argument('--name-column', default='Product Name')
    catalog.add_argument('--image-column', default='Product ImageRef')
    batch = commands.add_parser('batch', help='Run a batch without the browser')
    batch.add_argument('source')
    batch.add_argument('--checkpoint', required=True)
    batch.add_argument('--encoder', help='Local SigLIP snapshot; enables retrieval')
    batch.add_argument('--device', default='auto')
    batch.add_argument('--image-size', type=int, default=1280)
    batch.add_argument('--confidence', type=float, default=0.1)
    batch.add_argument('--catalog-id', action='append', default=[], help='Catalog ID (repeatable); default all ingested catalogs')
    args = parser.parse_args()
    root = Path(args.home).resolve()
    if args.command == 'serve':
        import uvicorn
        from .app import create_app
        # Single process: SQLite worker ownership and model memory are local.
        print(f'Open http://127.0.0.1:{args.port}')
        uvicorn.run(create_app(root), host='127.0.0.1', port=args.port)
    elif args.command == 'doctor':
        import psutil
        report = {'platform': platform.platform(), 'python': sys.version,
                  'memory_gb': round(psutil.virtual_memory().total / 1024**3, 2),
                  'dependencies': {m: importlib.util.find_spec(m) is not None for m in ('torch', 'ultralytics', 'transformers', 'rapidocr')}}
        if report['dependencies']['torch']:
            import torch
            report.update(torch=torch.__version__, cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available())
        print(json.dumps(report, indent=2))
    elif args.command == 'download-model':
        from huggingface_hub import snapshot_download
        destination = root / 'models' / 'siglip2-base-patch16-224'
        snapshot_download('google/siglip2-base-patch16-224', revision=args.revision, local_dir=destination,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'README.md'], max_workers=2)
        # Vision adapter loads only the vision tower. No photos are uploaded.
        print(f'Local retrieval model: {destination}')
        settings_path = root / 'settings.json'
        if settings_path.exists():
            settings = json.loads(settings_path.read_text(encoding='utf-8'))
            settings['encoder_path'] = str(destination)
            write_json(settings_path, settings)
    elif args.command == 'catalog':
        from .catalog import ingest_catalog
        result = ingest_catalog(Path(args.source), root / 'catalogs', target=not args.excluded,
                                sheet=args.sheet, header_row=args.header_row, barcode_column=args.barcode_column,
                                name_column=args.name_column, image_column=args.image_column)
        print(json.dumps({'catalog_id': result['catalog_id'], 'records': len(result['records']), 'warnings': result['warnings']}, ensure_ascii=False, indent=2))
    elif args.command == 'batch':
        from .app import Settings
        from .images import import_images
        from .jobs import JobStore, JobRunner
        import uuid
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.is_file():
            parser.error('Detector checkpoint is missing')
        if not 0.001 <= args.confidence <= 1 or not 320 <= args.image_size <= 4096 or args.image_size % 32:
            parser.error('Invalid detector score or image size')
        config = Settings(checkpoint=str(checkpoint), encoder_path=args.encoder or '', device=args.device,
                          confidence=args.confidence, image_size=args.image_size).model_dump()
        config.update(checkpoint_sha256=digest(checkpoint), mode='retrieval' if args.encoder else 'baseline')
        config['catalogs'] = [json.loads(p.read_text(encoding='utf-8')) for p in (root / 'catalogs').glob('*/catalog.json')]
        if args.catalog_id:
            available = {c['catalog_id'] for c in config['catalogs']}
            if not set(args.catalog_id) <= available:
                parser.error('Unknown catalog ID')
            config['catalogs'] = [c for c in config['catalogs'] if c['catalog_id'] in args.catalog_id]
        store = JobStore(root)
        job = store.create(import_images(Path(args.source), root / 'inputs' / uuid.uuid4().hex), config)
        JobRunner(store).run_job(job['id'])
        result = store.get(job['id'])
        print(json.dumps({k: result[k] for k in ('id', 'status', 'completed', 'errors', 'export_path', 'evidence_path', 'elapsed_seconds')}, indent=2))
        if result['status'] != 'completed':
            sys.exit(1)
