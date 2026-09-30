"""Explicit setup only. Batch processing never accesses the network."""
import json
import shutil
import subprocess
from pathlib import Path

from .files import write_json
from .laya import LayaEngine, MODEL_FILES, PACKAGE_VERSION, REVISION


def setup(root):
    root = Path(root).resolve()
    node, npm = shutil.which('node'), shutil.which('npm.cmd' if __import__('os').name == 'nt' else 'npm')
    if not node or not npm:
        raise ValueError('Install Node.js 20 or newer (including npm), then run download-laya again.')
    package_dir, model_dir = root / 'laya-node', root / 'models/laya'
    subprocess.run([npm, 'install', '--prefix', str(package_dir), '--cache', str(root / 'npm-cache'),
                    '--save-exact', f'@receptron/laya@{PACKAGE_VERSION}', 'onnxruntime-node@1.30.0'], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download('receptron/laya-onnx', revision=REVISION, local_dir=model_dir,
                      allow_patterns=list(MODEL_FILES), max_workers=2)
    config = {'laya_node_path': node, 'laya_package_dir': str(package_dir), 'laya_model_dir': str(model_dir)}
    engine = LayaEngine(config)
    try:
        engine.predict({'ocr': [], 'candidates': [], 'scores_are_uncalibrated': True})
        write_json(model_dir / 'manifest.json', {'revision': REVISION, **engine.version})
    finally:
        engine.close()
    # Preserve every existing setting; OCR must already be installed/enabled separately.
    path = root / 'settings.json'
    settings = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    settings.update(config, fusion_backend='laya')
    write_json(path, settings)
    return {**config, 'fusion_backend': 'laya', 'revision': REVISION, 'offline_load_verified': True}
