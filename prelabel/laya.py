"""Local persistent Node adapter for the official, pinned Laya package."""
import json
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path

from .files import digest

REVISION = '68f27dfe5a27a54fb2b1fefc432f43f972e90868'
PACKAGE_VERSION = '0.1.2'
MODEL_FILES = ('laya.onnx', 'laya.onnx.data', 'laya_config.json',
               'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json')


class LayaEngine:
    def __init__(self, config):
        model = Path(config.get('laya_model_dir', ''))
        module = Path(config.get('laya_package_dir', '')) / 'node_modules/@receptron/laya'
        node = Path(config.get('laya_node_path', ''))
        for path in [node, module / 'dist/index.js', *(model / f for f in MODEL_FILES)]:
            if not path.is_file():
                raise ValueError(f'Laya file missing: {path}. Run python -m prelabel download-laya.')
        package = json.loads((module / 'package.json').read_text(encoding='utf-8'))
        if package['version'] != PACKAGE_VERSION:
            raise ValueError('Laya package version differs from supported 0.1.2')
        self.version = {'package': PACKAGE_VERSION, 'model_files': {f: digest(model / f) for f in MODEL_FILES},
                        'bridge': digest(Path(__file__).with_name('laya_bridge.mjs'))}
        lock = module.parents[2] / 'package-lock.json'
        if lock.is_file():
            self.version['runtime_lock'] = digest(lock)
        self.responses = queue.Queue()
        self.diagnostics = deque(maxlen=8)
        self.process = subprocess.Popen([str(node), str(Path(__file__).with_name('laya_bridge.mjs')),
                                         str(module / 'dist/index.js'), str(model)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        def read_errors():
            for line in self.process.stderr:
                self.diagnostics.append(line.strip()[:500])
        self.error_reader = threading.Thread(target=read_errors, daemon=True)
        self.error_reader.start()
        try:
            if self._receive(120) != {'ready': True}:
                raise RuntimeError('Laya startup did not report ready')
        except Exception:
            self.close()
            raise

    def _receive(self, timeout=30):
        try:
            line = self.responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('Laya response timed out; visual fallback used')
        if line is None:
            self.error_reader.join(timeout=1)
            raise RuntimeError('Laya process exited: ' + ' '.join(self.diagnostics))
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def predict(self, state):
        self.process.stdin.write(json.dumps(state, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        return self._receive()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.reader.join(timeout=1)
        self.error_reader.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
