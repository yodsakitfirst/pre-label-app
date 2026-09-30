import importlib.metadata
from pathlib import Path

import numpy as np

from .files import digest


def select_device(requested='auto'):
    import torch
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda:0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class YoloDetector:
    def __init__(self, checkpoint, device='auto', confidence=0.1, iou=0.7, image_size=1280, max_det=3000):
        from ultralytics import YOLO
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise ValueError('Detector checkpoint is missing')
        self.device = select_device(device)
        self.model = YOLO(str(checkpoint))
        if self.model.task != 'detect':
            raise ValueError('Checkpoint must be an object detection model')
        names = self.model.names
        names = names if isinstance(names, dict) else dict(enumerate(names))
        if len(names) != 1 or str(names.get(0, '')).lower() not in ('product', 'products', 'object'):
            raise ValueError(f'Expected generic product checkpoint; actual class mapping: {names}')
        self.options = dict(conf=confidence, iou=iou, imgsz=image_size, max_det=max_det,
                            device=self.device, verbose=False, save=False)
        self.metadata = {'checkpoint_sha256': digest(checkpoint), 'classes': names,
                         'ultralytics': importlib.metadata.version('ultralytics'),
                         'torch': importlib.metadata.version('torch'), 'device': self.device,
                         'preprocessing': 'RGB PIL, EXIF-normalized', **self.options}

    def detect(self, image):
        result = self.model.predict(source=image, **self.options)[0]
        if tuple(result.orig_shape) != (image.height, image.width):
            raise ValueError('Detector coordinate dimensions differ from input pixels')
        boxes = result.boxes
        if boxes is None:
            raise RuntimeError('Detector returned no boxes object')
        return [{'bbox_xyxy_pixels': [float(v) for v in box], 'detection_score': float(score),
                 'detector_class_id': int(cls)}
                for box, score, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]


class SiglipEncoder:
    def __init__(self, model_path, device='auto'):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        from huggingface_hub import scan_cache_dir
        # Runtime never downloads weights. Setup is an explicit CLI operation.
        self.device = select_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = SiglipVisionModel.from_pretrained(model_path, local_files_only=True).eval().to(self.device)
        model_dir = Path(model_path)
        files = sorted(model_dir.glob('*.safetensors')) + sorted(model_dir.glob('*.json')) if model_dir.is_dir() else []
        revision = getattr(self.model.config, '_commit_hash', None)
        if not files and not revision:
            raise ValueError('Cannot establish retrieval model revision; select a local snapshot folder')
        hashes = [(p.name, digest(p)) for p in files]
        import hashlib, json
        self.version = hashlib.sha256(json.dumps([str(model_path), revision, hashes, importlib.metadata.version('transformers')]).encode()).hexdigest()

    def encode(self, images):
        import torch
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            features = self.model(**inputs).pooler_output
        return features.float().cpu().numpy()


class LocalOCR:
    """Optional local ONNX OCR. Supply model/config paths; no runtime download."""
    def __init__(self, detection_model, recognition_model, dictionary):
        from rapidocr import RapidOCR
        for path in (detection_model, recognition_model, dictionary):
            if not Path(path).is_file():
                raise ValueError('OCR model/dictionary missing')
        self.engine = RapidOCR(params={'Det.model_path': str(detection_model),
                                      'Rec.model_path': str(recognition_model), 'Rec.rec_keys_path': str(dictionary)})
        self.version = {'detector': digest(Path(detection_model)), 'recognizer': digest(Path(recognition_model)),
                        'dictionary': digest(Path(dictionary)), 'runtime': importlib.metadata.version('rapidocr')}

    def read(self, image):
        result = self.engine(np.asarray(image))
        texts = result.txts if result.txts is not None else []
        scores = result.scores if result.scores is not None else []
        return [{'text': str(text), 'recognition_score': float(score)} for text, score in zip(texts, scores)]
