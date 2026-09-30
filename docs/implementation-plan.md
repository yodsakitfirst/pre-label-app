# Local retail pre-labelling implementation

The supplied prompt is the product specification. The user explicitly requested implementation on main with no worktree; execute inline on main.

## Design

Python 3.11+ / FastAPI serves a local browser batch interface. SQLite stores bounded jobs; one worker runs model inference. Catalog archives use their explicit YAML IDs to associate `references/<id>.<extension>` with names. Preserve source IDs and barcode strings, flag duplicates, and never merge conflicting identities. Workbook ingestion follows drawing and rich-data cell relationships, never archive order.

YOLO produces pixel boxes after EXIF orientation normalization. SigLIP 2 optionally retrieves per-SKU candidates, with versioned cached embeddings. Default policy retains all uncertain detections. Dedicated pack, excluded-product and non-product evidence interfaces accept only validated evidence; touching boxes do not imply packs. No automatic exact SKU export in v1. OCR supports local optional execution and records text only, never hard exclusions.

Export only `data.yaml`, `images/*`, `labels/*`; class 0, five fields, empty labels supported. Persist all decisions and failures separately. Retry/resume completed image work without rerunning it; restart interrupts unfinished jobs. Never pretend inference failures mean empty detections.

## Tasks and checks

- [x] Catalog and export contracts: tests for numeric ID mapping, duplicate barcodes, missing references, ZIP safety, coordinates, empty labels and EXIF alignment pass.
- [x] Detection/retrieval/policy: tests for uncertain retention, exclusion evidence, adjacent products, pack constituents, candidate aggregation and cache identity pass. Real detector and SigLIP runs completed.
- [x] Persistent worker and API: tests for queue limits, cancellation, resume, restart interruption, changed models/pixels and per-image failures pass. SQLite, local API, input and artifact downloads implemented.
- [x] Batch UI, setup and docs: catalog gallery, settings, batch processing, progress and downloads built. Windows/macOS setup documented; browser/API and full suite verified.
- [x] Assets inspected and execution measurements recorded in validation.md; reviewed accuracy, Mac benchmark, OCR and annotation-app pilot remain external validation work.

## Execution evidence

31 automated tests pass. Actual baseline and retrieval each processed three supplied photos and exported 789 class-0 YOLO boxes without failures. Independent ZIP inspection checks five fields, boundary validity, pairing and source-byte preservation. Editable package installation succeeds with runtime assets present. JavaScript syntax and Python compilation pass. One upstream Starlette/httpx deprecation warning remains in test runs.

## Implementation decisions

- Implement the useful v1 stages specified by the prompt; preserve experimental filtering and pack interfaces without pretending missing trained verifiers exist. Cost if this assumption is wrong: a reviewed verifier/model and acceptance pilot must be added before automatic removal can be enabled.
- Use explicit YAML ID associations from the supplied catalog ZIPs and support workbook cell relationships. Category ZIPs are the default references; workbook verification is kept separately to avoid duplicating the default target set. Cost if the supplied catalog set differs from intended annotation targets: select/import the corrected target catalogs.
- Optional OCR uses local ONNX model paths with RapidOCR; native PP-OCRv5/Paddle execution on Mac is not claimed. Cost if the desired OCR models are incompatible: validate/convert compatible ONNX weights or implement another OCR adapter.
- Work directly on main with no worktree, as the user requested. Files remain reviewable in the working tree; no remote publication or merge was needed.

## Independent review

All substantive findings were reproduced and fixed: staging collisions, final-image cancellation, workbook sheet/role identity, mixed YAML ID types, and packaging runtime discovery. Invalid catalog inputs now report a useful 422 error; running jobs display stage and elapsed progress. Workbook sheet discovery is exposed by API but the UI still takes an explicit sheet-name input. Current memory measurement is sampled RSS rather than peak memory. These limitations are documented.

## Review focus

Hostile archive paths; duplicate basenames across folders; incomplete jobs after restart; oriented images; large catalogs and dense shelves. Uploads are inputs, never executed. Keep model and user assets outside git. Mac throughput and reviewed accuracy cannot be established on this Windows host.

