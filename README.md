# Shelf — local retail pre-labelling

A local batch tool that prepares editable YOLO product boxes for an existing annotation app. It uses your supplied Ultralytics product detector and optionally retrieves SKU candidates using a local SigLIP 2 vision encoder. It does not build another annotation interface.

## Start on this computer

The provided model and nine catalog ZIPs have been copied/ingested under the ignored `runtime/` folder. Open **http://127.0.0.1:8765** while the service is running. To restart it from this repository:

```powershell
.\.venv\Scripts\python.exe -m prelabel serve
```

Or run `scripts/start.ps1`. In Batches, enter an image folder/ZIP path or choose an image ZIP. Use the default **Reference-matched export + review ZIP** mode with the local encoder and selected target catalogs. Keep **Exclude likely image-edge cuts** checked. Download the main YOLO ZIP, review ZIP and evidence when processing finishes. Existing batches keep their original settings; start a new batch to apply the filter.

This computer already has a configured Python 3.12 environment. Do not run `py -3 -m venv .venv` over it: Windows locks the launcher while the app runs, and an attempted recreation with another Python version can leave mixed incompatible packages. Use the start command above. Stop the server with Ctrl+C before deliberately rebuilding an environment.

The browser's uploaded files are copied locally. No shelf photos are sent to an inference service. Model and package downloads need internet during setup; batch processing uses local models.

## Install on another Windows computer

Install Python 3.12 (the tested version), then run from a new project folder without an existing environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[models,test]"
.\.venv\Scripts\python.exe -m prelabel doctor
.\.venv\Scripts\python.exe -m prelabel serve
```

## Install on a Mac mini

Use native macOS Python, with an Apple silicon Python installation on Apple silicon hardware:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[models,test]'
.venv/bin/python -m prelabel doctor
.venv/bin/python -m prelabel serve
```

Open **http://127.0.0.1:8765**. Select the `.pt` checkpoint under Model & settings, and import your reference ZIPs under Reference catalogs. `auto` selects CUDA, then Apple MPS, then CPU according to availability. Selecting `cpu` gives an explicit fallback; unsupported MPS operations fail visibly rather than silently changing the runtime. Mac compatibility and throughput still need a run on the actual Mac. The app uses one worker and an eight-job queue; do not run multiple service instances against one runtime directory.

## Reference catalogs

ZIP ingestion reads `catalog.yaml` and resolves references by explicit catalog ID:

```text
catalog.yaml       # names: { 7: "00123 - Product name" }
references/7.png   # exact ID association; ZIP ordering is irrelevant
```

Barcodes stay strings, including leading zeroes and longer unit codes. Similar names and duplicate barcodes are never automatically merged. The gallery lets you inspect the associations; correct structural relationships do not prove that a source image depicts the intended SKU. Catalogs may be marked as target or known excluded references. Excluded references participate in retrieval but do not automatically delete boxes.

XLSX import requires an explicit sheet, header row and columns. Columns may be header names or uppercase Excel letters. Images are resolved through drawing anchors or rich-value metadata and image relationships. Missing mapped images remain catalog records with warnings and are omitted from retrieval. Invalid or unresolved relationships fail visibly.

For the supplied `280926 Product ImageReference.xlsx`, the product name header is blank; use:

```sh
python -m prelabel catalog '/path/to/280926 Product ImageReference.xlsx' \
  --sheet 'Allbarcode+first productname' --header-row 3 \
  --barcode-column Barcode --name-column D --image-column 'Product ImageRef'
```

The nine supplied category ZIPs already contain this target set. Importing the workbook as well adds separate references; select the intended catalogs per batch to avoid duplicating them. Workbook sheet, column mapping and reference role are part of catalog identity.

## Optional SKU candidate retrieval

Download the proposed baseline model once, explicitly:

```sh
python -m prelabel download-model
```

This downloads `google/siglip2-base-patch16-224` into `runtime/models/siglip2-base-patch16-224` and updates existing settings. Alternatively supply a compatible local snapshot folder. Runtime loads only the vision tower. The full upstream checkpoint includes text weights, which the vision loader ignores; a load report listing unexpected text keys is expected.

Reference embeddings are normalized, cached, and compared by cosine similarity. Multiple reference views are aggregated by source SKU record before shortlisting. Cache identity includes reference hashes, target status, catalog identity, model weights/config and Transformers version. Crops come from full-resolution, correctly oriented image pixels and are encoded in bounded batches. Top-K and batch size are configurable.

Scores are similarities, not probabilities or calibrated SKU confidence. A nearest target match does not establish that a crop belongs to the target list. The default reference-filter mode uses the selected target references to gate the main export. The experimental default cutoff is **0.75**. Crops below it, without target candidates, or too close to a selected excluded-reference match go to `review_candidates.zip`. The strongest target and excluded candidates are considered even outside the displayed top-K. This is a recoverable export preference, not a validated membership classifier: similar non-target packaging can still pass, and real targets can fall below the cutoff. Tune it using reviewed shelf photos. Candidates and raw scores are in the evidence JSON; both YOLO ZIPs export class `0: product`, with no automatic SKU assignment.

**Detector baseline** ignores catalogs. **Catalog candidates only** computes reference matches but keeps uncertain detector boxes. Both still apply the image-edge filter when enabled. The default image-edge check defers boxes within 1% of the image boundary. It can exclude complete products near the edge and miss cropped products whose detected box stops inside that margin. It does not verify products hidden behind shelves or other products. Edge margin, cutoff and excluded-match margin are configurable under Model & settings; the checkbox and cutoff can also be overridden per batch.

## Optional local OCR

```sh
python -m pip install -e '.[ocr]'
```

Provide compatible local ONNX detection and Thai/English recognition weights and the matching character dictionary in Model & settings, then enable OCR. The adapter uses RapidOCR as an optional local ONNX runner; it does not install the Paddle runtime or automatically acquire OCR weights. OCR executes on a bounded number of ambiguous crops and records text and raw recognition scores. Missing text never rejects a product. PP-OCRv5 model conversion/runtime and Thai accuracy have not been validated on these shelf photos or on macOS; this capability remains experimental.

## Export and source preservation

```text
set_001.zip
  data.yaml
  images/<name>.<original extension>
  labels/<name>.txt
```

`data.yaml` is exactly:

```yaml
path: .
train: images

names:
  0: product
```

Each label has five fields: integer class `0`, normalized center X/Y and width/height, without confidence or extra flags. Successful photos with zero detections receive empty labels. Failed photos are omitted and listed prominently in job failures and evidence; a failed detector never becomes an empty prediction. Duplicate stems are disambiguated consistently for image/label pairs.

Source files are never overwritten. Images with ordinary orientation are copied byte-for-byte. EXIF-rotated images are exported with normalized pixels and stripped rotation metadata, with the transformation recorded. Such images may be re-encoded to keep pixels and imported coordinates aligned. Each completed batch provides a separate preview of the first successful photo with main-export boxes and candidate scores. Candidate barcodes on previews are suggestions, not assigned labels. Original export images have no overlays.

The output contract follows the supplied prompt. Import into the existing annotation app, empty-label behavior, human SKU re-export, automatic SKU mapping, and needs-review import flags still require a pilot in that app.

## Jobs, failures and evidence

SQLite persists job configuration and completed image results. Cancellation takes effect between images or bounded retrieval/index batches; current detector/model calls finish first. Cancelled/interrupted jobs can resume and failed images can retry. On service restart, interrupted running jobs are marked explicitly. Resume reuses completed photos with the same saved configuration. Completed image hashes and model versions are checked before reuse; changed models or pixels require a new job. Do not modify catalog reference files inside the runtime between attempts.

The separate evidence JSON includes raw detections, stable IDs, candidates, reasons, class mapping, model versions/hashes, source orientation, per-stage timings, errors and process memory at completion. Worker RSS is a sample, not peak memory. Evidence is kept under `runtime/jobs/<job ID>/`, including cancelled/failed jobs. Partial cancelled batches have evidence but no import ZIP until resumed processing completes.

Pack resolution and exclusion are independent interfaces. They accept externally validated evidence with a validation ID; no model in this release creates that evidence automatically. Complete-pack detection, constituent suppression, non-product removal and known excluded-product removal need reviewed examples and a calibrated verifier. Touching and overlapping boxes alone never trigger a pack merge. Exact SKU assignment remains disabled in version 1.

## CLI batch processing

```sh
python -m prelabel batch '/path/to/photos.zip' --checkpoint '/path/to/sku110k-2.pt'
python -m prelabel batch '/path/to/photos' --checkpoint '/path/to/sku110k-2.pt' \
  --encoder '/path/to/siglip2-base-patch16-224' --device cpu
```

With an encoder, CLI batches default to `reference_filter`; without one they default to `baseline`. Use `--mode retrieval` for candidates without reference filtering, `--reference-min-similarity 0.75` to adjust the gate, or `--allow-edge-products` to disable the edge check. CLI matching uses ingested catalogs; use repeatable `--catalog-id` arguments to restrict the selection. A batch with image errors prints details and exits nonzero even if a partial ZIP is available. Set `PRELABEL_HOME` or pass global `--home /path/to/runtime` before the subcommand to keep another workspace.

## Verification and accuracy pilot

```sh
python -m pytest -q
node --check prelabel/static/app.js
```

Tests exercise image/label pairing, empty labels, normalized coordinates, EXIF rotation, conservative retention, adjacent standalone products, evidence-gated pack suppression, duplicates, ZIP safety, workbook relationships, retrieval cache and aggregation, queue bounds, failure handling, cancellation and resume. Deterministic encoder/detector fixtures test software behavior; they do not prove model accuracy. See `docs/validation.md` for asset checks and real smoke-test evidence.

For an accuracy pilot, review approximately 100 representative shelf photos including targets, similar non-target variants, packs, adjacent standalone products, glare and occlusion. Split by photo/capture session. Compare against baseline: end-to-end target box recall (including detector misses), target boxes removed, non-product/excluded removal precision and coverage, localization, pack boundaries/remaining constituents, candidate recall@K and annotation time. Record throughput, per-stage times, errors and memory on the actual Mac. The experimental export gate is available now at the user’s request, with deferred boxes recoverable. Validate thresholds before trusting it for production; automatic verified exclusion and SKU assignment remain disabled. The supplied model-generated images/labels are not verified ground truth.

## Licensing and model references

The provided checkpoint records Ultralytics 8.4.80, YOLO26n, SKU-110K training and `0: object`. Ultralytics and the checkpoint metadata identify AGPL-3.0; review your intended distribution/deployment under the [Ultralytics licensing terms](https://www.ultralytics.com/license). The code does not grant rights to the supplied weights, workbook/catalog images or datasets. SigLIP 2's [model card](https://huggingface.co/google/siglip2-base-patch16-224) lists Apache-2.0. Check upstream model/data permissions before distribution.

Official references: [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/), [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html), [PaddleX recognition models](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html).
