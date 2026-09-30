# Shelf — local retail pre-labelling

A local batch tool that prepares editable YOLO product boxes for an existing annotation app. It uses your supplied Ultralytics product detector and optionally retrieves SKU candidates using a local SigLIP 2 vision encoder. It does not build another annotation interface.

## Start on this computer

The provided model and nine catalog ZIPs have been copied/ingested under the ignored `runtime/` folder. Open **http://127.0.0.1:8765** while the service is running. To restart it from this repository:

```powershell
.\.venv\Scripts\python.exe -m prelabel serve
```

Or run `scripts/start.ps1`. In Batches, enter an image folder/ZIP path or choose an image ZIP. Use the default **Catalog-guided pre-labeling · one ZIP** mode with the local encoder and selected target catalogs. Keep **Exclude likely image-edge cuts** checked. Download the single YOLO ZIP when processing finishes. It contains original photos and qualifying product boxes. Omitted detections remain visible in the app and evidence report. Existing batches keep their original settings; start a new batch to apply the filter.

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

Scores are similarities, not probabilities or calibrated SKU confidence. A nearest target match does not establish that a crop belongs to the target list. The default reference-filter mode uses the selected target references to gate the main export. The experimental default cutoff is **0.75**. Crops below it, without target candidates, or too close to a selected excluded-reference match are omitted from the annotation ZIP and kept in the evidence report. The strongest target and excluded candidates are considered even outside the displayed top-K. This is a recoverable export preference, not a validated membership classifier: similar non-target packaging can still pass, and real targets can fall below the cutoff. Tune it using reviewed shelf photos. Candidates and raw scores are in the evidence JSON; the single YOLO ZIP exports class `0: product`, with no automatic SKU assignment.

**Detector baseline** ignores catalogs. **Catalog candidates only** computes reference matches but keeps uncertain detector boxes. Both still apply the image-edge filter when enabled. The default image-edge check defers boxes within 1% of the image boundary. It can exclude complete products near the edge and miss cropped products whose detected box stops inside that margin. It does not verify products hidden behind shelves or other products. Edge margin, cutoff and excluded-match margin are configurable under Model & settings; the checkbox and cutoff can also be overridden per batch.

## Local Thai/English OCR

On this computer, EasyOCR and its Thai/English models are installed under `runtime/models/easyocr-th-en`; OCR is enabled for new batches. Restart the server and reload the page after an upgrade. Select **Read packaging text with OCR** in Batches. Existing jobs keep their saved configuration.

For a new installation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[models,ocr]"
.\.venv\Scripts\python.exe -m prelabel download-ocr
```

`download-ocr` explicitly downloads official EasyOCR CRAFT text-detection and Thai/English recognition models, validates an offline load, saves model hashes, and enables OCR in settings. Downloads need internet; batch processing never downloads models or uploads photos. The EasyOCR backend uses CPU independently of the detector/embedding device.

OCR runs on each full-resolution detected crop, in detector order, up to the configurable limit (300 crops per photo by default). Set a larger limit for dense shelves. Cancellation is checked between crops. Each box records completed/no-text/failed/skipped-limit status, raw text, recognition score and text quadrilaterals relative to the crop. Crop failures preserve YOLO boxes; missing models fail explicitly before processing.

**OCR text** opens a read-only results table. **OCR report** downloads `ocr.json`; complete evidence includes the same data, model versions and timing. Text from lines meeting the default 0.7 OCR score supports a separate catalog-candidate ranking by exact barcode or normalized word overlap. Accented Latin brand names are folded and Thai marks preserved. Generic words are ignored. This lexical ranking is experimental, considers only the visual shortlist, and can miss OCR spelling errors or Thai word boundaries.

OCR suggestions do **not** alter visual similarity scores, the reference cutoff, edge decisions or generic YOLO labels. They are not verified SKU assignments. Unreadable text does not reject a product. Laya is not included. Use reviewed shelf examples to assess OCR accuracy before enabling future acceptance rules.

The previous custom ONNX adapter remains available through `pip install -e ".[ocr-onnx]"` and the RapidOCR backend in settings. Supply compatible detector/recognizer/dictionary files. Existing ONNX configurations keep that backend. It remains an advanced option; its Thai model conversions are not validated here.

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
# Improving catalog matching

New catalog indexes use prepared reference views: white margins and disconnected
headings are cropped where a product silhouette can be established. Light or
ambiguous images keep their original canvas. Original catalog images are retained.
The cache is versioned, so existing indexes are rebuilt automatically. Small
references are flagged in the match review; enlargement cannot recover missing detail.

Reference-matched mode requires the best target to pass the similarity cutoff
and beat excluded examples. Close matches between two target SKUs do not block
the generic product box: either could establish target-list membership. The SKU
margin (default 0.03) reports an exact-identity ambiguity hint only; it does not
filter exports. Duplicate views of one barcode do not count as different products.
Scores remain raw, uncalibrated similarities. Omitted boxes stay in the evidence
report and optional match review. Only one annotation ZIP is generated per batch.

After processing, click **Review matches** on a batch:

1. Compare the shelf crop against the candidate reference images. Open **Original
   reference** to check preparation and image-to-barcode associations.
2. Choose the actual target SKU and save **Correct SKU**, or save **Outside selected
   catalogs** / **Incomplete or invalid box**. Clear a review to undo it.
3. Start a new batch with the same catalogs. Correct crops become additional target
   views; outside-catalog crops become excluded examples for that exact catalog
   selection. Incomplete boxes are recorded for evaluation, not used as references.
4. Use **Check reviewed accuracy** to replay cutoff choices on reviewed boxes.
   These results are diagnostic, not independent calibration. Review a representative
   set including accepted boxes, rejected targets, similar variants, and non-list
   products; test on separate photos before changing defaults.

Saved reviews do not change existing exports. Model weights are not trained by this
feature. New-job configurations snapshot reviewed examples; clearing a review does
not invalidate already queued jobs. Settings can disable reviewed examples. The CLI
supports `--reference-sku-margin` and `--no-reviewed-examples`.

OCR remains an auxiliary text view: weak or unreadable text does not establish a SKU
and does not override the visual gate. Interior occlusion and complete promotional
pack boundaries still require human review.
