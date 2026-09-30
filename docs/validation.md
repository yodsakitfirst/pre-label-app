# Validation record

Development date: 30 September 2026. Windows development host; no worktree, implementation on main.

## Assets

- All nine supplied ZIP catalogs ingest successfully: BODY 38, DEO 24, FABSEN 36, FABSOL 87, FACE 18, FOODS 42, H&H 21, HAIR 89, SKINCLEANSING 25. Total 380 records, 379 distinct barcode strings.
- Barcode `8851932473514` occurs in BODY and SKINCLEANSING; source records remain separate.
- Supplied workbook's `Allbarcode+first productname` sheet has headers on row 3, Barcode in C, name under blank header in D, Product ImageRef in S. Rich-data/drawing relationship extraction resolves 380 row associations to embedded media. Shared media are permitted; there are 361 media files in the source package.
- `AI.zip` found in Downloads contains 632 actual JPG photos plus macOS metadata and other files. Images only are consumed; included shell scripts are never executed.
- Trusted supplied detector loads successfully: YOLO26n detection model; class `{0: object}`; recorded training version 8.4.80. Actual installed runtime versions are saved per job.

## Real detector smoke test

Three photos selected at the start, midpoint and end of the example archive were processed on CPU with `imgsz=1280`, score floor `0.1`, IoU setting `0.7`, max detections `3000`.

| Photo prefix | Detections | Detector seconds | Whole photo seconds |
| --- | ---: | ---: | ---: |
| 00b7f26b | 521 | 5.70 | 5.79 |
| 807102c9 | 166 | 0.38 | 0.44 |
| ffb03471 | 102 | 0.29 | 0.35 |

Batch including startup/export: 11.76 seconds. Three successful images and matching labels, zero errors. All detections retained and exported as `product`. Eight-decimal normalized output provides enough precision for small shelf boxes. Timing is a small Windows CPU execution check, not a Mac benchmark or extrapolated throughput guarantee.

## Real retrieval smoke test

The browser submitted a retrieval job using all nine catalogs and the downloaded SigLIP 2 vision encoder. All 789 detections received five candidate records; all remained generic product annotations. Three successful photos, zero errors. First run including model setup and reference indexing: 548.71 seconds on Windows CPU. Per-photo retrieval time: 157.88s, 48.72s, and 30.09s. Sampled worker RSS at completion: 894,545,920 bytes (about 853 MiB). The reference embedding cache contains 380 rows and is persisted for subsequent runs. This is execution evidence only; candidate correctness was not evaluated against reviewed SKUs.

## Review and acceptance limits

An independent code review identified and regression tests reproduced: temporary ZIP staging colliding with real names, final-photo cancellation stuck in cancelling, sheet/role catalog identity collisions, mixed-type YAML ID collisions, and automatic package discovery including runtime assets. These are repaired. Explicit package discovery and editable installation were checked after runtime assets existed.

No human-reviewed target/non-target or pack dataset was supplied. No claim is made for target filtering accuracy, exact-SKU accuracy, complete-pack recognition or import behavior of the user's separate annotation app. Gallery associations are structurally resolved, not visually certified. Automatic exclusion and complete-pack resolution remain evidence-gated. OCR runtime/models and Mac MPS behavior remain unvalidated. Current workers report sampled RSS, not peak memory.

Pipeline evidence and import ZIPs are retained under the ignored `runtime/jobs/` directory. The UI links each result and evidence file.

## Reference-filter update smoke test

The follow-up change was made in the existing `fix/v1` checkout without a worktree. On the user's latest `8f6462f9` photo, detector baseline produced 147 boxes. Selected HAIR references contain 89 records. At cosine cutoff 0.65, 38 boxes entered the main ZIP; visual inspection showed unrelated similar packaging still passing. The stricter experimental default 0.75 produced 4 main-export boxes and 143 recoverable review boxes, including 42 possible image-edge cuts. Preview inspection shows four TRESemme bottles in the main result; candidate barcodes remain unverified. Many plausible target bottles are deferred too, so this cutoff sacrifices coverage and is not calibrated target membership. Cached CPU run completed in 58.40 seconds with zero image errors. Both ZIPs have valid five-field generic-product rows (4 main / 143 review); original exported photos have no drawn overlays.

42 automated tests pass, including reference gating, excluded-candidate coverage beyond top-K, separate review exports, edge checks, resume artifact reset and preview-failure isolation. JavaScript syntax passes. Independent review findings about stale batch defaults and optional previews invalidating exports were reproduced or inspected and fixed. Browser inspection confirms the default filtering mode, checked edge filter and 0.75 cutoff with all nine catalogs loaded. Interior occlusion, exact SKU identification, and accuracy across photos remain unvalidated.

## Local OCR workflow update

Implemented in the existing `feat/add-OCR` checkout, no worktree. EasyOCR 1.7.2 and official CRAFT + Thai/English recognition weights are installed locally; setup records SHA256 model hashes and validates an offline load. Packaging crops are passed as contiguous BGR arrays per the runtime contract. Existing custom RapidOCR settings/jobs retain their backend.

A full CPU run on `5b0db553` with selected HAIR references, visual cutoff 0.65 and OCR limit 300 completed without image/OCR failures: 224 crop attempts, 34 crops returning any text, 35 text lines, only 4 lines at or above the 0.7 OCR score threshold. None provided meaningful catalog-word overlap. OCR took 51.11 seconds; complete batch took 138.91 seconds. The main/review counts of 55/169 arise from the visual policy, not OCR improvements. These raw text counts do not establish recognition accuracy: most outputs were fragments, punctuation or promotional numbers. A six-crop 3x enlargement probe did not produce useful high-score readings, so it was not adopted.

OCR evidence, per-crop coordinates/status/error, text-supported candidate rankings, API report download and browser read-only results view are implemented. Unreadable/skipped/failed OCR does not remove detections or change similarity/export decisions. Candidate suggestions are not verified SKU labels. Browser validation shows the enabled batch checkbox, counts and readable report table. This photo's small angled text remains a material limitation; evaluate closer/full-resolution capture and alternative recognizers before adding OCR-based acceptance rules or Laya.

Independent review caught the BGR conversion and legacy-backend migration issues; both are repaired with regression coverage. See tests/test_ocr.py for adapter, text normalization, score filtering, barcode boundaries, failure isolation, limit handling, report API and policy independence checks.
