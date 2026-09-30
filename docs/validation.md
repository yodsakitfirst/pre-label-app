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
