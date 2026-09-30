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
## Matching improvements — 2026-09-30

Implemented on `fix/improve` in the existing checkout. No worktree was created.
The automated suite passes 57 tests; JavaScript syntax and `git diff --check`
pass. The suite still reports the existing Starlette/httpx deprecation warning.

Regression coverage includes disconnected headings touching the canvas edge,
light-colored bottle bodies, fallback where only a colored cap is visible,
distinct-barcode ambiguity at Top-K=1, duplicate-barcode views, review crop
coordinates, feedback persistence, scoped excluded examples, immutable crops
after undo, new-job example snapshots, and previously missing target views.

Real local inference: job `77f13d72cfe145e0b3073c56b5b12438` under
`runtime/matching_validation`, using the wide shelf photo and selected HAIR
catalog. OCR disabled to isolate matching changes. Preparation version
`white_canvas_components_v3`, cutoff 0.60, SKU margin 0.03, edge checks enabled:
224 detections, 24 main-export boxes, 200 deferred, zero image failures,
218.15 seconds. Deferred reasons: 99 competing-SKU ambiguities, 97 below cutoff,
4 image-edge risks. The earlier raw-reference job at the same cutoff exported
135 and deferred 89. This comparison measures changed eligibility, not accuracy.

Replaying the final scores at 0.65 / 0.70 / 0.75 exports 11 / 1 / 0 boxes.
Defaults were not lowered based on an unlabelled sample. Some incorrect
cross-brand matches still pass at 0.60; reviewed negative examples and validation
on separate photos are needed before claiming useful catalog precision or recall.

The repeated Sunsilk reference `8851932415804` now prepares from 91×138 to
37×115, removing the detached barcode heading and flagging its limited resolution.
Original files are preserved. The gallery does not establish image-to-barcode
correctness automatically.

Browser verification used a separate `runtime/matching_ui_test` store: all
candidate images displayed, review saved, threshold replay showed one reviewed
box, and review cleared. No test review was added to the primary runtime.
Screenshot: `runtime/matching-review-ui.png`. The temporary server was stopped
after verification.
## Correction: target membership versus exact SKU identity

The competing-target-SKU export gate described in the earlier matching run was
too strict for generic product pre-labelling. It has been removed. Similarity
between two target SKUs now produces an identity-ambiguity hint only. The minimum
target similarity, excluded-reference margin, and image-edge checks remain active.
No exact SKU is assigned by this gate.

Replaying saved scores from primary batch `3084c91ca2f14d55807f98b29c417e59` at its
0.60 cutoff with the corrected policy retains 123 boxes, defers 97 below cutoff,
and defers 4 edge-risk boxes. This is a policy replay, not a new inference run or
an accuracy measurement. Historical exports retain their original decisions.

The full suite passes 58 tests, including close target-target matches retaining
a generic box while close target-excluded matches remain deferred.
## Single annotation ZIP

New batches produce only `set_001.zip`: original photos and qualifying generic
product boxes. Omitted detections remain in the evidence JSON and match inspection
view. The worker no longer writes `review_candidates.zip`; the review download
route and link were removed, including for legacy job summaries. Previously saved
files are preserved on disk. The target-membership policy is unchanged by this
export simplification.

Verification: 59 tests passed, JavaScript syntax and diff whitespace checks passed.
Tests cover one physical ZIP despite omitted edge boxes, retained omitted-box
evidence, and hiding/disabling legacy second-ZIP downloads. Browser verification
showed one annotation download for a copied historical batch in an isolated test
store; its displayed counts are historical, not a fresh inference result.
Screenshot: `runtime/single-export-ui.png`. The temporary server was stopped.
