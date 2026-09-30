# OCR and visual membership fusion

The workflow is YOLO product boxes → SigLIP catalog candidates → crop OCR → membership decision → one YOLO ZIP. Every exported box remains class `0: product`; this does not assign exact SKU labels.

In **Model & settings**, choose a decision backend:

- **Visual matching only:** the original similarity gate.
- **OCR + visual rules:** reliable barcode or discriminating packaging words can support a target; reliable conflicting text can omit a box.
- **Laya + OCR + visual evidence:** the official local Laya model evaluates the evidence. A change requires its selected answer probability to reach the configured floor (default 0.90) and agree with the supporting OCR rule. Otherwise the visual decision is kept.

Fusion applies only to catalog-guided batches. Enable OCR for the batch. Unreadable text, OCR failures, absent Laya weights, model failures, and oversized model inputs fall back to visual matching. Image-edge checks and validated exclusions take precedence. Similarity to another target SKU does not reject a product box.

A shared brand or product category alone cannot rescue a low visual score. Rescue requires an exact target barcode or at least two additional matching words, plus a visual score within 0.05 of the batch cutoff. Brand-only rejection requires recognized brands for every selected target record; otherwise the catalog coverage is incomplete and the app keeps the visual decision. These are experimental evidence rules, not verified product identity.

## Setup and use

Node.js 20 or newer, including npm, is required. From the app folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[models,ocr,laya]"
.\.venv\Scripts\python.exe -m prelabel download-ocr
.\.venv\Scripts\python.exe -m prelabel download-laya
.\.venv\Scripts\python.exe -m prelabel serve
```

The explicit setup command installs `@receptron/laya@0.1.2` and `onnxruntime-node@1.30.0`, downloads the approximately 1.7 GB ONNX bundle, verifies local inference, and enables Laya in settings. Existing detector, OCR and similarity settings are preserved. On this workspace setup has already completed; restart the existing server and refresh the browser to load the updated code.

The published English bundle is pinned to Hugging Face revision `68f27dfe5a27a54fb2b1fefc432f43f972e90868`. Known Thai brand aliases are supplied as English brand names alongside the original text. This does not make the model multilingual. Thai variant interpretation requires validation.

Processing uses explicit local model paths, never the library's automatic downloader. One Node process loads the model per job, handles useful text crops, and closes after completion or cancellation. The evidence stores weight hashes, bridge hash, inputs, answer probabilities, fallbacks, the visual decision, and the OCR-rule comparison. The probability floor is not calibrated for these catalogs. Inputs beyond the model context budget are reported as failures rather than silently truncated.

Review matches displays per-box comparisons. Batch cards display fusion counts and startup warnings. Evidence and OCR JSON contain the detailed records; there is still only one annotation ZIP.

CLI batches can opt in with `--ocr --fusion-backend laya`, or use `--fusion-backend rules`. Local Laya paths are read from saved settings.

## Compare on saved evidence

```powershell
.\.venv\Scripts\python.exe -m prelabel compare-fusion JOB_ID
```

This reads the job database without changing jobs or exports, replays saved visual scores and OCR through all three policies, and writes `runtime/validation/fusion-JOB_ID.json`. Saved human reviews provide accuracy counts. Unreviewed boxes do not establish accuracy; reference-example reuse and selection bias mean this is not a held-out benchmark.

The existing wide shelf batch `3084c91ca2f14d55807f98b29c417e59` contains 224 boxes. At its saved 0.60 cutoff, all three policies retain 123 boxes. Its confident OCR output has no useful brand or product identification text, so Laya is not attempted for those boxes. There are zero human reviews, so no accuracy improvement can be claimed. Better crop text or clearer shelf images are needed before this decision layer can help that example.

Real local inference was also tested separately with controlled text. A brand-conflict example produced an incorrect, low-confidence Laya answer; the confidence and evidence guards kept the visual fallback. This is why Laya remains experimental and the simpler rule backend is available for comparison.
