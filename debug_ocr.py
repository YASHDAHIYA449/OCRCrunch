"""
Run this from your project folder:
    python debug_ocr.py dataset/0.jpg

It prints every single OCR box with its y-position, text, and confidence.
This tells us exactly what EasyOCR is seeing before any parsing logic runs.
"""
import sys
import json
from preprocess import process_receipt
from extractor import reader, _detect_store, parse_receipt_text

image_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/0.jpg"

print(f"\n{'='*60}")
print(f"IMAGE: {image_path}")
print(f"{'='*60}\n")

# Step 1: Show raw OCR on the ORIGINAL image (no preprocessing)
print("── RAW OCR (original image, no preprocessing) ──────────────")
raw_results = reader.readtext(image_path)
for (bbox, text, conf) in raw_results:
    y = int((bbox[0][1] + bbox[2][1]) / 2)
    print(f"  y={y:4d}  conf={conf:.2f}  text={repr(text)}")

print(f"\nTotal boxes: {len(raw_results)}\n")

# Step 2: Show raw OCR on the PREPROCESSED image
print("── RAW OCR (preprocessed image) ────────────────────────────")
_, clean = process_receipt(image_path)
pre_results = reader.readtext(clean)
for (bbox, text, conf) in pre_results:
    y = int((bbox[0][1] + bbox[2][1]) / 2)
    print(f"  y={y:4d}  conf={conf:.2f}  text={repr(text)}")

print(f"\nTotal boxes: {len(pre_results)}\n")

# Step 3: Show what _detect_store picks from each
print("── STORE DETECTION ─────────────────────────────────────────")
print(f"  From original:     {_detect_store(raw_results)}")
print(f"  From preprocessed: {_detect_store(pre_results)}")

# Step 4: Show final parsed output from each
print("\n── FINAL JSON (original image) ─────────────────────────────")
print(json.dumps(parse_receipt_text(raw_results), indent=2))

print("\n── FINAL JSON (preprocessed image) ────────────────────────")
print(json.dumps(parse_receipt_text(pre_results), indent=2))