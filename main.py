import os
import json
import traceback
from extractor import parse_receipt_text, reader


def generate_financial_summary(all_receipts_data, output_path):
    summary = {
        "total_spend": 0.0,
        "number_of_transactions": len(all_receipts_data),
        "spend_per_store": {}
    }

    for receipt in all_receipts_data:
        total_str = receipt.get("total_amount")
        store     = receipt.get("store_name", "Unknown Store")

        if not total_str or total_str == "Unknown":
            print(f"  ⚠️  Skipping summary entry for '{store}' — total_amount not found.")
            continue

        try:
            amount = float(total_str)
            summary["total_spend"] += amount
            summary["spend_per_store"][store] = round(
                summary["spend_per_store"].get(store, 0.0) + amount, 2
            )
        except ValueError:
            print(f"  ⚠️  Could not parse total '{total_str}' for store '{store}'. Skipping.")
            continue

    summary["total_spend"] = round(summary["total_spend"], 2)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Financial Summary generated at: {output_path}")
    return summary


def run_batch_pipeline(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(valid_extensions)
    ]

    print(f"Found {len(image_files)} receipt(s). Starting batch processing...\n")

    all_receipts_data = []

    for filename in sorted(image_files):
        image_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}...")

        try:
            # Feed raw image directly to EasyOCR.
            # Preprocessing (binarize + deskew) was found to degrade OCR quality
            # on real receipt photos — it garbles the TOTAL line and barcode area.
            ocr_results = reader.readtext(image_path)

            receipt_json = parse_receipt_text(ocr_results)
            all_receipts_data.append(receipt_json)

            base_name        = os.path.splitext(filename)[0]
            json_output_path = os.path.join(output_folder, f"{base_name}.json")

            with open(json_output_path, 'w') as f:
                json.dump(receipt_json, f, indent=4)

            n_items = len(receipt_json.get("items", []))
            print(
                f"  ✅ Saved → store: {receipt_json['store_name']} | "
                f"date: {receipt_json['date']} | "
                f"items: {n_items} | "
                f"total: {receipt_json['total_amount']}"
            )

        except Exception as e:
            print(f"  ❌ Failed to process {filename}: {e}")
            traceback.print_exc()

    summary_path = os.path.join(output_folder, "expense_summary.json")
    generate_financial_summary(all_receipts_data, summary_path)


if __name__ == "__main__":
    INPUT_DIR  = "dataset"
    OUTPUT_DIR = "output"

    if not os.path.exists(INPUT_DIR):
        print(
            f"Please create a folder named '{INPUT_DIR}' "
            f"and add receipt images (.png / .jpg / .jpeg) to it."
        )
    else:
        run_batch_pipeline(INPUT_DIR, OUTPUT_DIR)