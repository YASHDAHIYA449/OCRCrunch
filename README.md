# 🧾 Receipt OCR Expense Extractor

An end-to-end receipt processing pipeline that extracts structured expense data from receipt images using OCR and generates a financial summary.

The system automatically:

- Detects the store name
- Extracts the purchase date
- Parses item names and prices
- Identifies the total amount
- Generates a per-store expense summary

## 🚀 Features

- Batch receipt processing
- Automatic store detection (Walmart, Target, Costco, etc.)
- Robust date extraction (handles OCR noise)
- Smart item-price pairing
- TOTAL detection with fallback logic
- Expense summary generation
- Clean JSON output per receipt
- Aggregate financial report

## 📂 Project Structure

```text
├── main.py            # Batch pipeline + summary generation
├── extractor.py       # OCR parsing & structured extraction
├── preprocess.py      # Image preprocessing (optional)
├── dataset/           # Input receipt images
└── output/            # Generated JSON outputs + summary
```

## ⚙️ Installation

```bash
pip install easyocr opencv-python numpy matplotlib
```

OCR runs on CPU by default:

```python
reader = easyocr.Reader(['en'], gpu=False)
```

## 📥 Input

Place receipt images inside `dataset/`. Supported formats: `.png`, `.jpg`, `.jpeg`.

```text
dataset/
├── receipt1.jpg
├── receipt2.png
└── receipt3.jpg
```

## ▶️ Run the Pipeline

```bash
python main.py
```

The script will:

1. Read all receipts from `dataset/`
2. Run OCR extraction
3. Parse structured data
4. Save a JSON file per receipt
5. Generate the expense summary

## 📤 Output

Generated inside `output/`:

```text
output/
├── receipt1.json
├── receipt2.json
└── expense_summary.json
```

## 🧠 Extraction Logic

The parser:

- Detects the store using a known store dictionary
- Groups OCR boxes into lines
- Identifies price patterns
- Matches items with the nearest prices
- Detects TOTAL using keywords
- Falls back to the item sum if TOTAL is missing

### 🔧 Optional Preprocessing

`preprocess.py` includes grayscale conversion, denoising, adaptive thresholding, and deskewing. You can integrate it before OCR if needed.

### 🏪 Supported Store Detection

Walmart, Target, Costco, Kroger, Safeway, Whole Foods, Trader Joe's, Aldi, CVS, Walgreens, Best Buy, Home Depot, Starbucks, McDonald's, Subway, 7-Eleven, and more.

## 📈 Use Cases

- Expense tracking
- Receipt digitization
- Financial analytics
- Personal finance automation
- OCR research projects
- Computer vision pipelines

## 🛠️ Future Improvements

- Currency detection
- Multi-language OCR
- PDF receipt support
- UI dashboard
- Database integration
- ML-based store detection
