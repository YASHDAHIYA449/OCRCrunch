🧾 Receipt OCR Expense Extractor

An end-to-end receipt processing pipeline that extracts structured expense data from receipt images using OCR and generates a financial summary.

The system automatically:

Detects store name
Extracts purchase date
Parses item names and prices
Identifies total amount
Generates per-store expense summary

🚀 Features
Batch receipt processing
Automatic store detection (Walmart, Target, Costco, etc.)
Robust date extraction (handles OCR noise)
Smart item-price pairing
TOTAL detection fallback logic
Expense summary generation
Clean JSON output per receipt
Aggregate financial report

📂 Project Structure

├── main.py            # Batch pipeline + summary generation
├── extractor.py       # OCR parsing & structured extraction
├── preprocess.py      # Image preprocessing (optional)
├── dataset/           # Input receipt images
└── output/            # Generated JSON outputs + summary

⚙️ Installation
pip install easyocr opencv-python numpy matplotlib

If using CPU (default already set):

reader = easyocr.Reader(['en'], gpu=False)
📥 Input

Place receipt images inside:

dataset/

Supported formats:

.png
.jpg
.jpeg

Example:

dataset/
├── receipt1.jpg
├── receipt2.png
└── receipt3.jpg

▶️ Run Pipeline
python main.py

The script will:

Read all receipts from dataset/
Run OCR extraction
Parse structured data
Save JSON per receipt
Generate expense summary

📤 Output

Generated inside:

output/

Example:

output/
├── receipt1.json
├── receipt2.json
└── expense_summary.json

🧠 Extraction Logic

The parser intelligently:

Detects store using known store dictionary
Groups OCR boxes into lines
Identifies price patterns
Matches items with nearest prices
Detects TOTAL using keywords
Falls back to item sum if TOTAL missing
🔧 Optional Preprocessing

preprocess.py includes:

Grayscale conversion
Denoising
Adaptive thresholding
Deskewing

You can integrate it before OCR if needed.

🏪 Supported Store Detection

Includes automatic detection for:

Walmart
Target
Costco
Kroger
Safeway
Whole Foods
Trader Joe’s
Aldi
CVS
Walgreens
Best Buy
Home Depot
Starbucks
McDonald's
Subway
7-Eleven

(And more…)

📈 Use Cases
Expense tracking
Receipt digitization
Financial analytics
Personal finance automation
OCR research projects
Computer vision pipelines

🛠️ Future Improvements
Currency detection
Multi-language OCR
PDF receipt support
UI dashboard
Database integration
ML-based store detection