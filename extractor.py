import easyocr
import re
import json
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)

# ── Known store chains ────────────────────────────────────────────────────────
KNOWN_STORES = [
    ("WAL-MART",     ["WAL-MART", "WALMART", "WAL+MART", "WAL*MART", "WALAMART", "WAL MART"]),
    ("TARGET",       ["TARGET"]),
    ("COSTCO",       ["COSTCO"]),
    ("KROGER",       ["KROGER"]),
    ("SAFEWAY",      ["SAFEWAY"]),
    ("WHOLE FOODS",  ["WHOLE FOODS"]),
    ("TRADER JOE'S", ["TRADER JOE"]),
    ("ALDI",         ["ALDI"]),
    ("LIDL",         ["LIDL"]),
    ("CVS",          ["CVS PHARMACY", "CVS/PHARMACY"]),
    ("WALGREENS",    ["WALGREENS"]),
    ("RITE AID",     ["RITE AID"]),
    ("BEST BUY",     ["BEST BUY"]),
    ("HOME DEPOT",   ["HOME DEPOT"]),
    ("LOWE'S",       ["LOWE'S", "LOWES"]),
    ("IKEA",         ["IKEA"]),
    ("MCDONALD'S",   ["MCDONALD"]),
    ("STARBUCKS",    ["STARBUCKS"]),
    ("SUBWAY",       ["SUBWAY"]),
    ("7-ELEVEN",     ["7-ELEVEN", "7 ELEVEN"]),
]

HEADER_NOISE = {
    "always", "always.", "supercenter", "open 24 hours", "manager",
    "thank you", "thank you for shopping", "welcome", "low prices",
    "low prices.", "always low prices", "always low prices.",
}


def _detect_store(ocr_results):
    boxes = []
    for (bbox, text, conf) in ocr_results:
        y_coords = [pt[1] for pt in bbox]
        boxes.append({
            "text":   text.strip(),
            "conf":   conf,
            "y_min":  min(y_coords),
            "height": max(y_coords) - min(y_coords),
        })

    all_y          = [b["y_min"] for b in boxes]
    top_half_limit = (max(all_y) if all_y else 1000) / 2

    tokens = [b["text"].upper() for b in boxes]
    pairs  = [tokens[i] + tokens[i+1] for i in range(len(tokens)-1)]
    corpus = " ".join(tokens + pairs)

    for canonical, fragments in KNOWN_STORES:
        for frag in fragments:
            if frag in corpus:
                return canonical

    largest_height = 0
    best_candidate = None
    for b in boxes:
        if b["y_min"] > top_half_limit:
            continue
        if b["text"].lower() in HEADER_NOISE:
            continue
        if b["height"] > largest_height:
            largest_height = b["height"]
            best_candidate = b["text"]

    return best_candidate


def _is_valid_price(price_str):
    try:
        return float(price_str) > 0
    except (ValueError, TypeError):
        return False


def _extract_date_from_boxes(ocr_results):
    """
    Multi-pass date extractor working at individual OCR-box level.
    Handles garbled dates, split dates, and OCR character substitutions.
    """
    sorted_boxes = sorted(ocr_results, key=lambda r: r[0][0][1])
    all_y = [r[0][0][1] for r in sorted_boxes]
    if not all_y:
        return None
    bottom_third_y = max(all_y) * 0.6

    candidates = [
        (r[0][0][1], r[1].strip())
        for r in sorted_boxes
        if r[0][0][1] >= bottom_third_y
    ]

    full_date_re    = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b')
    partial_date_re = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-])\s*$')
    year_re         = re.compile(r'^\s*(\d{2,4})\b')

    # Pass 1 — clean full date in one box
    for y, text in candidates:
        m = full_date_re.search(text)
        if m:
            return m.group(1)

    # Pass 2 — partial "MM/DD/" prefix + year in next box
    for i, (y, text) in enumerate(candidates):
        m = partial_date_re.search(text)
        if m:
            prefix = m.group(1)
            for _, nxt in candidates[i+1: i+4]:
                ym = year_re.match(nxt)
                if ym:
                    return prefix + ym.group(1)

    # Pass 3 — fuzzy: normalise OCR character substitutions (0↔O, 1↔I, 8↔B …)
    fuzzy_re   = re.compile(r'[0-9OoBbIiLlSs]{1,2}[/\-][0-9OoBbIiLlSs]{1,2}[/\-][0-9OoBbIiLlSs]{2,4}')
    digit_map  = str.maketrans('OoBbIiLlSs', '0088111155')

    for _, text in candidates:
        m = fuzzy_re.search(text)
        if m:
            normalised = m.group().translate(digit_map)
            dm = full_date_re.match(normalised)
            if dm:
                return dm.group(1)

    return None


def _normalise_prices(text):
    return re.sub(r'(\d)\.\s+(\d{2})\b', r'\1.\2', text)


def parse_receipt_text(ocr_results):
    # ── Timestamps — strip before date searching ───────────────────────────────
    time_pattern = re.compile(r'\b\d{1,2}:\d{2}(:\d{2})?\b')

    # ── Weight-line detection ──────────────────────────────────────────────────
    weight_line_pattern = re.compile(
        r'\d+\.?\d*\s*(?:lb|Ib|1b|kg|oz|g)\b',
        re.IGNORECASE,
    )

    # ── Price pattern ──────────────────────────────────────────────────────────
    price_pattern = re.compile(
        r'\$?\d{1,4}\.\d{2}(?!\s*(?:lb|Ib|1b|kg|oz|g)\b)',
        re.IGNORECASE,
    )

    # ── Standalone TOTAL keyword ───────────────────────────────────────────────
    standalone_total_kw = re.compile(
        r'(?<![A-Za-z])(TOTAL|T0TAL|TO1AL|TOTA1|AMOUNT DUE|BALANCE DUE|GRAND TOTAL)(?![A-Za-z])',
        re.IGNORECASE,
    )
    subtotal_kw = re.compile(
        r'\b(subtotal|sub-total|sub total|subt)\b',
        re.IGNORECASE,
    )

    ref_header_pattern = re.compile(
        r'\b(?:st|op|te|tr|tc)\s*#',   # ST#, OP#, TE#, TR#, TC#
        re.IGNORECASE,
    )

    voided_pattern  = re.compile(r'\bvoided?\b|\bentry\b', re.IGNORECASE)
    savings_pattern = re.compile(r'\byou\s+saved\b|\bwas\s+\d|\bsave\b', re.IGNORECASE)
    footer_pattern  = re.compile(
        r'\b(scan\s+with|back\s+of|to\s+win|see\s+back|receipt\s+for|survey)\b',
        re.IGNORECASE,
    )

    skip_kw = re.compile(
        r'\b(tax|hst|gst|pst|vat|tip|discount|change|cash|savings|tend|sold|'
        r'items\s+sold|thank|manager|open\s+24|shop|nathaniel|'
        r'always|supercenter|low\s+prices)\b',
        re.IGNORECASE,
    )

    def _is_barcode_line(text):
        stripped = re.sub(r'[TC#H\s.,]', '', text.upper())
        return bool(re.fullmatch(r'\d{8,}', stripped))

    store_name = _detect_store(ocr_results)
    date       = _extract_date_from_boxes(ocr_results)

    LINE_TOLERANCE = 12  # px  (was 20)

    def y_center(bbox):
        return (bbox[0][1] + bbox[2][1]) / 2

    sorted_results = sorted(ocr_results, key=lambda r: y_center(r[0]))
    lines = []
    for item in sorted_results:
        placed = False
        for line in lines:
            if abs(y_center(item[0]) - y_center(line[0][0])) <= LINE_TOLERANCE:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    total_candidates = []
    all_price_lines  = []
    named_lines      = []
    priced_lines     = []

    for line in lines:
        line_sorted = sorted(line, key=lambda r: r[0][0][0])
        raw_text    = " ".join(frag[1].strip() for frag in line_sorted)
        line_yc     = y_center(line[0][0])


        full_text = _normalise_prices(raw_text)

        # ── Skip barcode / TC# lines ───────────────────────────────────────────
        if _is_barcode_line(full_text):
            continue

        if ref_header_pattern.search(full_text):
            continue

        if voided_pattern.search(full_text):   # "** VOIDED ENTRY **"
            continue
        if savings_pattern.search(full_text):  # "WAS 4.54 YOU SAVED 0.54"
            continue
        if footer_pattern.search(full_text):   # "Scan with Walmart app …"
            continue

        if date is None:
            text_no_time = time_pattern.sub('', full_text)
            m = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', text_no_time)
            if m:
                date = m.group(1)

        # ── Standalone TOTAL before skip_kw ───────────────────────────────────
        has_standalone_total = (
            standalone_total_kw.search(full_text) is not None
            and subtotal_kw.search(full_text) is None
        )
        if has_standalone_total:
            pm = price_pattern.findall(full_text)
            if pm:
                val = pm[-1].lstrip('$')
                try:
                    f = float(val)
                    if 0 < f <= 9999.99:
                        total_candidates.append((f, val))
                except ValueError:
                    pass
            continue

        # ── Skip noise lines ───────────────────────────────────────────────────
        if skip_kw.search(full_text) or subtotal_kw.search(full_text):
            continue

        # ── Skip weight descriptor lines (but save their price) ────────────────
        if weight_line_pattern.search(full_text):
            pm = price_pattern.findall(full_text)
            if pm:
                raw_price   = pm[-1]
                clean_price = raw_price.lstrip('$')
                try:
                    price_float = float(clean_price)
                    if 0 < price_float <= 9999.99:
                        priced_lines.append({
                            "yc":    line_yc,
                            "price": clean_price,
                            "float": price_float,
                        })
                except ValueError:
                    pass
            continue

        # ── Price detection ────────────────────────────────────────────────────
        price_matches = price_pattern.findall(full_text)

        if not price_matches:
            clean = full_text.strip()
            if (len(clean) > 2
                    and clean.lower() not in HEADER_NOISE
                    and not (store_name and clean.upper() == store_name.upper())
                    and not re.fullmatch(r'[\d\s\-\+\*#/:]+', clean)):
                named_lines.append({"yc": line_yc, "text": clean})
            continue

        raw_price   = price_matches[-1]
        clean_price = raw_price.lstrip('$')

        try:
            price_float = float(clean_price)
        except ValueError:
            continue
        if price_float <= 0 or price_float > 9999.99:
            continue

        price_pos = full_text.rfind(raw_price)
        name_raw  = full_text[:price_pos].strip().rstrip('-').strip()

        name_raw = re.sub(r'\s+\d{6,}\s*\w*$', '', name_raw).strip()
        # Strip trailing single-letter tax/flag codes ("F", "N", "T", "O", "D", "X")
        name_raw = re.sub(r'\s+[A-Z]$', '', name_raw).strip()

        if name_raw and not re.fullmatch(r'\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?', name_raw.rstrip('/')):
            all_price_lines.append({"name": name_raw, "price": clean_price})
        else:
            priced_lines.append({"yc": line_yc, "price": clean_price, "float": price_float})

    # ── Cross-line pairing ─────────────────────────────────────────────────────
    MAX_PAIR_GAP = 150  # px

    used_named = set()
    for pl in priced_lines:
        best_idx  = None
        best_dist = float('inf')
        for i, nl in enumerate(named_lines):
            if i in used_named:
                continue
            if nl["yc"] < pl["yc"]:
                dist = pl["yc"] - nl["yc"]
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i
        if best_idx is not None and best_dist <= MAX_PAIR_GAP:
            used_named.add(best_idx)
            all_price_lines.append({
                "name":  named_lines[best_idx]["text"],
                "price": pl["price"],
            })

    total_amount = None
    item_sum = sum(
        float(r["price"]) for r in all_price_lines
        if _is_valid_price(r["price"])
    )

    if total_candidates:
        # Trust keyword-matched totals; pick the smallest to avoid outliers.
        total_amount = min(total_candidates, key=lambda x: x[0])[1]
    elif item_sum > 0:
        # TOTAL line was garbled — best estimate is the sum of found items.
        total_amount = f"{item_sum:.2f}"
    elif all_price_lines:
        try:
            total_amount = max(all_price_lines, key=lambda x: float(x["price"]))["price"]
        except ValueError:
            pass

    items = [r for r in all_price_lines if r["price"] != total_amount]

    return {
        "store_name":   store_name   or "Unknown",
        "date":         date         or "Unknown",
        "items":        items,
        "total_amount": total_amount or "Unknown",
    }


def run_extraction(image_path):
    """Runs EasyOCR on an image file and returns pretty-printed JSON."""
    print(f"Extracting text from: {image_path}")
    results    = reader.readtext(image_path)
    structured = parse_receipt_text(results)
    return json.dumps(structured, indent=4)


if __name__ == "__main__":
    import sys
    sample_image = sys.argv[1] if len(sys.argv) > 1 else "receipt.jpg"
    try:
        json_output = run_extraction(sample_image)
        print("\n--- Extracted JSON Output ---")
        print(json_output)
    except FileNotFoundError:
        print(f"Error: File '{sample_image}' not found.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise