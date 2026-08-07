import easyocr
import cv2
import numpy as np
import re

# EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Brand name → composition mapping
BRAND_MAP = {
    'aziagma': 'azithromycin',
    'azee': 'azithromycin',
    'azithral': 'azithromycin',
    'crocin': 'paracetamol',
    'dolo': 'paracetamol',
    'calpol': 'paracetamol',
    'augmentin': 'amoxicillin',
    'allegra': 'fexofenadine',
    'ascoril': 'ambroxol',
    'combiflam': 'ibuprofen paracetamol',
    'montair': 'montelukast',
    'levocet': 'levocetirizine',
    'omez': 'omeprazole',
}


def clean_text(text):

    replacements = {
        'compoiltlon': 'composition',
        'containg': 'containing',
        'tablct': 'tablet',
        'conted': 'coated',
        'exclplon': 'excipion',
        'illm': 'film',
        '5oomg': '500mg',
        '50omg': '500mg'
    }

    text = text.lower()

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    text = re.sub(r'[^a-zA-Z0-9\s\+\(\)\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_text_from_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR
    results = reader.readtext(
        thresh,
        detail=1,
        paragraph=False
    )

    print("\n===== RAW OCR WORDS =====")

    text_parts = []

    for (bbox, text, confidence) in results:

        print(text, confidence)

        if confidence > 0.3:
            text_parts.append(text)

    full_text = " ".join(text_parts)

    print("\nEasyOCR extracted:", full_text)

    # Clean text
    full_text = clean_text(full_text)

    print("Cleaned text:", full_text)

    # Brand mapping
    words = full_text.split()

    extra = []

    for brand, composition in BRAND_MAP.items():

        if brand in words:
            extra.append(composition)

    if extra:

        full_text += " " + " ".join(extra)

        print("After brand mapping:", full_text)

    return full_text.strip()


def extract_medicine_name(text):

    if not text:
        return ""

    words = text.split()

    skip_words = {
        'excipients',
        'contains',
        'tablet',
        'tablets',
        'capsule',
        'capsules',
        'methylcobalamin',
        'pyridoxine',
        'folic',
        'acid',
        'mouth',
        'dissolving',
        'therapeutic',
        'use',
        'directed',
        'india',
        'healthcare',
        'private',
        'limited',
        'company'
    }

    candidates = []

    for word in words:

        word = word.strip()

        if word.lower() in skip_words:
            continue

        if len(word) < 4:
            continue

        candidates.append(word)

    if len(candidates) >= 3:
        return candidates[1] + " " + candidates[2]

    elif len(candidates) >= 2:
        return candidates[0] + " " + candidates[1]

    elif len(candidates) >= 1:
        return candidates[0]

    return ""