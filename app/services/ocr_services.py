import io
import logging
from PIL import Image
import pytesseract
from pytesseract import Output

from app.config import settings

logger = logging.getLogger("app.ocr")

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def load_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes))


def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = load_image(file_bytes)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.error(f"Text extraction execution error: {str(e)}")
        return ""


def extract_layout_from_image(file_bytes: bytes) -> list[dict]:
    try:
        image = load_image(file_bytes)
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        
        layout_elements = []
        number_of_items = len(data["text"])

        for index in range(number_of_items):
            text = data["text"][index].strip()

            # 1. FIX: Check and skip empty structural spacing rows immediately
            if not text:
                continue

            # 2. FIX: Wrap type conversions safely to avoid crash triggers on bad characters
            try:
                confidence = float(data["conf"][index])
                # Skip any invalid noise blocks caught by Tesseract (-1 indicates a structural box)
                if confidence < 0:
                    continue
            except (ValueError, TypeError):
                confidence = 0.0

            layout_element = {
                "text": text,
                "confidence": confidence,
                "page_number": int(data.get("page_num", [1])[index]), # Dynamically reads pages if multi-page logs exist
                "block_number": int(data["block_num"][index]),
                "paragraph_number": int(data["par_num"][index]),
                "line_number": int(data["line_num"][index]),
                "word_number": int(data["word_num"][index]),
                "bounding_box": {
                    "left": int(data["left"][index]),
                    "top": int(data["top"][index]),
                    "width": int(data["width"][index]),
                    "height": int(data["height"][index]),
                },
            }

            layout_elements.append(layout_element)
            
        return layout_elements

    except Exception as e:
        logger.error(f"Layout mapping extraction failure encountered: {str(e)}")
        return []
