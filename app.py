import gradio as gr
import easyocr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2
import re

# ---------------------------------------------------------------------------
yolo = YOLO(r'runs/detect/runs/train_v2/weights/best.pt')
reader = easyocr.Reader(['en'], gpu=False)
print("Модели загружены.")
# ---------------------------------------------------------------------------

# ====================== OCR (НЕ ТРОГАТЬ) ======================
def remove_grid_lines(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)), iterations=2)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)), iterations=2)
    mask = cv2.dilate(cv2.add(h_lines, v_lines), np.ones((3, 3), np.uint8), iterations=1)
    img_bgr[mask == 255] = 255
    return img_bgr

def preprocess_crop(crop: Image.Image) -> np.ndarray:
    w, h = crop.size
    if h < 64:
        scale = max(2, 64 // h)
        crop = crop.resize((w * scale, h * scale), Image.LANCZOS)
    crop = ImageOps.expand(crop.convert("RGB"), border=12, fill=(255, 255, 255))
    img_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    return remove_grid_lines(img_bgr)

def run_easyocr(img_rgb: np.ndarray) -> list[tuple[str, float]]:
    results = reader.readtext(
        img_rgb, allowlist='0123456789', detail=1, paragraph=False,
        text_threshold=0.65, low_text=0.3, link_threshold=0.4,
        width_ths=0.6, height_ths=0.5, mag_ratio=1.5, min_size=8
    )
    results = sorted(results, key=lambda r: r[0][0][0])
    return [(r[1], r[2]) for r in results if r[2] > 0.45]

def ocr_crop(crop: Image.Image) -> tuple[str, str]:
    img_bgr = preprocess_crop(crop)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    variants = [
        cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(cv2.dilate(cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], kernel, 1), cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(cv2.dilate(cv2.bitwise_not(cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]), kernel, 1), cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(cv2.dilate(adaptive, kernel, 1), cv2.COLOR_GRAY2RGB)
    ]

    candidates = [run_easyocr(var) for var in variants]
    best_digits = ""
    best_score = -999
    for cand in candidates:
        if not cand: continue
        raw = ''.join(text for text, _ in cand)
        digits = re.sub(r'[^0-9]', '', raw)
        num = len(digits)
        avg_conf = sum(c for _, c in cand) / len(cand) if cand else 0

        if num == 6:          score = 1000 + avg_conf * 100
        elif num == 7:        digits = digits[:6]; score = 750 + avg_conf * 70
        elif num == 8:        digits = digits[:6]; score = 600 + avg_conf * 50
        else:                 score = -abs(num - 6) * 50 + avg_conf * 20

        if score > best_score:
            best_score = score
            best_digits = digits
    return best_digits, ''
# ====================== OCR КОНЕЦ ======================

def process_image(image, conf_threshold):
    if image is None:
        return None, "Изображение не загружено"

    results = yolo(image, conf=conf_threshold)[0]
    pil_img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    indexes = []
    for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        crop = pil_img.crop((max(0, x1-8), max(0, y1-8), min(pil_img.width, x2+8), min(pil_img.height, y2+8)))

        digits, _ = ocr_crop(crop)
        is_valid = len(digits) == 6
        color = (0, 200, 0) if is_valid else (255, 140, 0)
        label = digits if is_valid else f'? ({float(conf):.2f})'

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_y = y1 - 22 if y1 > 25 else y2 + 5
        draw.rectangle([x1, text_y, x1 + len(label)*9 + 8, text_y + 18], fill=color)
        draw.text((x1 + 3, text_y + 1), label, fill=(255, 255, 255))

        if is_valid:
            indexes.append((digits, float(conf)))

    text = f"Найдено индексов: {len(indexes)}\n" + \
           "\n".join(f"• {idx}  (YOLO conf: {cf:.2f})" for idx, cf in indexes) \
           if indexes else "Индексы не найдены"

    return pil_img, text

# ---------------------------------------------------------------------------
with gr.Blocks(title="Детектор почтовых индексов") as demo:
    gr.Markdown("# Детектор почтовых индексов")
    gr.Markdown("Загрузите изображение транспортного листа")

    with gr.Row():
        with gr.Column():
            inp_image = gr.Image(label="Транспортный лист", type="numpy")
            with gr.Row(equal_height=True):
                conf_slider = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="Порог YOLO")
                btn = gr.Button("Распознать", variant="primary", size="large")
                btn_clear = gr.Button("Очистить", variant="secondary", size="large")

        with gr.Column():
            out_image = gr.Image(label="Результат", interactive=False)
            out_text = gr.Textbox(label="Найденные индексы", lines=8, interactive=False)

    btn.click(process_image, [inp_image, conf_slider], [out_image, out_text])
    btn_clear.click(lambda: (None, None, ""), outputs=[inp_image, out_image, out_text])   # ← надёжно и минимально

if __name__ == '__main__':
    demo.launch()