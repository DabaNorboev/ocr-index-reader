import gradio as gr
import easyocr
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

model  = YOLO(r'runs/detect/runs/train_v2/weights/best.pt')
reader = easyocr.Reader(['en'], gpu=False)

def process_image(image):
    if image is None:
        return None, "Изображение не загружено"

    results = model(image, conf=0.5)[0]
    pil_img = Image.fromarray(image)
    draw    = ImageDraw.Draw(pil_img)
    indexes = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        crop   = pil_img.crop((max(0, x1-5), max(0, y1-5),
                               min(pil_img.width, x2+5),
                               min(pil_img.height, y2+5)))
        text   = reader.readtext(np.array(crop),
                                 allowlist='0123456789', detail=0)
        digits = ''.join(text).replace(' ', '')

        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        draw.text((x1, y1 - 20), digits if digits else '?', fill=(0, 200, 0))

        if len(digits) == 6:
            indexes.append(digits)

    result = f"Найдено: {len(indexes)}\n" + '\n'.join(f"• {i}" for i in indexes) \
             if indexes else "Индексы не найдены"

    return pil_img, result

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Транспортный лист"),
    outputs=[
        gr.Image(label="Результат"),
        gr.Textbox(label="Индексы"),
    ],
    title="Детектор почтовых индексов",
)

if __name__ == '__main__':
    demo.launch()