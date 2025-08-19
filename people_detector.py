import os
import sys
import cv2
import torch
import numpy as np
from collections import defaultdict
from sort import Sort

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {DEVICE}")
if DEVICE == 'cuda':
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

IMAGES_PER_SECOND = 1
MAX_IMAGES_PER_RANGE = 15

if len(sys.argv) < 2:
    print("Debes proporcionar la ruta del video como argumento.")
    sys.exit(1)

video_path = sys.argv[1]
video_name = os.path.splitext(os.path.basename(video_path))[0]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECCIONES_DIR = os.path.join(BASE_DIR, "detecciones")
OUTPUT_DIR = os.path.join(DETECCIONES_DIR, f"{video_name}_output")
PERSONAS_DIR = os.path.join(OUTPUT_DIR, "personas")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PERSONAS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

TXT_OUT_PATH = os.path.join(OUTPUT_DIR, "personas_detectadas.txt")

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.to(DEVICE)
model.conf = 0.6
model.eval()

tracker = Sort()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ No se pudo abrir el video: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(3))
height = int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
duration_seconds = int(total_frames / fps) if total_frames > 0 else 0

frames_between_saves = max(1, int(round(fps / IMAGES_PER_SECOND)))
print(f"[people_detector] fps={fps:.2f}  images/s={IMAGES_PER_SECOND}  frames_between_saves={frames_between_saves}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out_path = os.path.join(OUTPUT_DIR, 'video_detectado.mp4')
out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

frame_number = 0
rango_detecciones = defaultdict(bool)
rango_imagenes = defaultdict(int)
full_guardado = set()

def rango_label_from_second(sec: int) -> str:
    base = (sec // 10) * 10
    return f"{base:04}-{base+10:04}"

print("▶️ Procesando:", video_path)
with open(TXT_OUT_PATH, 'w', encoding='utf-8') as txtf:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_second = int(frame_number / fps)
        rango_label = rango_label_from_second(current_second)

        raw_frame = frame.copy()

        with torch.no_grad():
            results = model(frame)

        dets = []
        for *box, conf, cls in results.xyxy[0]:
            label = results.names[int(cls)]
            if label.lower() == 'person' and float(conf) > 0.6:
                x1, y1, x2, y2 = map(int, box)
                dets.append([x1, y1, x2, y2, float(conf)])

        dets_np = np.array(dets)
        tracks = tracker.update(dets_np) if len(dets) > 0 else np.empty((0, 5))

        if len(tracks) > 0:
            rango_detecciones[rango_label] = True

        for x1, y1, x2, y2, track_id in tracks:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Persona', (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if (frame_number % frames_between_saves == 0) and (rango_imagenes[rango_label] < MAX_IMAGES_PER_RANGE):
            idx = rango_imagenes[rango_label] + 1
            img_path = os.path.join(PERSONAS_DIR, f"{rango_label}_persona_{idx}.png")
            cv2.imwrite(img_path, raw_frame)
            rango_imagenes[rango_label] += 1

        if rango_label not in full_guardado:
            full_path = os.path.join(FRAMES_DIR, f"{rango_label}_full.jpg")
            cv2.imwrite(full_path, raw_frame)
            full_guardado.add(rango_label)

        out.write(frame)

    final_second = ((duration_seconds // 10) + 1) * 10
    for start in range(0, final_second, 10):
        end = start + 10
        sm, ss = divmod(start, 60)
        em, es = divmod(end, 60)
        label = f"{start:04}-{end:04}"
        persona_txt = "se detectaron personas en movimiento" if rango_detecciones[label] else "no se detectaron personas"
        txtf.write(f"Entre {sm:02}:{ss:02} - {em:02}:{es:02} segundos: {persona_txt}\n")

cap.release()
out.release()
print(f"✅ Proceso completado. Archivos en: '{OUTPUT_DIR}'")















