import os
import uuid
import cv2
import mediapipe as mp
import numpy as np

YOLO_GATE = True
YOLO_MODEL_NAME = 'yolov5s'
YOLO_CONF = 0.25
MIN_PERSON_AREA = 40 * 40

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECCIONES_FOLDER = os.path.join(BASE_DIR, "detecciones")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PREFER_FRAMES = True

MODEL_TRY_ORDER = [1, 0]
MIN_CONF_PRIMARY   = 0.68
MIN_CONF_FALLBACK  = 0.50

UPSCALE_MIN_SIDE        = 960
UPSCALE_MIN_SIDE_HEAD   = 640
MARGIN                  = 0.12
MAX_FACES_PER_IMAGE     = 5
MIN_FACE_PIXELS         = 90

ASPECT_MIN, ASPECT_MAX  = 0.75, 1.55
HEAD_TOP_ONLY           = True
HEAD_TOP_RATIO          = 0.58

RELAX_ASPECT_MIN, RELAX_ASPECT_MAX = 0.55, 1.90
RELAX_HEAD_TOP_ONLY     = False
RELAX_MIN_CONF_FINAL    = 0.45

USE_YUNET               = True
YUNET_MODEL_PATH        = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THR_STRICT  = 0.85
YUNET_SCORE_THR_RELAX   = 0.60
YUNET_NMS_THR           = 0.30
YUNET_TOPK              = 5000

DEBUG = False

mp_face_detection = mp.solutions.face_detection

def _ensure_yunet_model():
    if not USE_YUNET:
        return False
    if os.path.exists(YUNET_MODEL_PATH):
        return True
    try:
        import requests
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(YUNET_MODEL_PATH, "wb") as f:
                f.write(r.content)
            print("✅ YuNet descargado:", YUNET_MODEL_PATH)
            return True
        else:
            print("⚠️ No se pudo descargar YuNet (status", r.status_code, ")")
    except Exception as e:
        print("ℹ️ Descarga de YuNet saltada:", e)
    print("→ Si quieres usar YuNet, coloca manualmente el onnx en:", YUNET_MODEL_PATH)
    return False

_yolo = None
if YOLO_GATE:
    try:
        import torch
        _yolo = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True)
        _yolo.conf = YOLO_CONF
        _yolo.classes = None
        _yolo.eval()
    except Exception as e:
        print("ℹ️ YOLO no disponible para gating:", e)
        _yolo = None

_yunet = None
if _ensure_yunet_model():
    try:
        _yunet = cv2.FaceDetectorYN_create(
            YUNET_MODEL_PATH, "",
            (320, 320),
            YUNET_SCORE_THR_STRICT,
            YUNET_NMS_THR,
            YUNET_TOPK
        )
    except Exception as e:
        print("ℹ️ YuNet no disponible:", e)
        _yunet = None

def _ensure_min_size(img, min_side):
    h, w = img.shape[:2]
    s = min(h, w)
    if s >= min_side:
        return img
    scale = float(min_side) / max(1, s)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def _safe_crop(img, x, y, w, h, margin=MARGIN):
    H, W = img.shape[:2]
    mx = int(w * margin); my = int(h * margin)
    x1 = max(0, x - mx); y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def _choose_input_dir(nombre_carpeta):
    frames_dir = os.path.join(DETECCIONES_FOLDER, nombre_carpeta, "frames")
    personas_dir = os.path.join(DETECCIONES_FOLDER, nombre_carpeta, "personas")
    if PREFER_FRAMES and os.path.isdir(frames_dir):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(frames_dir)):
            return frames_dir
    return personas_dir

def _detect_mp(img_bgr, model_selection, min_conf):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=model_selection,
                                         min_detection_confidence=min_conf) as fd:
        res = fd.process(rgb)
    if not res.detections:
        return []

    H, W = img_bgr.shape[:2]
    out = []
    for det in res.detections:
        score = float(det.score[0]) if det.score else 0.0
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * W); y = int(bbox.ymin * H)
        w = int(bbox.width * W); h = int(bbox.height * H)
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

        eyes_ok = True
        try:
            kps = det.location_data.relative_keypoints
            if kps and len(kps) >= 2:
                rx = int(kps[0].x * W); ry = int(kps[0].y * H)
                lx = int(kps[1].x * W); ly = int(kps[1].y * H)
                eyes_ok = (x <= rx <= x+w and y <= ry <= y+h and
                           x <= lx <= x+w and y <= ly <= y+h)
        except Exception:
            eyes_ok = True

        out.append((x, y, w, h, score, eyes_ok))
    return out

def _detect_yunet(img_bgr, score_thr):
    if _yunet is None:
        return []
    h, w = img_bgr.shape[:2]
    _yunet.setInputSize((w, h))
    faces, _ = _yunet.detect(img_bgr)
    if faces is None:
        return []
    out = []
    for f in faces:
        x, y, fw, fh, score = f[:5]
        if float(score) < score_thr:
            continue
        x, y, fw, fh = int(x), int(y), int(fw), int(fh)
        x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
        fw = max(1, min(fw, w - x)); fh = max(1, min(fh, h - y))
        out.append((x, y, fw, fh, float(score), True))
    return out

def _yolo_person_boxes(img_bgr):
    if _yolo is None:
        return []
    results = _yolo(img_bgr)
    boxes = []
    for *box, conf, cls in results.xyxy[0]:
        label = results.names[int(cls)]
        if label != 'person':
            continue
        x1, y1, x2, y2 = map(int, box)
        if (x2 - x1) * (y2 - y1) < MIN_PERSON_AREA:
            continue
        boxes.append((x1, y1, x2, y2, float(conf)))
    return boxes

def _head_roi_from_person(box, img_shape):
    x1, y1, x2, y2, _ = box
    W = x2 - x1; H = y2 - y1
    head_h = int(0.45 * H)
    hx1 = x1 + int(0.10 * W)
    hx2 = x2 - int(0.10 * W)
    hy1 = y1
    hy2 = y1 + head_h
    Himg, Wimg = img_shape[:2]
    hx1 = max(0, hx1); hy1 = max(0, hy1)
    hx2 = min(Wimg-1, hx2); hy2 = min(Himg-1, hy2)
    if hx2 <= hx1 or hy2 <= hy1:
        return None
    return (hx1, hy1, hx2, hy2)

def _nms(dets, iou_thr=0.45):
    if not dets:
        return []
    boxes = np.array([[d[0], d[1], d[0]+d[2], d[1]+d[3]] for d in dets], dtype=np.float32)
    scores = np.array([d[4] for d in dets], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i,0], boxes[order,0])
        yy1 = np.maximum(boxes[i,1], boxes[order,1])
        xx2 = np.minimum(boxes[i,2], boxes[order,2])
        yy2 = np.minimum(boxes[i,3], boxes[order,3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_o = (boxes[order,2]-boxes[order,0])*(boxes[order,3]-boxes[order,1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds]
    return keep

def _filter_and_save(proc, detections, out_dir, base_name,
                     aspect_min, aspect_max, head_top_only, min_conf_accept):
    H, W = proc.shape[:2]
    saved = 0

    keep = _nms(detections, iou_thr=0.45)
    detections = [detections[i] for i in keep]

    for i, (x, y, w, h, score, eyes_ok) in enumerate(detections):
        if score < min_conf_accept:
            if DEBUG: print("score bajo:", score)
            continue
        if w < MIN_FACE_PIXELS or h < MIN_FACE_PIXELS:
            if DEBUG: print("muy pequeño:", w, h)
            continue
        if head_top_only:
            cy = y + h/2.0
            if cy > HEAD_TOP_RATIO * H:
                if DEBUG: print("zona inferior")
                continue

        crop = _safe_crop(proc, x, y, w, h, margin=MARGIN)
        if crop is None or crop.size == 0:
            if DEBUG: print("crop vacío")
            continue

        ar = crop.shape[1] / float(crop.shape[0])
        if ar < aspect_min or ar > aspect_max:
            if DEBUG: print("aspecto:", ar)
            continue

        if not eyes_ok:
            if DEBUG: print("ojos fuera")
            continue

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        nombre_rostro = f"{base_name}_face_{i}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(os.path.join(out_dir, nombre_rostro), crop)
        saved += 1
        if saved >= MAX_FACES_PER_IMAGE:
            break

    return saved

def detectar_rostros_en_carpeta(nombre_carpeta):
    in_dir = _choose_input_dir(nombre_carpeta)
    out_dir = os.path.join(DETECCIONES_FOLDER, nombre_carpeta, "rostros")

    if not os.path.exists(in_dir):
        print(f"❌ No existe la carpeta de entrada: {in_dir}")
        return

    total_faces_saved = 0
    out_dir_created = False

    for nombre_imagen in sorted(os.listdir(in_dir)):
        if not nombre_imagen.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        ruta_imagen = os.path.join(in_dir, nombre_imagen)
        img = cv2.imread(ruta_imagen)
        if img is None:
            continue

        proc = _ensure_min_size(img, UPSCALE_MIN_SIDE)
        base_name = os.path.splitext(nombre_imagen)[0]

        candidates = []

        for model_sel in MODEL_TRY_ORDER:
            dets = _detect_mp(proc, model_sel, MIN_CONF_PRIMARY)
            if dets: candidates.extend(dets); break
            dets = _detect_mp(proc, model_sel, MIN_CONF_FALLBACK)
            if dets: candidates.extend(dets); break

        if len(candidates) == 0 and _yolo is not None:
            pboxes = _yolo_person_boxes(proc)
            for pb in pboxes:
                roi = _head_roi_from_person(pb, proc.shape)
                if roi is None:
                    continue
                hx1, hy1, hx2, hy2 = roi
                head = proc[hy1:hy2, hx1:hx2]
                head = _ensure_min_size(head, UPSCALE_MIN_SIDE_HEAD)

                for model_sel in MODEL_TRY_ORDER:
                    dets = _detect_mp(head, model_sel, MIN_CONF_FALLBACK)
                    for (rx, ry, rw, rh, sc, eyes_ok) in dets:
                        candidates.append((hx1+rx, hy1+ry, rw, rh, sc, eyes_ok))

                if _yunet is not None and len(candidates) == 0:
                    dets = _detect_yunet(head, YUNET_SCORE_THR_RELAX)
                    for (rx, ry, rw, rh, sc, eyes_ok) in dets:
                        candidates.append((hx1+rx, hy1+ry, rw, rh, sc, eyes_ok))

        if len(candidates) == 0 and _yunet is not None:
            proc_for_yunet = _ensure_min_size(proc, UPSCALE_MIN_SIDE)
            candidates.extend(_detect_yunet(proc_for_yunet, YUNET_SCORE_THR_STRICT))

        saved = _filter_and_save(
            proc, candidates, out_dir, base_name,
            aspect_min=ASPECT_MIN, aspect_max=ASPECT_MAX,
            head_top_only=HEAD_TOP_ONLY, min_conf_accept=MIN_CONF_FALLBACK
        )
        if saved == 0:
            saved = _filter_and_save(
                proc, candidates, out_dir, base_name,
                aspect_min=RELAX_ASPECT_MIN, aspect_max=RELAX_ASPECT_MAX,
                head_top_only=RELAX_HEAD_TOP_ONLY, min_conf_accept=RELAX_MIN_CONF_FINAL
            )

        if saved > 0:
            total_faces_saved += saved
            out_dir_created = True
            print(f"✅ {nombre_imagen}: {saved} rostro(s)")
        else:
            print(f"⚠️ Sin rostros válidos: {nombre_imagen}")

    if out_dir_created:
        print(f"\n✅ Detección de rostros completada. Guardados totales: {total_faces_saved} | Carpeta: {out_dir}")
    else:
        print("\nℹ️ No se detectaron rostros aceptados; NO se creó la carpeta 'rostros'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("⚠️ Debes indicar el nombre de la carpeta (por ejemplo: '<video_name>_output')")
        sys.exit(1)
    detectar_rostros_en_carpeta(sys.argv[1])









