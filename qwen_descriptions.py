import os, io, re, base64, json, sys, requests
from time import sleep
from PIL import Image

print(">>> QWEN DESCRIPTIONS [ANTI-HURTO v5.2] | FAST_MODE-aware <<<")

FAST_MODE   = os.getenv("FAST_MODE", "1") == "1"
VLM_MODEL   = os.getenv("VLM_MODEL", "llava:7b")
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "90"))

OLLAMA_URL  = "http://localhost:11434/api/generate"

PROMPT_DESC = (
    "Eres analista de seguridad en farmacias de Chile. Recibirás varias imágenes del MISMO rango (~10s).\n"
    "Describe en 2–4 líneas las ACCIONES relevantes y determina si hay INDICIOS DE HURTO.\n"
    "Señales: ocultar productos, meter a mochila/bolsillo/bolsa, manipular blíster/empaque/antirrobo,\n"
    "cambiar etiquetas, pasar por caja sin pagar, correr hacia la salida, cubrir cámara, forzar vitrinas.\n"
    "AL FINAL agrega EXACTAMENTE:\n"
    "CLASIFICACION: SOSPECHOSO | NO SOSPECHOSO\n"
    "MOTIVO: <frase breve>"
)

MAX_SIDE = 640 if FAST_MODE else 896
JPG_QUALITY = 85
BATCH_IMAGES_PER_RANGE = 3 if FAST_MODE else 6
TIMEOUT_S  = VLM_TIMEOUT
MAX_RETRIES = 1 if FAST_MODE else 2

RANGE_RE = re.compile(r"^(\d{4}-\d{4})_")

def load_and_downsize_to_b64(path: str) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, float(MAX_SIDE) / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def rango_key_from_name(fname: str):
    m = RANGE_RE.match(os.path.basename(fname))
    return m.group(1) if m else None

def call_vlm(images_b64, range_key: str) -> str | None:
    payload = {
        "model": VLM_MODEL,
        "prompt": PROMPT_DESC + f"\n\nRango: {range_key}\n",
        "images": images_b64,
        "stream": False,
        "keep_alive": "20m",
        "options": {
            "temperature": 0.2,
            "num_predict": 192 if FAST_MODE else 256,
            "num_ctx": 4096
        }
    }
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_S)
            if resp.status_code == 200:
                data = resp.json()
                return (data.get("response") or "").strip()
            else:
                print(f"❌ Ollama {resp.status_code} (intento {attempt}): {resp.text}")
        except Exception as e:
            print(f"❌ Excepción en llamada a VLM (intento {attempt}): {e}")
        sleep(2 * attempt)
    return None

def main():
    if len(sys.argv) < 2:
        print("⚠️ Uso: python qwen_descriptions.py <carpeta_output>")
        sys.exit(0)

    carpeta = sys.argv[1]
    base_carpeta = os.path.join("detecciones", carpeta)
    dir_personas = os.path.join(base_carpeta, "personas")
    dir_frames = os.path.join(base_carpeta, "frames")

    if not os.path.isdir(base_carpeta):
        print(f"⚠️ No existe carpeta: {base_carpeta}")
        sys.exit(0)

    per_range = {}

    def add_img(path: str):
        rk = rango_key_from_name(os.path.basename(path))
        if rk:
            per_range.setdefault(rk, []).append(path)

    if os.path.isdir(dir_personas):
        for nm in sorted(os.listdir(dir_personas)):
            if nm.lower().endswith((".jpg", ".jpeg", ".png")):
                add_img(os.path.join(dir_personas, nm))
    if os.path.isdir(dir_frames):
        for nm in sorted(os.listdir(dir_frames)):
            if nm.lower().endswith((".jpg", ".jpeg", ".png")):
                add_img(os.path.join(dir_frames, nm))

    if not per_range:
        print("⚠️ No hay imágenes para describir.")
        sys.exit(0)

    work_items = []
    for rk, paths in sorted(per_range.items()):
        full = [p for p in paths if p.endswith("_full.jpg")]
        personas = sorted([p for p in paths if not p.endswith("_full.jpg")])
        selec = []
        if full:
            selec.append(full[0])
        if personas:
            selec.append(personas[0])
        if len(personas) > 1 and len(selec) < BATCH_IMAGES_PER_RANGE:
            selec.append(personas[-1])
        for p in personas:
            if len(selec) >= BATCH_IMAGES_PER_RANGE:
                break
            if p not in selec:
                selec.append(p)
        work_items.append((rk, selec))

    print(f"🖼️ Rangos a describir: {len(work_items)} | batch={BATCH_IMAGES_PER_RANGE} | FAST_MODE={FAST_MODE}")

    rangos_info = {}

    def parse_cls_and_reason(text: str):
        if not text:
            return ("DESCONOCIDO", "")
        t = text.strip()
        m_cls = re.search(r"CLASIFICACION:\s*([A-ZÁÉÍÓÚÑ\s]+)", t, re.IGNORECASE)
        m_reason = re.search(r"MOTIVO:\s*(.+)", t, re.IGNORECASE)
        cls = (m_cls.group(1).strip().upper() if m_cls else "DESCONOCIDO")
        if "SOSPECHOSO" in cls:
            cls = "SOSPECHOSO"
        elif "NO" in cls:
            cls = "NO SOSPECHOSO"
        reason = (m_reason.group(1).strip() if m_reason else "")
        return (cls, reason)

    for rk, paths in work_items:
        existing = [p for p in paths if os.path.isfile(p)]
        if not existing:
            rangos_info[rk] = {
                "descripcion": "Descripción no disponible.",
                "clasificacion": "DESCONOCIDO",
                "motivo": "",
                "imagenes": []
            }
            print(f"✅ {rk}: DESCONOCIDO")
            continue

        try:
            images64 = [load_and_downsize_to_b64(p) for p in existing]
        except Exception:
            images64 = []

        desc = call_vlm(images64, rk) or "Descripción no disponible."
        cls, reason = parse_cls_and_reason(desc)

        if desc == "Descripción no disponible." and cls == "NO SOSPECHOSO":
            cls = "DESCONOCIDO"

        rangos_info[rk] = {
            "descripcion": desc,
            "clasificacion": cls,
            "motivo": reason,
            "imagenes": [os.path.basename(p) for p in existing]
        }

        print(f"✅ {rk}: {cls}{' - ' + reason if reason else ''}")

    descripciones = {}
    for rk, info in rangos_info.items():
        desc = info.get("descripcion") or "Descripción no disponible."
        for p in per_range.get(rk, []):
            if os.path.isfile(p):
                descripciones[os.path.basename(p)] = desc

    json_path = os.path.join(base_carpeta, "qwen_descriptions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(descripciones, f, indent=2, ensure_ascii=False)

    json_ranges = os.path.join(base_carpeta, "qwen_descriptions_ranges.json")
    with open(json_ranges, "w", encoding="utf-8") as f:
        json.dump(rangos_info, f, indent=2, ensure_ascii=False)

    print(f"✅ Guardado: {json_path}")
    print(f"✅ Guardado: {json_ranges}")

if __name__ == "__main__":
    main()



