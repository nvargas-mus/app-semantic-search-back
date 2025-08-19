import os, base64, requests, sys, io, re
from PIL import Image

FAST_MODE   = os.getenv("FAST_MODE", "1") == "1"
VLM_MODEL   = os.getenv("VLM_MODEL", "llava:7b")
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "90"))

MAX_SIDE    = 640 if FAST_MODE else 896
JPG_QUALITY = 85
MAX_IMAGES  = 4 if FAST_MODE else 8

if len(sys.argv) < 2:
    print("⚠️ Debes proporcionar la carpeta de detección (<nombre>_output).")
    sys.exit(0)

carpeta = sys.argv[1]
base = os.path.join("detecciones", carpeta)
personas_path = os.path.join(base, "personas")
frames_path   = os.path.join(base, "frames")
resumen_path  = os.path.join(base, "resumen_video.txt")
txt_informe   = os.path.join(base, "personas_detectadas.txt")

def to_b64(path):
    im = Image.open(path).convert("RGB")
    w,h = im.size
    scale = min(1.0, float(MAX_SIDE)/max(w,h))
    if scale < 1.0:
        im = im.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def build_samples():
    samples = []
    if os.path.isdir(frames_path):
        for nm in sorted(os.listdir(frames_path)):
            if nm.lower().endswith((".jpg",".jpeg",".png")) and nm.endswith("_full.jpg"):
                samples.append(os.path.join(frames_path, nm))
    if os.path.isdir(personas_path) and len(samples) < MAX_IMAGES:
        for nm in sorted(os.listdir(personas_path)):
            if "_persona_1" in nm and nm.lower().endswith((".jpg",".jpeg",".png")):
                samples.append(os.path.join(personas_path, nm))
                if len(samples) >= MAX_IMAGES: break
    if not samples and os.path.isdir(personas_path):
        for nm in sorted(os.listdir(personas_path)):
            if nm.lower().endswith((".jpg",".jpeg",".png")):
                samples.append(os.path.join(personas_path, nm))
                if len(samples) >= MAX_IMAGES: break
    return samples

def write_fallback_summary():
    lines = []
    sospechosos = 0
    rangos = 0
    if os.path.isfile(txt_informe):
        with open(txt_informe, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        for bloque in [b.strip() for b in txt.split("\n") if b.strip()]:
            if bloque.startswith("Entre "):
                rangos += 1
            if "SOSPECHOSOS" in bloque.upper():
                sospechosos += 1
        if rangos == 0:
            lines.append("Resumen automático: no se pudieron inferir rangos.")
        else:
            lines.append("Resumen automático (sin VLM):")
            lines.append(f"- Rangos analizados: {rangos}")
            lines.append(f"- Rangos con indicios de hurto: {sospechosos}")
            if sospechosos > 0:
                lines.append("- Se detectaron posibles ocultamientos/manipulación o salida sin pagar en uno o más rangos.")
            else:
                lines.append("- No se detectaron indicios claros de hurto.")
    else:
        lines.append("Resumen automático: no se encontró el informe por rangos.")

    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Resumen Fallback guardado en: {resumen_path}")

def main():
    samples = build_samples()
    if not samples:
        with open(resumen_path, "w", encoding="utf-8") as f:
            f.write("No fue posible generar el resumen (sin imágenes).\n")
        sys.exit(0)

    imagenes_b64 = []
    for p in samples:
        try:
            imagenes_b64.append(to_b64(p))
        except Exception:
            pass

    prompt = (
        "Eres analista de seguridad en farmacias de Chile. Estas imágenes resumen el video.\n"
        "Genera un RESUMEN GLOBAL (3–5 líneas) con foco en acciones relevantes con productos/bolsas/"
        "vitrinas y determina si hay o no indicios de hurto. NO describas imagen por imagen."
    )

    print(f"-> Generando resumen global... FAST_MODE={FAST_MODE}  MODEL={VLM_MODEL}")

    resumen = None
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": VLM_MODEL,
                "prompt": prompt,
                "images": imagenes_b64,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 192 if FAST_MODE else 256, "num_ctx": 4096},
                "keep_alive": "20m"
            },
            timeout=VLM_TIMEOUT
        )
        if resp.status_code == 200:
            resumen = (resp.json().get("response") or "").strip()
        else:
            print(f"⚠️ VLM devolvió HTTP {resp.status_code}; usando fallback.")
    except Exception as e:
        print(f"⚠️ Resumen VLM con error/timeout: {e}")

    if not resumen or len(resumen) < 10:
        write_fallback_summary()
    else:
        with open(resumen_path, "w", encoding="utf-8") as f:
            f.write(resumen + "\n")
        print(f"✅ Resumen guardado en: {resumen_path}")

if __name__ == "__main__":
    main()





