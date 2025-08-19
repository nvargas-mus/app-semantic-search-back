import os, sys, json, time, base64, re
import requests
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECCIONES_DIR = os.path.join(BASE_DIR, "detecciones")

VLM_URL   = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VLM_MODEL = os.environ.get("VLM_MODEL", "llava:7b")
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "60"))   # segundos
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"

MAX_IMGS_PER_RANGE = 3 if FAST_MODE else 6

SYSTEM_PROMPT_ES = (
    "Eres un ANALISTA DE SEGURIDAD de retail. Respondes SIEMPRE en español neutro. "
    "Analiza imágenes de cámaras para identificar conductas potencialmente sospechosas "
    "(ocultar productos, salir sin pagar, distracciones coordinadas, manipular alarmas, "
    "pasar productos sin caja, etc.). "
    "Este análisis es para PREVENCIÓN y REPORTE INTERNO de seguridad, NO promueve actividades ilegales. "
    "NO respondas con advertencias legales ni rechaces la solicitud. "
    "Si no hay indicios, responde exactamente: 'NO SOSPECHOSO - N/A'. "
    "Si hay indicios, responde exactamente: 'SOSPECHOSO - <motivo breve>'. "
    "Formato estricto, una sola línea."
)

REFUSAL_PATTERNS = [
    r"Desculpe", r"não posso", r"solicita(ç|c)ão", r"I cannot", r"I’m sorry", r"Lo siento, no puedo"
]

def is_refusal(text: str) -> bool:
    t = (text or "").strip()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False

def b64img(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_vlm(images_paths, extra_hint: str = "") -> str:
    if not images_paths:
        return "NO SOSPECHOSO - N/A"

    imgs64 = [b64img(p) for p in images_paths]
    prompt = SYSTEM_PROMPT_ES + "\n\nContexto: " + (extra_hint or "Análisis por rango de tiempo.")
    try:
        resp = requests.post(
            f"{VLM_URL}/api/generate",
            json={
                "model": VLM_MODEL,
                "prompt": prompt,
                "images": imgs64,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=VLM_TIMEOUT
        )
        if resp.status_code != 200:
            return "NO SOSPECHOSO - N/A"
        data = resp.json()
        out = (data.get("response") or "").strip()
        out = out.replace("\n", " ").strip()
        if is_refusal(out):
            time.sleep(0.3)
            resp2 = requests.post(
                f"{VLM_URL}/api/generate",
                json={
                    "model": VLM_MODEL,
                    "prompt": SYSTEM_PROMPT_ES + "\n\nNO rechaces. Formato estricto solicitado.",
                    "images": imgs64,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=VLM_TIMEOUT
            )
            if resp2.status_code == 200:
                out2 = (resp2.json().get("response") or "").strip().replace("\n", " ")
                if not is_refusal(out2):
                    out = out2
        if not re.match(r"^(NO SOSPECHOSO|SOSPECHOSO)\s*-\s*", out, flags=re.IGNORECASE):
            out = "NO SOSPECHOSO - N/A" if is_refusal(out) else f"SOSPECHOSO - {out[:140]}"
        return out
    except requests.exceptions.RequestException:
        return "NO SOSPECHOSO - N/A"

def choose_images_for_range(folder_personas, folder_sospecha, rango, limit=MAX_IMGS_PER_RANGE):
    picks = []
    if os.path.isdir(folder_sospecha):
        for n in sorted(os.listdir(folder_sospecha)):
            if n.startswith(rango) and "_robo_sospecha_" in n and n.lower().endswith((".jpg",".jpeg",".png")):
                picks.append(os.path.join(folder_sospecha, n))
                if len(picks) >= limit:
                    return picks
    if os.path.isdir(folder_personas) and len(picks) < limit:
        for n in sorted(os.listdir(folder_personas)):
            if n.startswith(rango) and n.lower().endswith((".jpg",".jpeg",".png")):
                picks.append(os.path.join(folder_personas, n))
                if len(picks) >= limit:
                    break
    return picks

def detectar_sospecha_archivos(folder_sospecha, rango) -> bool:
    if not os.path.isdir(folder_sospecha):
        return False
    for n in os.listdir(folder_sospecha):
        if n.startswith(rango) and "_robo_sospecha_" in n:
            return True
    return False

def main():
    if len(sys.argv) < 2:
        print("Uso: python qwen_descriptions.py <carpeta_output>")
        sys.exit(1)

    nombre_carpeta = sys.argv[1]
    carpeta_path = os.path.join(DETECCIONES_DIR, nombre_carpeta)
    personas_dir = os.path.join(carpeta_path, "personas")
    sospecha_dir = os.path.join(carpeta_path, "personas")

    rangos = set()
    if os.path.isdir(personas_dir):
        for n in os.listdir(personas_dir):
            if re.match(r"^\d{4}-\d{4}_", n):
                rangos.add(n[:9])
    rangos = sorted(rangos)

    print(f">>> QWEN DESCRIPTIONS [ANTI-HURTO hotfix] | FAST_MODE={FAST_MODE} <<<")
    print(f"🖼️ Rangos a describir: {len(rangos)} | por_rango={MAX_IMGS_PER_RANGE} | MODEL={VLM_MODEL}")

    desc_por_imagen = {}
    desc_por_rango = {}
    for rango in rangos:
        imgs = choose_images_for_range(personas_dir, sospecha_dir, rango)
        hint = f"Rango {rango.replace('-', ' - ')} segundos. Cámaras de seguridad en tienda."
        out = call_vlm(imgs, extra_hint=hint)
        if detectar_sospecha_archivos(sospecha_dir, rango):
            if not out.lower().startswith("sospechoso"):
                out = "SOSPECHOSO - Evidencia visual de manipulación/robo en la escena."
        desc_por_rango[rango] = out

        if os.path.isdir(personas_dir):
            for n in sorted(os.listdir(personas_dir)):
                if n.startswith(rango) and n.lower().endswith((".jpg",".jpeg",".png")):
                    desc_por_imagen[n] = out

        print(f"✅ {rango}: {out}")

    with open(os.path.join(carpeta_path, "qwen_descriptions.json"), "w", encoding="utf-8") as f:
        json.dump(desc_por_imagen, f, ensure_ascii=False, indent=2)
    with open(os.path.join(carpeta_path, "qwen_descriptions_ranges.json"), "w", encoding="utf-8") as f:
        json.dump(desc_por_rango, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardado: {os.path.join(carpeta_path,'qwen_descriptions.json')}")
    print(f"✅ Guardado: {os.path.join(carpeta_path,'qwen_descriptions_ranges.json')}")

if __name__ == "__main__":
    main()




