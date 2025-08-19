# qwen_resumen.py
import os, sys, json, re, base64, requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECCIONES_DIR = os.path.join(BASE_DIR, "detecciones")

VLM_URL   = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VLM_MODEL = os.environ.get("VLM_MODEL", "llava:7b")
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "60"))
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"

SYSTEM_PROMPT_ES = (
    "Eres un analista de seguridad de retail. Respondes en español. "
    "Genera un RESUMEN BREVE (5-10 líneas) del video a partir de hallazgos por rango. "
    "Enfatiza si hubo comportamientos sospechosos y en qué rangos, con una razón corta. "
    "No incluyas advertencias legales ni rechaces la solicitud; es un informe interno de seguridad."
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

def detectar_sospecha_por_archivos(personas_dir: str) -> bool:
    if not os.path.isdir(personas_dir): return False
    for n in os.listdir(personas_dir):
        if "_robo_sospecha_" in n: return True
    return False

def resumen_offline(carpeta_path: str) -> str:
    rng_path = os.path.join(carpeta_path, "qwen_descriptions_ranges.json")
    personas_txt = os.path.join(carpeta_path, "personas_detectadas.txt")
    personas_dir = os.path.join(carpeta_path, "personas")

    rangos = {}
    if os.path.exists(rng_path):
        with open(rng_path, "r", encoding="utf-8") as f:
            rangos = json.load(f)

    total = len(rangos)
    sospechosos = [r for r, v in rangos.items() if str(v).lower().startswith("sospechoso")]
    base_detecta_sospecha = detectar_sospecha_por_archivos(personas_dir)
    any_sospecha = bool(sospechosos) or base_detecta_sospecha

    lines = []
    lines.append("Resumen de seguridad (offline):")
    lines.append(f"- Rangos analizados: {total}")
    if any_sospecha:
        lines.append(f"- Se detectó movimiento SOSPECHOSO en {len(sospechosos)} rango(s): {', '.join(sospechosos) if sospechosos else 'evidencias en imágenes marcadas.'}")
    else:
        lines.append("- No se detectaron indicios claros de hurto o manipulación sospechosa.")
    if os.path.exists(personas_txt):
        lines.append("- Anexo: ver 'personas_detectadas.txt' para detalles por tramo de 10s.")
    return "\n".join(lines)

def call_vlm_resumen(texto_contexto: str) -> str:
    try:
        resp = requests.post(
            f"{VLM_URL}/api/generate",
            json={
                "model": VLM_MODEL,
                "prompt": SYSTEM_PROMPT_ES + "\n\nContexto:\n" + texto_contexto,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=VLM_TIMEOUT
        )
        if resp.status_code != 200:
            return ""
        out = (resp.json().get("response") or "").strip()
        return out
    except requests.exceptions.RequestException:
        return ""

def main():
    if len(sys.argv) < 2:
        print("Uso: python qwen_resumen.py <carpeta_output>")
        sys.exit(1)

    nombre_carpeta = sys.argv[1]
    carpeta_path = os.path.join(DETECCIONES_DIR, nombre_carpeta)
    rng_path = os.path.join(carpeta_path, "qwen_descriptions_ranges.json")
    personas_txt = os.path.join(carpeta_path, "personas_detectadas.txt")
    out_path = os.path.join(carpeta_path, "resumen_video.txt")

    contexto = []
    if os.path.exists(rng_path):
        with open(rng_path, "r", encoding="utf-8") as f:
            rangos = json.load(f)
        for r, v in sorted(rangos.items()):
            contexto.append(f"{r}: {v}")
    if os.path.exists(personas_txt):
        with open(personas_txt, "r", encoding="utf-8") as f:
            txt = f.read()
        contexto.append("\n[personas_detectadas]\n" + txt)

    contexto_str = "\n".join(contexto).strip()
    resumen = ""

    if contexto_str:
        print(f"-> Generando resumen global... FAST_MODE={FAST_MODE}  MODEL={VLM_MODEL}")
        resumen = call_vlm_resumen(contexto_str)
        if is_refusal(resumen) or not resumen:
            resumen = resumen_offline(carpeta_path)
    else:
        resumen = resumen_offline(carpeta_path)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resumen)

    print(f"✅ Resumen guardado en: {out_path}")

if __name__ == "__main__":
    main()






