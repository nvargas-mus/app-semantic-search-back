import os, sys, json, re, glob, requests
from pathlib import Path

BASE_DIR = Path(__file__).parent
DETECCIONES = BASE_DIR / "detecciones"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VLM_MODEL  = os.environ.get("VLM_MODEL", "llava:7b")
TIMEOUT_S  = int(os.environ.get("VLM_TIMEOUT", "90"))

NEGATION_PATTERNS = [
    r"no puedo ayudar", r"no puedo proporcionar", r"no proporcionar", r"no asist",
    r"no estoy autorizado", r"no.*cumple.*pol", r"cannot assist", r"cannot provide",
    r"i'?m not allowed", r"policy", r"contenid[o|a] restringid", r"desculp", r"sorry",
    r"não posso", r"no puedo cumplir", r"lo siento.*no.*puedo",
]

SYSTEM_PROMPT = (
    "Eres un analista de seguridad minorista. "
    "Tu tarea es RESUMIR en ESPAÑOL, de forma neutra y breve, lo observado en el video. "
    "No incluyas advertencias, políticas ni negativas. No cambies de idioma. "
    "Si hay indicios de hurto (ocultamiento de productos, manipulación sospechosa, "
    "pago evitado, salida apresurada), menciónalo claramente. "
    "Formato: 2-5 frases, sin viñetas."
)

RETRY_PROMPT_SUFFIX = (
    "\n\nIMPORTANTE: Responde SÓLO con 2-5 frases en ESPAÑOL describiendo lo observado. "
    "No incluyas mensajes de política, disculpas ni negativas."
)

def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def _is_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    for pat in NEGATION_PATTERNS:
        if re.search(pat, t):
            return True
    if len(t) < 12 and "no" in t:
        return True
    return False

def _ollama_generate(prompt: str, timeout=TIMEOUT_S) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": VLM_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    js = r.json()
    return (js.get("response") or "").strip()

def _load_context(carpeta: Path) -> dict:
    ctx = {
        "rangos": [],
        "sospechosos": [],
        "personas_detectadas": [],
    }

    txt = _read_text(carpeta / "personas_detectadas.txt")
    if txt:
        ctx["personas_detectadas"] = [l.strip() for l in txt.splitlines() if l.strip()]

    sospechosos = []
    for img in sorted((carpeta.glob("*_robo_sospecha_*.jpg"))):
        m = re.match(r"^(\d{4}-\d{4})_", img.name)
        if m:
            sospechosos.append(m.group(1))

    ctx["sospechosos"] = sospechosos

    jr = carpeta / "qwen_descriptions_ranges.json"
    if jr.exists():
        try:
            data = json.loads(jr.read_text(encoding="utf-8"))
            ctx["rangos"] = data
        except Exception:
            pass

    return ctx

def _build_prompt(ctx: dict) -> str:
    resumen_personas = "\n".join(ctx.get("personas_detectadas", [])[:6]) or "Sin detalle por rango."
    rangos_json = ctx.get("rangos") or {}
    sospechosos = ctx.get("sospechosos") or []
    sospecha_txt = (
        f"Rangos con señales de hurto a partir de imágenes sospechosas: {', '.join(sorted(set(sospechosos)))}."
        if sospechosos else "No se detectaron imágenes etiquetadas como sospechosas."
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Datos locales del análisis:\n"
        f"- Resumen por rangos (detección de personas):\n{resumen_personas}\n"
        f"- Clasificación por rangos (si existe): {json.dumps(rangos_json, ensure_ascii=False)}\n"
        f"- Señales de sospecha basadas en imágenes: {sospecha_txt}\n\n"
        f"Tarea: Redacta un RESUMEN GLOBAL del video en 2-5 frases. "
        f"Todo en español. Sin políticas ni negativas."
    )

def _fallback_summary(ctx: dict) -> str:
    rangos = ctx.get("rangos") or {}
    sospechosos = set(ctx.get("sospechosos") or [])
    personas = ctx.get("personas_detectadas") or []

    total_rangos = len(personas)
    rangos_con_personas = sum(1 for l in personas if "se detectaron personas" in l.lower())
    rangos_sos = sorted(sospechosos)

    partes = []
    if total_rangos:
        partes.append(f"Se analizaron {total_rangos} rangos de tiempo; "
                      f"hubo personas en {rangos_con_personas} de ellos.")
    else:
        partes.append("Se analizaron varios rangos de tiempo del video.")

    if rangos_sos:
        partes.append("Se observaron comportamientos potencialmente sospechosos "
                      f"en los rangos {', '.join(rangos_sos)}.")
    else:
        partes.append("No se encontraron señales claras de hurto a partir de las evidencias locales.")

    etiquetas = list({(v or "").upper() for v in rangos.values()})
    if etiquetas:
        if "SOSPECHOSO" in etiquetas:
            partes.append("La clasificación automática marcó algunos segmentos como SOSPECHOSOS.")
        else:
            partes.append("La clasificación automática no marcó segmentos como sospechosos.")

    partes.append("Recomendación: revisar manualmente los segmentos marcados o con mayor afluencia de personas.")

    return " ".join(partes)

def main():
    if len(sys.argv) < 2:
        print("Uso: python qwen_resumen.py <carpeta_output>")
        sys.exit(1)

    nombre = sys.argv[1]
    carpeta = DETECCIONES / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    ctx = _load_context(carpeta)
    prompt = _build_prompt(ctx)

    salida = carpeta / "resumen_video.txt"
    texto = ""

    try:
        resp = _ollama_generate(prompt)
        if _is_refusal(resp):
            resp2 = _ollama_generate(prompt + RETRY_PROMPT_SUFFIX)
            texto = resp2.strip()
        else:
            texto = resp.strip()
    except Exception as e:
        texto = ""

    if _is_refusal(texto):
        texto = _fallback_summary(ctx)

    salida.write_text(texto, encoding="utf-8")
    print(f"✅ Resumen guardado en: {salida}")

if __name__ == "__main__":
    main()







