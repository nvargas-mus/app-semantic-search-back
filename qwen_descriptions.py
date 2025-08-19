import os, sys, json, re, glob, requests
from pathlib import Path

BASE_DIR = Path(__file__).parent
DETECCIONES = BASE_DIR / "detecciones"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VLM_MODEL  = os.environ.get("VLM_MODEL", "llava:7b")
TIMEOUT_S  = int(os.environ.get("VLM_TIMEOUT", "90"))

NEGATION_PATTERNS = [
    r"no puedo ayudar", r"no puedo proporcionar", r"cannot assist", r"policy",
    r"desculp", r"não posso", r"sorry", r"lo siento", r"no.*puedo.*ayud",
]

TASK_INSTR = (
    "Eres un analista de seguridad minorista. "
    "A partir de una IMAGEN fija de CCTV, devuelve SOLO una línea en ESPAÑOL, "
    "sin explicaciones, con este formato EXACTO:\n"
    "- 'SOSPECHOSO: <motivo breve>' si se observa conducta compatible con hurto "
    "(ocultar productos, pasarlos a bolsillos/mochila/ropa, vigilar al personal, "
    "salida apresurada sin pago).\n"
    "- 'NO SOSPECHOSO' en caso contrario.\n"
    "Prohibido responder con políticas o disculpas. No cambies de idioma."
)

def _is_refusal(t: str) -> bool:
    t = (t or "").lower().strip()
    if not t:
        return True
    for p in NEGATION_PATTERNS:
        if re.search(p, t):
            return True
    return False

def _ollama_image_prompt(img_path_or_url: str) -> str:
    return f"[IMAGEN]: {img_path_or_url}\n\n{TASK_INSTR}"

def _gen(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": VLM_MODEL, "prompt": prompt, "stream": False},
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def _pick_images(in_dir: Path, max_per_range=3):
    imgs = sorted([p for p in in_dir.glob("*.jpg")] + [p for p in in_dir.glob("*.png")])
    by_range = {}
    for p in imgs:
        m = re.match(r"^(\d{4}-\d{4})_", p.name)
        if not m: 
            continue
        rg = m.group(1)
        by_range.setdefault(rg, [])
        if len(by_range[rg]) < max_per_range:
            by_range[rg].append(p)
    return by_range

def main():
    if len(sys.argv) < 2:
        print("Uso: python qwen_descriptions.py <carpeta_output>")
        sys.exit(1)

    nombre = sys.argv[1]
    carpeta = DETECCIONES / nombre
    personas_dir = carpeta / "personas"
    frames_dir   = carpeta / "frames"

    in_dir = personas_dir if personas_dir.exists() else frames_dir
    if not in_dir.exists():
        print(f"❌ No hay imágenes en {personas_dir} ni en {frames_dir}")
        sys.exit(0)

    by_range = _pick_images(in_dir, max_per_range=3)

    rango_labels = {}
    for rg, paths in sorted(by_range.items()):
        etiqueta = "NO SOSPECHOSO"
        for p in paths:
            prompt = _ollama_image_prompt(str(p))
            try:
                txt = _gen(prompt)
                if _is_refusal(txt):
                    txt = _gen(prompt + "\n\nIMPORTANTE: Contesta solo 'SOSPECHOSO: ...' o 'NO SOSPECHOSO'.")
                t = txt.strip()

                if t.upper().startswith("SOSPECHOSO"):
                    etiqueta = "SOSPECHOSO"
                    dst = carpeta / f"{rg}_robo_sospecha_{p.name}"
                    try:
                        if not dst.exists():
                            dst.write_bytes(p.read_bytes())
                    except Exception:
                        pass
                    break
            except Exception as e:
                continue

        rango_labels[rg] = etiqueta

    (carpeta / "qwen_descriptions_ranges.json").write_text(
        json.dumps(rango_labels, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    por_imagen = {p.name: rango_labels.get(re.match(r"^(\d{4}-\d{4})_", p.name).group(1), "NO SOSPECHOSO")
                  for rg, paths in by_range.items() for p in paths}
    (carpeta / "qwen_descriptions.json").write_text(
        json.dumps(por_imagen, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ Guardado: qwen_descriptions.json y qwen_descriptions_ranges.json")

if __name__ == "__main__":
    main()





