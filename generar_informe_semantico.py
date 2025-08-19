import os
import sys
import json
import re
import base64
import requests
from collections import defaultdict
from typing import List, Tuple
import shutil

FAST_MODE = os.getenv("FAST_MODE", "1") == "1"
VLM_MODEL   = os.getenv("VLM_MODEL", "qwen2.5vl:latest")
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "120"))

SMART_VERIFY_ON_AMBIGUOUS = True
OLLAMA_URL = "http://localhost:11434/api/generate"

MAX_IMAGES_PER_RANGE = 1 if FAST_MODE else 3
DEBUG = True

# Señales de hurto
SUSPECT_KEYWORDS = [
    r"\bhurt\w+\b", r"\brob\w+\b", r"\bmecher\w+\b", r"\bladron\w*\b",
    r"\bsin\s+pagar\b", r"\bno\s+paga\b", r"\bpasa\s+por\s*caja\s+sin\s+pagar\b",
    r"\bsale\s+(corriend\w+|arranc\w+|a\s+la\s+fuga)\b", r"\bsaque\w+\b",
    r"\b(ocult\w+|escond\w+)\s+(producto|medicamento|art[íi]culo)\b",
    r"\b(met\w+|guard\w+)\s+(en|dentro\s+de|entre)\s+(mochila|bolsillo|cartera|chaqueta|ropa|carrito|bolsa|pañalera)\b",
    r"\b(romp\w+|abr\w+)\s+(bl[íi]ster|empaque|envase|seguro|antirrobo|etiqueta)\b",
    r"\b(cort\w+)\s+(etiqueta|alarma)\b",
    r"\b(desactiv\w+)\s+alarma\b",
    r"\b(retir\w+)\s+(antirrobo|seguro)\b",
    r"\b(cambi\w+|intercambi\w+)\s+(precio|etiqueta)\b",
    r"\b(forz\w+|fuerz\w+)\s+(cerradura|puerta|vitrina|candado|caja)\b",
    r"\b(cubr\w+)\s+c[áa]mara\b",
    r"\bshoplift(ing|er)?\b", r"\bsteal(s|ing|er)?\b", r"\btheft\b",
    r"\bconceal(s|ed|ing)?\b", r"\bwithout\s+pay(ing)?\b",
    r"\bputs?\s+(item|product)\s+in\s+(bag|pocket|backpack)\b",
    r"\bruns?\s+away\b", r"\bcut(s|ting)?\s+tag\b", r"\bdisable(s|d|ing)?\s+alarm\b",
    r"\bbreak(s|ing)?\s+(lock|seal|tag|glass|case)\b",
]

# Indicadores benignos
BENIGN_HINTS = [
    r"\bconsulta\s+con\s+(qu[íi]mico|farmac[ée]utico)\b",
    r"\blee\s+(prospecto|instructivo)\b",
    r"\bcompara\s+precios?\b",
    r"\bse\s+prueba\s+(crema|loci[óo]n)\s+(en|sobre)\s+la\s+mano\b",
    r"\bhabla(n)?\s+con\s+vendedor(a)?\b",
    r"\best[aá]\s+en\s+fila\b",
]

# Pistas de acción"
ACTION_HINTS = [
    r"\b(toma|agarra|saca|extrae|retira)\b",
    r"\b(manipula|mueve|cambia)\b",
    r"\b(mete|guarda|oculta|esconde)\b",
    r"\b(bolsa|mochila|bolsillo|chaqueta)\b",
    r"\b(abre|rompe|corta)\b",
    r"\b(desactiva|quita)\s+(alarma|etiqueta|antirrobo)\b",
    r"\b(sale|se retira|abandona)\b",
    r"\b(puts?|place?s?)\b",
    r"\b(bag|pocket|backpack)\b",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECCIONES_FOLDER = os.path.join(BASE_DIR, "detecciones")

def mmss_to_hhmmss(mmss: str) -> str:
    if len(mmss) != 4 or not mmss.isdigit():
        return mmss
    m = int(mmss[:2]); s = int(mmss[2:])
    total = m * 60 + s
    hh = total // 3600; mm = (total % 3600) // 60; ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh > 0 else f"{mm:02d}:{ss:02d}"

def group_images_by_range(personas_dir: str):
    grupos = defaultdict(list)
    if not os.path.exists(personas_dir):
        return grupos
    for fname in sorted(os.listdir(personas_dir)):
        fpath = os.path.join(personas_dir, fname)
        if not os.path.isfile(fpath): continue
        m = re.match(r"^(\d{4}-\d{4})[_\-].+|^(\d{4}-\d{4})\.", fname)
        key = m.group(1) or m.group(2) if m else None
        if not key:
            p = fname.split("_")[0]
            if re.match(r"^\d{4}-\d{4}$", p): key = p
        if key: grupos[key].append(fname)
    return grupos

def load_qwen_descriptions(json_path: str):
    if not os.path.exists(json_path): return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {os.path.basename(k): v for k, v in data.items()}

def load_ranges_labels(json_path: str):
    if not os.path.exists(json_path): return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for rk, info in data.items():
        cls = (info.get("clasificacion") or "").strip().upper()
        if "SOSPECHOSO" in cls: out[rk] = "SOSPECHOSO"
        elif "NO" in cls:      out[rk] = "NO SOSPECHOSO"
    return out

def contains_suspect_keywords(text: str) -> bool:
    t = (text or "").lower()
    if any(re.search(p, t) for p in BENIGN_HINTS) and not any(re.search(p, t) for p in SUSPECT_KEYWORDS):
        return False
    return any(re.search(p, t) for p in SUSPECT_KEYWORDS)

def contains_action_hints(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in ACTION_HINTS)

def _b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

def verify_with_vlm(range_key: str, descriptions, image_b64s) -> bool:
    if not descriptions and not image_b64s:
        return False
    prompt = (
        "Eres analista de seguridad en farmacias de Chile. Clasifica este rango como HAY INDICIOS DE HURTO/MOVIMIENTO SOSPECHOSO "
        "o NO SOSPECHOSO usando SOLO la evidencia provista.\n"
        f"Rango: {range_key}\nDescripciones:\n- " + "\n- ".join(descriptions or ['(sin descripciones)']) +
        "\n\nResponde SOLO una línea EXACTA: CLASIFICACION: SOSPECHOSO | CLASIFICACION: NO SOSPECHOSO"
    )
    try:
        payload = {
            "model": VLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 128, "num_ctx": 4096},
            "keep_alive": "20m"
        }
        if image_b64s:
            payload["images"] = image_b64s[:MAX_IMAGES_PER_RANGE]
        resp = requests.post(OLLAMA_URL, json=payload, timeout=VLM_TIMEOUT)
        if resp.status_code != 200:
            return False
        out = (resp.json().get("response") or "").strip().upper()
        return "SOSPECHOSO" in out
    except Exception:
        return False

def copy_suspect_images(base_dir: str, range_key: str, fnames: List[str], max_copies: int = 2):
    personas_dir = os.path.join(base_dir, "personas")
    if not os.path.isdir(personas_dir):
        return []
    made = []
    count = 0
    for fn in sorted(fnames):
        if count >= max_copies:
            break
        src = os.path.join(personas_dir, fn)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(personas_dir, f"{range_key}_robo_sospecha_{count+1}.jpg")
        try:
            shutil.copyfile(src, dst)
            made.append(os.path.basename(dst))
            count += 1
        except Exception:
            pass
    return made

def main():
    if len(sys.argv) < 2:
        print("Uso: python generar_informe_semantico.py <carpeta>")
        sys.exit(0)

    carpeta = sys.argv[1]
    carpeta_dir = os.path.join(DETECCIONES_FOLDER, carpeta)
    personas_dir = os.path.join(carpeta_dir, "personas")
    descriptions_path = os.path.join(carpeta_dir, "qwen_descriptions.json")
    ranges_json_path = os.path.join(carpeta_dir, "qwen_descriptions_ranges.json")
    out_txt = os.path.join(carpeta_dir, "personas_detectadas.txt")

    grupos = defaultdict(list)
    for fname in sorted(os.listdir(personas_dir)) if os.path.isdir(personas_dir) else []:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
        m = re.match(r"^(\d{4}-\d{4})[_\-].+|^(\d{4}-\d{4})\.", fname)
        key = m.group(1) or m.group(2) if m else None
        if not key:
            p = fname.split("_")[0]
            if re.match(r"^\d{4}-\d{4}$", p): key = p
        if key: grupos[key].append(fname)

    desc_map = load_qwen_descriptions(descriptions_path)
    ranges_map = load_ranges_labels(ranges_json_path)

    resultados = []
    debug_dump = {}

    for range_key, fnames in sorted(grupos.items()):
        hay_personas = len(fnames) > 0

        cls_qwen = ranges_map.get(range_key)
        descripciones_rango = [desc_map.get(fn) for fn in fnames if desc_map.get(fn)]

        kw_hit = any(contains_suspect_keywords(d) for d in descripciones_rango)
        flag_text = any("CLASIFICACION: SOSPECHOSO" in (d or "").upper() for d in descripciones_rango)
        action_hint = any(contains_action_hints(d) for d in descripciones_rango)

        sospechoso = (cls_qwen == "SOSPECHOSO") or kw_hit or flag_text

        if (not sospechoso) and hay_personas and FAST_MODE and SMART_VERIFY_ON_AMBIGUOUS and action_hint:
            imgs_abs = [os.path.join(personas_dir, x) for x in fnames[:1]]
            image_b64s = []
            for p in imgs_abs:
                b = _b64(p)
                if b: image_b64s.append(b)
            qwen_flag = verify_with_vlm(range_key, descripciones_rango, image_b64s)
            sospechoso = sospechoso or qwen_flag

        if sospechoso and fnames:
            copy_suspect_images(carpeta_dir, range_key, fnames, max_copies=2 if FAST_MODE else 3)

        resultados.append((range_key, hay_personas, sospechoso))

        if DEBUG:
            debug_dump[range_key] = {
                "imgs": len(fnames),
                "desc": len(descripciones_rango),
                "kw_hit": bool(kw_hit),
                "flag_text": bool(flag_text),
                "cls_qwen": cls_qwen or "",
                "action_hint": bool(action_hint),
                "smart_verify": bool((not ((cls_qwen == "SOSPECHOSO") or kw_hit or flag_text)) and hay_personas and FAST_MODE and SMART_VERIFY_ON_AMBIGUOUS and action_hint),
                "resultado_final": "SOSPECHOSO" if sospechoso else "NO"
            }

    resultados.sort(key=lambda x: x[0])

    lines = []
    for key, hay_personas, sospechoso in resultados:
        start, end = key.split("-")
        def to_hhmmss(mmss: str) -> str:
            m = int(mmss[:2]); s = int(mmss[2:])
            total = m*60 + s
            hh = total // 3600; mm = (total % 3600) // 60; ss = total % 60
            return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh > 0 else f"{mm:02d}:{ss:02d}"
        t0 = to_hhmmss(start); t1 = to_hhmmss(end)
        lines.append(f"Entre {t0} - {t1} segundos: " + ("se detectaron personas en movimiento." if hay_personas else "no se detectaron personas."))
        if sospechoso: lines.append("⚠️ SE DETECTARON MOVIMIENTOS SOSPECHOSOS (posible hurto).")
        lines.append("")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    if DEBUG:
        try:
            dbg_path = os.path.join(carpeta_dir, "informe_semantico_debug.json")
            with open(dbg_path, "w", encoding="utf-8") as f:
                json.dump(debug_dump, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    print(f"Informe generado: {out_txt}")
    print("Listo.")

if __name__ == "__main__":
    main()



