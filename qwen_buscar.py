import os, sys, json, re, unicodedata
from collections import defaultdict

def normalize(s: str) -> str:
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('\n', ' ')
    return s

def expand_terms(q: str) -> set:
    tokens = [t for t in re.split(r'\s+', q) if t]
    expanded = set()
    for t in tokens:
        expanded.add(t)
        # -->singular/plural
        if t.endswith('es') and len(t) > 3: expanded.add(t[:-2])
        if t.endswith('s') and len(t) > 2: expanded.add(t[:-1])
        if not t.endswith('s'): expanded.add(t + 's')
    # -->sinónimos útiles
    syn = {
        'persona': {'persona','personas','gente','individuo','individuos','person'},
        'rostro': {'rostro','cara','face','rostros','caras'},
        'sospechoso': {'sospechoso','sospechosa','sospechosos','sospechosas','hurto','robo','saqueo','mechero','mecheros'},
        'mochila': {'mochila','bolso','cartera','bag','backpack'},
        'camara': {'camara','cctv'},
        'producto': {'producto','articulo','mercaderia','item','items'},
    }
    for t in list(expanded):
        if t in syn:
            expanded.update(syn[t])
    return expanded

def score_match(text: str, terms: set) -> int:
    return sum(1 for t in terms if t in text)

def main():
    if len(sys.argv) < 3:
        print("Uso: python qwen_buscar.py <carpeta> <termino_busqueda>")
        sys.exit(1)

    carpeta = sys.argv[1]
    termino = sys.argv[2]

    json_path = os.path.join("detecciones", carpeta, "qwen_descriptions.json")
    if not os.path.exists(json_path):
        print(f"No se encontró el archivo {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q_norm = normalize(termino)
    terms = expand_terms(q_norm)

    results = []
    for nombre, descripcion in data.items():
        desc_norm = normalize(descripcion)
        name_norm = normalize(nombre)

        sc = score_match(desc_norm, terms) + score_match(name_norm, terms)

        if any(t.startswith('sospech') or t in ('robo','hurto','saqueo') for t in terms):
            if 'clasificacion: sospechoso' in desc_norm:
                sc += 5

        if sc > 0:
            results.append((sc, nombre))

    results.sort(key=lambda x: (-x[0], x[1]))
    coincidentes = [n for _, n in results]

    salida_path = os.path.join("detecciones", carpeta, "qwen_busqueda.json")
    with open(salida_path, "w", encoding="utf-8") as f:
        json.dump(coincidentes, f, ensure_ascii=False, indent=2)

    print(json.dumps(coincidentes, ensure_ascii=False))

if __name__ == "__main__":
    main()



