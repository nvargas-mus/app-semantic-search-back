from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess
from uuid import uuid4
from flask_cors import CORS
import json
from flask_cors import cross_origin
import sys
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_FOLDER = os.path.join(BASE_DIR, 'videos')
DETECCIONES_FOLDER = os.path.join(BASE_DIR, 'detecciones')
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(DETECCIONES_FOLDER, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

def ollama_disponible() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def limpiar_cache_semantica(output_folder):
    try:
        objetivos = [
            "qwen_descriptions.json",
            "qwen_descriptions_ranges.json",
            "qwen_busqueda.json",
            "resumen_video.txt",
            "informe_semantico_debug.json",
        ]
        for fname in objetivos:
            p = os.path.join(output_folder, fname)
            if os.path.exists(p):
                os.remove(p)
        personas_dir = os.path.join(output_folder, "personas")
        if os.path.isdir(personas_dir):
            for fn in os.listdir(personas_dir):
                if "_robo_sospecha_" in fn:
                    try:
                        os.remove(os.path.join(personas_dir, fn))
                    except Exception:
                        pass
    except Exception:
        pass

procesos_activos = {}

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No se encontró el archivo de video."}), 400

    video = request.files['video']
    filename = video.filename
    if not filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        return jsonify({"error": "Formato de video no soportado."}), 400

    video_path = os.path.join(VIDEOS_FOLDER, filename)
    video.save(video_path)

    try:
        size = os.path.getsize(video_path)
    except Exception:
        size = -1
    if size <= 0:
        return jsonify({"error": "El archivo de video parece estar vacío o no se pudo guardar correctamente."}), 400

    job_id = str(uuid4())

    try:
        process = subprocess.Popen(
            [sys.executable, 'people_detector.py', video_path],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        procesos_activos[job_id] = process

        stdout, stderr = process.communicate()
        exit_code = process.returncode

        procesos_activos.pop(job_id, None)

        if exit_code != 0:
            print("people_detector exit code:", exit_code)
            print("----- STDOUT -----\n", stdout)
            print("----- STDERR -----\n", stderr)
            return jsonify({
                "error": "Error al procesar el video",
                "code": exit_code,
                "stdout": stdout,
                "stderr": stderr
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    video_name = os.path.splitext(filename)[0]
    output_folder = os.path.join(DETECCIONES_FOLDER, f"{video_name}_output")

    incluir_busqueda = (request.form.get("incluir_busqueda") == "true")
    incluir_rostros  = (request.form.get("incluir_rostros") == "true")
    fast_mode = request.form.get("analisis_rapido", "true").lower() == "true"
    force_clear = request.form.get("limpiar_cache_semantica", "false").lower() == "true"

    analisis_semantico_estado = {"descripciones": "omitido", "resumen": "omitido", "informe": "omitido"}

    if incluir_busqueda:
        if force_clear:
            limpiar_cache_semantica(output_folder)

        if not ollama_disponible():
            print("⚠️ Ollama no disponible: se omite análisis semántico (modo degradado).")
            analisis_semantico_estado = {
                "descripciones": "omitido_por_ollama_off",
                "resumen": "omitido_por_ollama_off",
                "informe": "omitido_por_ollama_off",
            }
        else:
            try:
                ruta_desc = os.path.join(BASE_DIR, 'qwen_descriptions.py')
                ruta_resumen = os.path.join(BASE_DIR, 'qwen_resumen.py')
                ruta_sem = os.path.join(BASE_DIR, 'generar_informe_semantico.py')

                common_env = {
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "FAST_MODE": "1" if fast_mode else "0",
                }

                try:
                    subprocess.run([sys.executable, ruta_desc, f"{video_name}_output"],
                                   cwd=BASE_DIR, check=True, env=common_env)
                    analisis_semantico_estado["descripciones"] = "ok"
                except subprocess.CalledProcessError as e:
                    analisis_semantico_estado["descripciones"] = f"error:{e}"

                try:
                    subprocess.run([sys.executable, ruta_resumen, f"{video_name}_output"],
                                   cwd=BASE_DIR, check=True, env=common_env)
                    analisis_semantico_estado["resumen"] = "ok"
                except subprocess.CalledProcessError as e:
                    analisis_semantico_estado["resumen"] = f"error:{e}"

                try:
                    subprocess.run([sys.executable, ruta_sem, f"{video_name}_output"],
                                   cwd=BASE_DIR, check=True, env=common_env)
                    analisis_semantico_estado["informe"] = "ok"
                except subprocess.CalledProcessError as e:
                    analisis_semantico_estado["informe"] = f"error:{e}"

                print("✅ Semántica terminada (con tolerancia a fallos).")
            except Exception as e:
                print(f"⚠️ Error orquestando análisis semántico: {e}")

    if incluir_rostros:
        try:
            subprocess.run([sys.executable, 'face_detector.py', f"{video_name}_output"],
                           cwd=BASE_DIR, check=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            print("✅ Detección de rostros completada.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error ejecutando face_detector.py: {e}")

    txt_path = os.path.join(output_folder, 'personas_detectadas.txt')
    if not os.path.exists(txt_path):
        return jsonify({"error": "No se encontró el archivo de salida."}), 500

    sospecha_detectada = False
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if "MOVIMIENTOS SOSPECHOSOS" in content.upper():
            sospecha_detectada = True

    personas_dir = os.path.join(output_folder, 'personas')
    imagenes = sorted(os.listdir(personas_dir)) if os.path.exists(personas_dir) else []
    rostros_dir = os.path.join(output_folder, 'rostros')
    rostros_imgs = sorted(os.listdir(rostros_dir)) if os.path.exists(rostros_dir) else []

    robo_imgs = []
    if os.path.exists(personas_dir):
        for img in sorted(os.listdir(personas_dir)):
            if "_robo_sospecha_" in img and img.lower().endswith((".jpg", ".jpeg", ".png")):
                robo_imgs.append(img)

    return jsonify({
        "resultado": content,
        "imagenes": imagenes,
        "imagenes_robo": robo_imgs,
        "carpeta": f"{video_name}_output",
        "sospecha_detectada": sospecha_detectada,
        "job_id": job_id,
        "imagenes_rostros": rostros_imgs,
        "analisis_semantico_estado": analisis_semantico_estado
    })

@app.route('/cancelar/<job_id>', methods=['POST'])
def cancelar_proceso(job_id):
    proceso = procesos_activos.get(job_id)
    if proceso:
        proceso.terminate()
        procesos_activos.pop(job_id, None)
        return jsonify({"mensaje": f"Proceso {job_id} cancelado correctamente."})
    else:
        return jsonify({"error": "No se encontró el proceso activo con ese ID."}), 404

@app.route('/imagen/<carpeta>/<path:nombre>')
def get_image(carpeta, nombre):
    # Log suave
    try:
        print(f"[/imagen] carpeta={carpeta} nombre={nombre}")
    except Exception:
        pass

    nombre = os.path.basename((nombre or "").strip())
    posibles_subcarpetas = ['personas', 'rostros', 'frames']

    for sub in posibles_subcarpetas:
        path = os.path.join(DETECCIONES_FOLDER, carpeta, sub)
        full_path = os.path.join(path, nombre)
        if os.path.isfile(full_path):
            return send_from_directory(path, nombre)

    lower = nombre.lower()
    for sub in posibles_subcarpetas:
        path = os.path.join(DETECCIONES_FOLDER, carpeta, sub)
        if not os.path.isdir(path):
            continue
        for fn in os.listdir(path):
            if fn.lower() == lower:
                return send_from_directory(path, fn)

    stem, _ = os.path.splitext(nombre)
    for sub in posibles_subcarpetas:
        path = os.path.join(DETECCIONES_FOLDER, carpeta, sub)
        if not os.path.isdir(path):
            continue
        for fn in os.listdir(path):
            if stem and stem in fn:
                return send_from_directory(path, fn)

    return jsonify({"error": "Imagen no encontrada.", "carpeta": carpeta, "nombre": nombre}), 404

@app.route('/descargar/<carpeta>')
def descargar_txt(carpeta):
    path = os.path.join(DETECCIONES_FOLDER, carpeta)
    return send_from_directory(path, 'personas_detectadas.txt', as_attachment=True)

@app.route('/qwen_describir/<carpeta>', methods=['GET'])
def obtener_descripciones_qwen(carpeta):
    path = os.path.join(DETECCIONES_FOLDER, carpeta, "qwen_descriptions.json")
    if not os.path.exists(path):
        return jsonify({"error": "No se encontró el archivo de descripciones."}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/resumen_video/<carpeta>', methods=['GET'])
def obtener_resumen_video(carpeta):
    path = os.path.join(DETECCIONES_FOLDER, carpeta, "resumen_video.txt")
    if not os.path.exists(path):
        return jsonify({"error": "No se encontró el archivo de resumen."}), 404
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()
    return jsonify({"resumen": contenido})

@app.route('/qwen_buscar/<carpeta>', methods=['POST', 'OPTIONS'])
@cross_origin()
def buscar_con_qwen(carpeta):
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "Falta la consulta (query)."}), 400

    try:
        descripciones_path = os.path.join(DETECCIONES_FOLDER, carpeta, "qwen_descriptions.json")

        if not os.path.exists(descripciones_path):
            print("⚠️ No existe qwen_descriptions.json, generando automáticamente...")
            subprocess.run(
                [sys.executable, 'qwen_descriptions.py', carpeta],
                cwd=BASE_DIR,
                check=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )

        ruta_script = os.path.join(BASE_DIR, 'qwen_buscar.py')
        resultado = subprocess.run(
            [sys.executable, ruta_script, carpeta, query],
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

        print("------ STDOUT ------")
        print(resultado.stdout)
        print("------ STDERR ------")
        print(resultado.stderr)

        if resultado.returncode != 0:
            return jsonify({"error": "Error en búsqueda", "detalle": resultado.stderr}), 500

        busqueda_path = os.path.join(DETECCIONES_FOLDER, carpeta, "qwen_busqueda.json")

        if not os.path.exists(busqueda_path) or not os.path.exists(descripciones_path):
            return jsonify({"error": "Faltan archivos de resultado o descripción."}), 404

        with open(busqueda_path, "r", encoding="utf-8") as f:
            nombres_imagenes = [os.path.basename(s.strip()) for s in json.load(f)]  # normaliza

        with open(descripciones_path, "r", encoding="utf-8") as f:
            descripciones = json.load(f)

        resultados = []
        for nombre in nombres_imagenes:
            resultados.append({
                "imagen": nombre,
                "descripcion": descripciones.get(nombre, "Sin descripción disponible."),
                "similitud": 0.95
            })

        return jsonify({"resultados": resultados})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/buscar_semantico_qwen', methods=['POST'])
def buscar_semantico_qwen():
    data = request.get_json()
    carpeta = data.get("carpeta")
    termino = data.get("termino")

    if not carpeta or not termino:
        return jsonify({"error": "Faltan parámetros 'carpeta' o 'termino'."}), 400

    try:
        subprocess.run(
            [sys.executable, 'qwen_buscar.py', carpeta, termino],
            cwd=BASE_DIR,
            check=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Error ejecutando búsqueda: {str(e)}"}), 500

    path_resultado = os.path.join(DETECCIONES_FOLDER, carpeta, "qwen_busqueda.json")
    if not os.path.exists(path_resultado):
        return jsonify({"error": "No se encontraron resultados."}), 404

    with open(path_resultado, "r", encoding="utf-8") as f:
        imagenes = json.load(f)

    return jsonify({"resultados": imagenes})

if __name__ == '__main__':
    app.run(debug=True)







   