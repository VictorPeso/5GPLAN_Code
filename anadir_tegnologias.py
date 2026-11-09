"""
anadir_tecnologias.py
Segunda fase: para cada estacion de estaciones_raw.csv consulta su pagina de detalle
en Infoantenas, extrae las bandas asignadas (MHz) y DEDUCE la tecnologia a partir de ellas.

Importante: Infoantenas NO indica 4G/5G, solo bandas. La clasificacion es una estimacion
basada en como se usan esas bandas en Espana (ver clasificar_bandas).

Salida: estaciones_tec.csv (todas las columnas de estaciones_raw.csv + bandas y etiquetas).
Se guarda una cache (detalle_cache.json): si se corta, al volver a ejecutar continua donde iba.

Dependencias: requests, pandas
"""
import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

CSV_ENTRADA = "estaciones_raw.csv"
CSV_SALIDA = "estaciones_tec.csv"
CACHE = "detalle_cache.json"
HILOS = 4            # peticiones en paralelo (no subir mucho: es un servidor publico)
PAUSA = 0.3          # segundos de pausa por peticion en cada hilo
HEADERS = {
    "Referer": "https://geoportal.minetur.gob.es/VCTEL/vcne.do",
    "User-Agent": "Mozilla/5.0",
}


# ---------------- Lectura de la pagina de detalle ----------------
def html_a_texto(h):
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)
    h = re.sub(r"(?i)<br\s*/?>|</tr>|</p>|</h\d>|</li>", "\n", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    h = html.unescape(h)
    h = re.sub(r"[ \t\r\f\v]+", " ", h)
    h = re.sub(r"\n\s*\n+", "\n", h)
    return h.strip()


def extraer_bandas(texto):
    """Devuelve lista de [ini, fin] en MHz de la tabla 'Banda Asignada (MHz)'."""
    t = texto.upper()
    ini = t.find("CARACTERISTICAS")
    fin = t.find("NIVELES MEDIDOS")
    seccion = texto[ini:fin] if ini >= 0 and fin > ini else texto
    pares = re.findall(r"(\d+\.\d+)\s*-\s*(\d+\.\d+)", seccion)
    return [[float(a), float(b)] for a, b in pares]


def consultar(item):
    id_est, url = item
    time.sleep(PAUSA)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        r.encoding = "utf-8"
        return id_est, {"bandas": extraer_bandas(html_a_texto(r.text))}
    except Exception as e:  # noqa: BLE001
        return id_est, {"error": str(e)}


# ---------------- Clasificacion por banda ----------------
def clasificar_bandas(bandas):
    """
    Deduce la tecnologia a partir del centro de cada banda (MHz). ESTIMACION:
      5G       : 700 MHz (694-790) y 3500 MHz (3300-3800), o mmWave (>= 24000)
      4G       : 800 MHz (791-862), 1800 MHz (1710-1880), 2600 MHz (2500-2690)
      ambigua  : 900 MHz (880-960) y 2100 MHz (2110-2170): se usan para 2G/3G y tambien 4G
    Un 1800 MHz tambien puede ser 2G; un 700 MHz podria llevar 4G. Es una aproximacion.
    """
    tiene_5g = tiene_4g = tiene_amb = False
    for ini, fin in bandas:
        c = (ini + fin) / 2
        if 694 <= c <= 790 or 3300 <= c <= 3800 or c >= 24000:
            tiene_5g = True
        elif 791 <= c <= 862 or 1710 <= c <= 1880 or 2500 <= c <= 2690:
            tiene_4g = True
        elif 880 <= c <= 960 or 2110 <= c <= 2170:
            tiene_amb = True
    if tiene_5g and tiene_4g:
        etiqueta = "4G+5G"
    elif tiene_5g:
        etiqueta = "5G"
    elif tiene_4g:
        etiqueta = "4G"
    elif tiene_amb:
        etiqueta = "ambigua"      # solo 900/2100: no se puede asegurar 4G
    else:
        etiqueta = "sin_datos"
    return tiene_4g or tiene_5g, tiene_5g, etiqueta


def main():
    df = pd.read_csv(CSV_ENTRADA)

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)

    pendientes = [
        (r["id_estacion"], r["detalle_url"])
        for _, r in df.iterrows()
        if "bandas" not in cache.get(r["id_estacion"], {})
    ]
    print(f"Estaciones: {len(df)} | ya en cache: {len(df) - len(pendientes)} | por consultar: {len(pendientes)}")

    hechas = 0
    try:
        with ThreadPoolExecutor(max_workers=HILOS) as ex:
            futuros = [ex.submit(consultar, it) for it in pendientes]
            for fut in as_completed(futuros):
                id_est, res = fut.result()
                cache[id_est] = res
                hechas += 1
                if hechas % 100 == 0:
                    with open(CACHE, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)
                    print(f"  {hechas}/{len(pendientes)}")
    except KeyboardInterrupt:
        print("Interrumpido: se guarda lo hecho hasta ahora.")
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    # Construir columnas nuevas
    bandas_txt, es_4g5g, es_5g, etiquetas = [], [], [], []
    for _, r in df.iterrows():
        info = cache.get(r["id_estacion"], {})
        bandas = info.get("bandas")
        if bandas is None:
            bandas_txt.append("")
            es_4g5g.append(False)
            es_5g.append(False)
            etiquetas.append("sin_datos")
            continue
        a, b, e = clasificar_bandas(bandas)
        bandas_txt.append("; ".join(f"{i:g}-{f:g}" for i, f in bandas))
        es_4g5g.append(a)
        es_5g.append(b)
        etiquetas.append(e)
    df["bandas_mhz"] = bandas_txt
    df["soporta_4g_o_5g"] = es_4g5g
    df["soporta_5g"] = es_5g
    df["etiqueta"] = etiquetas

    df.to_csv(CSV_SALIDA, index=False, encoding="utf-8")
    print(f"\nGuardado {CSV_SALIDA}")
    print(df["etiqueta"].value_counts().to_string())
    fallos = sum(1 for v in cache.values() if "error" in v)
    if fallos:
        print(f"\nAVISO: {fallos} estaciones dieron error; vuelve a ejecutar para reintentarlas.")


if __name__ == "__main__":
    main()