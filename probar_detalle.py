"""
probar_detalle.py
Pide la pagina de detalle de 3 estaciones (de operadores distintos) de estaciones_raw.csv,
guarda cada respuesta en detalle_ejemplo_N.html y muestra solo el TEXTO util
(sin el codigo HTML), desde el apartado de localizacion.
"""
import html
import re

import pandas as pd
import requests

CSV = "estaciones_raw.csv"
HEADERS = {
    "Referer": "https://geoportal.minetur.gob.es/VCTEL/vcne.do",
    "User-Agent": "Mozilla/5.0",
}


def html_a_texto(h):
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)   # quitar scripts y estilos
    h = re.sub(r"(?i)<br\s*/?>|</tr>|</p>|</h\d>|</li>", "\n", h)  # saltos de linea utiles
    h = re.sub(r"(?s)<[^>]+>", " ", h)                    # quitar etiquetas
    h = html.unescape(h)
    h = re.sub(r"[ \t\r\f\v]+", " ", h)
    h = re.sub(r"\n\s*\n+", "\n", h)
    return h.strip()


df = pd.read_csv(CSV)
muestra = df.drop_duplicates(subset="operador").head(3)

for n, (_, fila) in enumerate(muestra.iterrows(), start=1):
    url = fila["detalle_url"]
    print("=" * 70)
    print(f"[{n}] {fila['operador']}  ->  {fila['id_estacion']}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.encoding = r.apparent_encoding
        nombre = f"detalle_ejemplo_{n}.html"
        with open(nombre, "w", encoding="utf-8") as f:
            f.write(r.text)
        texto = html_a_texto(r.text)
        # empezar en el apartado de localizacion si existe
        pos = texto.upper().find("LOCALIZACI")
        if pos > 0:
            texto = texto[pos:]
        print(f"Estado: {r.status_code}. Guardado en {nombre}. Texto:\n")
        print(texto[:3000])
    except Exception as e:  # noqa: BLE001
        print("ERROR:", e)