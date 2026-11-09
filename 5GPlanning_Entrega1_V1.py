"""
infoantenas_scraper.py
Descarga estaciones base de Infoantenas por teselas y genera CSV + JSON.

1) Ejecutar con MODO_PRUEBA = True  -> imprime la respuesta cruda de UNA tesela.
2) Cuando se vea la estructura, poner MODO_PRUEBA = False -> recorre toda la zona.

Dependencias: requests, pandas
"""
import json
import math
import time

import pandas as pd
import requests

# ---- Parametros ----
MODO_PRUEBA = False
LAT0, LON0 = 41.3926, 2.1542      # centro de la zona (ahora Barcelona) -> CAMBIAR
ANCHO_KM, ALTO_KM = 30, 30        # zona aprox. 30 x 30 km
D_LON, D_LAT = 0.02, 0.03         # tamano de tesela (similar al del navegador)
ZOOM = 6                          # mismo valor que vimos en el navegador
PAUSA = 0.5                       # segundos entre peticiones

BASE = "https://geoportal.minetur.gob.es/VCTEL/infoantenasGeoJSON.do"
HEADERS = {
    "Referer": "https://geoportal.minetur.gob.es/VCTEL/vcne.do",
    "User-Agent": "Mozilla/5.0",
}


def frange(start, stop, step):
    """Lista de valores desde start hasta stop (incluido el ultimo borde)."""
    n = (math.ceil((stop - start) / step))
    return [start + i * step for i in range(n + 1)]


def pedir_tesela(session, lon_min, lat_min, lon_max, lat_max):
    params = {
        "idCapa": "null",
        "bbox": f"{lon_min:.10f},{lat_min:.10f},{lon_max:.10f},{lat_max:.10f}",
        "zoom": ZOOM,
    }
    r = session.get(BASE, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def parsear(texto):
    """GeoJSON -> lista de dicts planos (lon, lat + todas las propiedades)."""
    data = json.loads(texto)
    filas = []
    for f in data.get("features", []):
        fila = {"lon": None, "lat": None}
        try:
            lon, lat = f["geometry"]["coordinates"][:2]
            fila["lon"], fila["lat"] = lon, lat
        except (KeyError, TypeError, ValueError):
            pass
        for k, v in (f.get("properties") or {}).items():
            fila[k] = v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v)

        # Campos derivados utiles
        codigo = str(fila.get("Código") or f.get("id") or "")
        fila["id_estacion"] = codigo
        # "OPERADOR - ID": el ID es lo ultimo tras " - "
        if " - " in codigo:
            operador, _, sid = codigo.rpartition(" - ")
            fila["operador"], fila["site_id"] = operador.strip(), sid.strip()
        direccion = str(fila.get("Dirección") or "")
        if "," in direccion:
            partes = [p.strip() for p in direccion.split(",")]
            muni = partes[-2] if len(partes) >= 2 else ""
            # "96. CASTELLDEFELS" -> "CASTELLDEFELS" (quitar numero de calle)
            fila["municipio"] = muni.split(". ")[-1].strip()
            fila["provincia"] = partes[-1]
        detalle = str(fila.get("Detalle") or "")
        fila["detalle_url"] = detalle.replace(
            "@@<url-aplicacion>", "https://geoportal.minetur.gob.es/VCTEL"
        )
        filas.append(fila)
    return filas


def main():
    half_lat = (ALTO_KM / 2) / 111.32
    half_lon = (ANCHO_KM / 2) / (111.32 * math.cos(math.radians(LAT0)))
    lons = frange(LON0 - half_lon, LON0 + half_lon, D_LON)
    lats = frange(LAT0 - half_lat, LAT0 + half_lat, D_LAT)
    n_tiles = (len(lons) - 1) * (len(lats) - 1)
    print(f"Teselas: {len(lons) - 1} x {len(lats) - 1} = {n_tiles}")

    session = requests.Session()

    if MODO_PRUEBA:
        txt = pedir_tesela(session, lons[0], lats[0], lons[1], lats[1])
        print(f"Longitud respuesta: {len(txt)} caracteres")
        print(txt[:1500])
        return

    filas = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            try:
                txt = pedir_tesela(session, lons[i], lats[j], lons[i + 1], lats[j + 1])
                filas += parsear(txt)
            except Exception as e:  # noqa: BLE001
                print(f"AVISO: tesela ({i},{j}) fallo: {e}")
            time.sleep(PAUSA)
        print(f"Columna {i + 1}/{len(lons) - 1}, filas acumuladas: {len(filas)}")

    df = pd.DataFrame(filas)
    # Una estacion = un id_estacion (operador + ID). Las teselas se solapan en bordes.
    df = df.drop_duplicates(subset="id_estacion").reset_index(drop=True)
    print(f"Estaciones unicas: {len(df)}")
    print(list(df.columns))

    df.to_csv("estaciones_raw.csv", index=False, encoding="utf-8")
    df.to_json("estaciones_raw.json", orient="records", force_ascii=False, indent=2)


if __name__ == "__main__":
    main()