"""
cran_escenario.py
Primera parte del proyecto C-RAN:
  1) Carga estaciones_raw.csv y agrupa estaciones por emplazamiento (mismas coordenadas).
  2) Inyecta RRHs sinteticos (numero variable, distancia minima entre RRHs).
  3) Elige K BBU pools con farthest-first k-center (K variable) y compara varios K.
  4) Guarda CSV/JSON, grafica distancia maxima vs K y genera un mapa Folium.
 
Dependencias: numpy, pandas, matplotlib, folium
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------- Parametros ----------------
CSV_ENTRADA = "estaciones_tec.csv"  # tras anadir_tecnologias.py usar "estaciones_tec.csv"
# Filtro por tecnologia (solo si CSV_ENTRADA es estaciones_tec.csv). Etiquetas posibles:
# "4G", "5G", "4G+5G", "ambigua" (solo 900/2100 MHz), "sin_datos". None = no filtrar.
FILTRO_ETIQUETAS = ["4G", "5G", "4G+5G"]            # p. ej. ["4G", "5G", "4G+5G"]
N_SINTETICOS = 50            # numero de RRHs sinteticos a inyectar (0 = ninguno)
MIN_DIST_RRH_KM = 0.1        # distancia minima entre dos RRHs al colocar sinteticos
MAX_DIST_A_REAL_KM = 1.0     # radio en el que se cuentan estaciones reales vecinas
MIN_VECINOS_REALES = 3       # un sintetico necesita al menos tantas reales dentro de ese radio
                             # (evita colocarlos en el mar o en zonas vacias; si no se
                             # consiguen todos, bajar MIN_VECINOS_REALES o subir el radio)
K_LISTA = [2, 3, 4, 5, 6, 8, 10, 15, 20]   # valores de K a comparar
K_ELEGIDO = 10                # K usado para el mapa y la asignacion final
SEMILLA = 42
METODO_POOLS = "k-means"    # "k-center": minimiza la distancia MAXIMA (reparte pools por el territorio)
                             # "k-means" : minimiza la distancia MEDIA (mas pools donde hay mas RRHs)
SEMILLA = 42
DECIMALES_AGRUPAR = 5        # ~1 m: estaciones con igual coordenada = mismo emplazamiento


# ---------------- Utilidades geometricas ----------------
def a_km(lon, lat, lon0, lat0):
    """Proyeccion equirectangular local (suficiente para 30 x 30 km)."""
    x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * 111.32
    return np.column_stack([x, y])


def dist_a_punto(P, p):
    return np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])


def farthest_first(P, k, candidatos=None):
    """k-center greedy. Devuelve (indices_pools, asignacion, dist_a_su_pool).

    candidatos: mascara booleana de los RRH que pueden ser BBU pool
    (por defecto todos). Todos los RRH se asignan a su pool mas cercano.
    """
    n = len(P)
    if candidatos is None:
        candidatos = np.ones(n, dtype=bool)
    idx_cand = np.flatnonzero(candidatos)
    k = min(k, len(idx_cand))
    # primer pool: el candidato mas cercano al centroide (determinista)
    c = P.mean(axis=0)
    primero = int(idx_cand[np.argmin(dist_a_punto(P[idx_cand], c))])
    pools = [primero]
    dmin = dist_a_punto(P, P[primero])
    asign = np.zeros(n, dtype=int)
    for idx in range(1, k):
        # el candidato mas lejos de todos los pools elegidos
        nuevo = int(idx_cand[np.argmax(dmin[idx_cand])])
        pools.append(nuevo)
        d = dist_a_punto(P, P[nuevo])
        mejora = d < dmin
        asign[mejora] = idx
        dmin = np.minimum(dmin, d)
    return pools, asign, dmin


def kmeans_pools(P, k, candidatos=None, semilla=0, reinicios=10, iteraciones=100):
    """k-means (Lloyd) y cada pool = RRH candidato mas cercano al centro de su grupo.

    Minimiza la distancia media, asi que pone mas pools donde hay mas RRHs.
    Devuelve (indices_pools, asignacion, dist_a_su_pool), igual que farthest_first.
    """
    n = len(P)
    if candidatos is None:
        candidatos = np.ones(n, dtype=bool)
    idx_cand = np.flatnonzero(candidatos)
    k = min(k, len(idx_cand))
    rng = np.random.default_rng(semilla)

    def matriz(C):
        return np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1])

    mejor = None
    for _ in range(reinicios):
        C = P[rng.choice(idx_cand, k, replace=False)]
        for _ in range(iteraciones):
            a = matriz(C).argmin(axis=1)
            Cn = np.array([P[a == j].mean(axis=0) if np.any(a == j) else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        coste = matriz(C).min(axis=1).sum()
        if mejor is None or coste < mejor[0]:
            mejor = (coste, C)
    pools = []
    for c in mejor[1]:                      # candidato real mas cercano a cada centro (sin repetir)
        d = np.hypot(P[idx_cand, 0] - c[0], P[idx_cand, 1] - c[1])
        for j in np.argsort(d):
            if int(idx_cand[j]) not in pools:
                pools.append(int(idx_cand[j]))
                break
    D = matriz(P[pools])
    return pools, D.argmin(axis=1), D.min(axis=1)


def elegir_pools(P, k, candidatos, metodo, semilla=0):
    if metodo == "k-means":
        return kmeans_pools(P, k, candidatos, semilla)
    if metodo == "k-center":
        return farthest_first(P, k, candidatos)
    raise ValueError(f"METODO_POOLS desconocido: {metodo}")


def inyectar_sinteticos(P_real, n, min_dist, bbox, rng, max_dist_real=None, min_vecinos=1):
    """Puntos uniformes en bbox=(xmin,xmax,ymin,ymax) en km.

    Respetan una distancia minima a cualquier RRH existente y, si se indica
    max_dist_real, deben tener al menos min_vecinos estaciones reales a menos
    de esa distancia.
    """
    if n <= 0:
        return np.empty((0, 2))
    xmin, xmax, ymin, ymax = bbox
    existentes = [p for p in P_real]
    nuevos, intentos = [], 0
    while len(nuevos) < n and intentos < n * 2000:
        intentos += 1
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if max_dist_real is not None and \
                (dist_a_punto(P_real, p) <= max_dist_real).sum() < min_vecinos:
            continue
        todos = np.array(existentes) if existentes else np.empty((0, 2))
        if len(todos) == 0 or dist_a_punto(todos, p).min() >= min_dist:
            existentes.append(p)
            nuevos.append(p)
    if len(nuevos) < n:
        print(f"AVISO: solo se pudieron colocar {len(nuevos)} de {n} sinteticos")
    return np.array(nuevos) if nuevos else np.empty((0, 2))


# ---------------- Programa principal ----------------
def main():
    rng = np.random.default_rng(SEMILLA)

    # 1) Cargar y agrupar por emplazamiento
    df = pd.read_csv(CSV_ENTRADA, dtype={"site_id": str})
    df = df.dropna(subset=["lon", "lat"])
    if FILTRO_ETIQUETAS is not None:
        if "etiqueta" not in df.columns:
            raise SystemExit(
                f"{CSV_ENTRADA} no tiene la columna 'etiqueta'. Ejecuta primero "
                "anadir_tecnologias.py y pon CSV_ENTRADA = 'estaciones_tec.csv', "
                "o deja FILTRO_ETIQUETAS = None."
            )
        antes = len(df)
        df = df[df["etiqueta"].isin(FILTRO_ETIQUETAS)]
        print(f"Filtro {FILTRO_ETIQUETAS}: {antes} -> {len(df)} estaciones")
    df["clave"] = df["lon"].round(DECIMALES_AGRUPAR).astype(str) + "," + df["lat"].round(DECIMALES_AGRUPAR).astype(str)
    sitios = (
        df.groupby("clave")
        .agg(lon=("lon", "first"), lat=("lat", "first"),
             n_estaciones=("id_estacion", "count"),
             operadores=("operador", lambda s: "; ".join(sorted(set(map(str, s))))))
        .reset_index(drop=True)
    )
    sitios["tipo"] = "real"
    print(f"Estaciones: {len(df)}  ->  emplazamientos (RRH reales): {len(sitios)}")

    lon0, lat0 = sitios["lon"].mean(), sitios["lat"].mean()
    P_real = a_km(sitios["lon"], sitios["lat"], lon0, lat0)

    # 2) RRHs sinteticos dentro del bounding box del escenario
    bbox = (P_real[:, 0].min(), P_real[:, 0].max(), P_real[:, 1].min(), P_real[:, 1].max())
    P_sint = inyectar_sinteticos(P_real, N_SINTETICOS, MIN_DIST_RRH_KM, bbox, rng,
                                 max_dist_real=MAX_DIST_A_REAL_KM,
                                 min_vecinos=MIN_VECINOS_REALES)
    sint = pd.DataFrame({
        "lon": lon0 + P_sint[:, 0] / (111.32 * np.cos(np.radians(lat0))) if len(P_sint) else [],
        "lat": lat0 + P_sint[:, 1] / 111.32 if len(P_sint) else [],
        "n_estaciones": 0, "operadores": "", "tipo": "sintetico",
    })
    rrh = pd.concat([sitios.drop(columns=[]), sint], ignore_index=True)
    rrh.insert(0, "rrh_id", range(len(rrh)))
    P = np.vstack([P_real, P_sint]) if len(P_sint) else P_real
    print(f"RRHs totales: {len(rrh)} ({len(sitios)} reales + {len(P_sint)} sinteticos)")

    # Solo los RRH reales pueden ser BBU pool (los reales van primero en P)
    candidatos = np.arange(len(P)) < len(P_real)

    # 3) Comparar varios K
    resultados = []
    for k in K_LISTA:
        if k > len(P_real):
            continue
        _, _, d = elegir_pools(P, k, candidatos, METODO_POOLS, SEMILLA)
        resultados.append({"K": k, "dist_max_km": d.max(), "dist_media_km": d.mean()})
    res = pd.DataFrame(resultados)
    print(res.to_string(index=False))
    res.to_csv("resultados_k.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(res["K"], res["dist_max_km"], "o-", label="Distancia maxima")
    plt.plot(res["K"], res["dist_media_km"], "s--", label="Distancia media")
    plt.xlabel("Numero de BBU pools (K)")
    plt.ylabel("Distancia RRH - BBU pool (km)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("distancia_vs_K.png", dpi=150)
    plt.close()

    # 4) Asignacion final con K_ELEGIDO
    pools, asign, d = elegir_pools(P, K_ELEGIDO, candidatos, METODO_POOLS, SEMILLA)
    print(f"Metodo de pools: {METODO_POOLS}")
    rrh["bbu_pool"] = asign
    rrh["dist_a_pool_km"] = d
    rrh["es_pool"] = False
    rrh.loc[pools, "es_pool"] = True
    rrh.to_csv("rrh_escenario.csv", index=False)
    rrh.to_json("rrh_escenario.json", orient="records", force_ascii=False, indent=2)
    print(f"K={K_ELEGIDO}: dist max = {d.max():.2f} km, media = {d.mean():.2f} km")

    # 5) Mapa Folium
    try:
        import folium
    except ImportError:
        print("folium no instalado: se omite el mapa (pip install folium)")
        return
    colores = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue",
               "darkgreen", "pink", "gray", "black", "lightblue"]
    # Fondo Esri a color (sin clave de API). Otras opciones: sustituir
    # World_Street_Map por World_Topo_Map (relieve), World_Imagery (satelite)
    # o Canvas/World_Light_Gray_Base (gris claro).
    m = folium.Map(
        location=[lat0, lon0], zoom_start=10,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
    )
    for _, r in rrh.iterrows():
        col = colores[int(r["bbu_pool"]) % len(colores)]
        if r["es_pool"]:
            folium.Marker([r["lat"], r["lon"]], icon=folium.Icon(color=col, icon="cloud"),
                          popup=f"BBU pool {int(r['bbu_pool'])} (RRH {int(r['rrh_id'])})").add_to(m)
        else:
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=3 if r["tipo"] == "real" else 5,
                color=col if r["tipo"] == "real" else "black",   # sinteticos: borde negro
                fill=True, fill_color=col, fill_opacity=0.8,
                weight=1 if r["tipo"] == "real" else 2,
                popup=f"RRH {int(r['rrh_id'])} ({r['tipo']}) -> pool {int(r['bbu_pool'])}<br>{r['operadores']}",
            ).add_to(m)
    m.save("mapa_escenario.html")
    print("Mapa guardado en mapa_escenario.html")


if __name__ == "__main__":
    main()