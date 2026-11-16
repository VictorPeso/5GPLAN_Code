"""
fase1_min_delay.py
Fase 1 del proyecto C-RAN: algoritmo "minimise delay".

  1) Construye el escenario a partir de estaciones_tec.csv: filtra por operador (por defecto solo
     Telefonica) y tecnologia, agrupa por emplazamiento (RRH reales), anade N_SINTETICOS RRH
     sinteticas y asigna un patron de trafico (porcentajes variables) a cada RRH.
  2) Fija N_POOLS posiciones candidatas de BBU pool sobre RRH reales (k-means o k-center).
  3) Para cada RRH calcula la lista de pools alcanzables (retardo ida y vuelta <= maximo),
     el pool optimo (el mas cercano) y marca como no servidas las que no llegan a ninguno.
  4) Clasifica las RRH de cada pool en tres tipos:
        A) solo tienen ese pool como opcion
        B) ese pool es el optimo por distancia, pero tienen otras opciones
        C) pueden conectarse a el, pero no es el optimo
  5) Calcula el numero minimo de pools que sirve a todas las RRH (cobertura greedy),
     como cambia al variar el retardo maximo y cuales sobran (retirada de pools uno a uno).
  6) Prueba de robustez con distinto numero de RRH sinteticas y, si se indica
     POOLS_DESACTIVADOS, mapa con la configuracion reducida.
  7) Genera figuras, CSV y mapas Folium.

TODOS los parametros estan en config.py y se pueden sobrescribir con --nombre=valor.
Dependencias: numpy, pandas, matplotlib, folium
"""
from pathlib import Path

import cran_escenario as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

# ---------------- Parametros (se leen de config.py; se pueden sobrescribir con --nombre=valor) ----------------
cfg = ce.cargar_parametros()
N_POOLS = cfg.n_pools
METODO_POOLS = cfg.metodo_pools
N_SINTETICOS = cfg.n_sinteticos
BARRIDO_SINTETICOS = cfg.barrido_sinteticos
RETARDO_RT_MAX_US = float(cfg.retardo_rt_max_us)
US_POR_KM = float(cfg.us_por_km)
PORCENTAJES_TRAFICO = cfg.porcentajes_trafico       # {"mixed": .., "office": .., "residential": ..} o None
SEMILLA_TRAFICO = cfg.semilla_trafico
SEMILLA = cfg.semilla
DIR_FIGURAS = Path(__file__).resolve().parent / cfg.dir_figuras
POOLS_DESACTIVADOS = cfg.pools_desactivados
RETARDO_PODA_ESTRICTO_US = cfg.retardo_poda_estricto_us
RETARDOS_BARRIDO_US = cfg.retardos_barrido_us

COLORES_TRAFICO = {"residential": "#2ca02c", "office": "#1f77b4", "mixed": "#ff7f0e"}


# ---------------- Utilidades ----------------
def a_km(lon, lat, lon0, lat0):
    """Proyeccion equirectangular local (suficiente para 30 x 30 km)."""
    x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * 111.32
    return np.column_stack([x, y])


def distancias(P, Q):
    """Matriz de distancias (km) entre cada punto de P y cada punto de Q."""
    return np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1])


def poda_pools(D, rt_max_us):
    """Retira pools de uno en uno (eliminacion hacia atras) y devuelve la tabla de pasos.

    En cada paso se retira el pool cuya ausencia mantiene servidas a mas RRH y, en caso de
    empate, aumenta menos la distancia media al pool mas cercano. Solo se considera el retardo.
    """
    d_max = rt_max_us / (2 * US_POR_KM)
    n = D.shape[0]

    def metricas(act):
        Da = D[:, act]
        serv = (Da <= d_max).any(axis=1)
        dmin = Da.min(axis=1)
        if not serv.any():
            return 0, np.nan, np.nan
        return int(serv.sum()), dmin[serv].mean(), dmin[serv].max()

    activos = list(range(D.shape[1]))
    s0, m0, x0 = metricas(activos)
    filas = [{"pools_activos": len(activos), "pool_retirado": 0, "rrh_servidas": s0,
                  "pct_servidas": 100 * s0 / n, "dist_media_km": m0, "dist_max_km": x0}]
    while len(activos) > 1:
        mejor = None
        for p in activos:
            s_, m_, x_ = metricas([q for q in activos if q != p])
            clave = (-s_, m_)
            if mejor is None or clave < mejor[0]:
                mejor = (clave, p, s_, m_, x_)
        _, p, s_, m_, x_ = mejor
        activos.remove(p)
        filas.append({"pools_activos": len(activos), "pool_retirado": p + 1, "rrh_servidas": s_,
                          "pct_servidas": 100 * s_ / n, "dist_media_km": m_, "dist_max_km": x_})
    tabla = pd.DataFrame(filas)
    tabla["aumento_dist_media_pct"] = 100 * (tabla.dist_media_km / m0 - 1)
    tabla.insert(0, "retardo_rt_us", rt_max_us)
    return tabla


def cobertura_greedy(alcanzable):
    """Minimo (aprox.) de pools para servir a todas las RRH alcanzables por alguno.

    alcanzable: matriz booleana (n_rrh x n_pools). Devuelve la lista de pools elegidos.
    """
    pendientes = alcanzable.any(axis=1).copy()
    elegidos = []
    while pendientes.any():
        ganancia = (alcanzable & pendientes[:, None]).sum(axis=0)
        j = int(ganancia.argmax())
        elegidos.append(j)
        pendientes &= ~alcanzable[:, j]
    return elegidos


# ---------------- Programa ----------------
def main():
    rng = np.random.default_rng(SEMILLA)
    sitios = ce.cargar_sitios()                                   # RRH reales (operador + tecnologia)
    rrh, P, P_real, lon0, lat0 = ce.construir_rrh(sitios, N_SINTETICOS, rng)
    r = rrh[["rrh_id", "lon", "lat", "n_estaciones", "operadores", "tipo"]].copy()
    n = len(r)

    # Patron de trafico aleatorio con porcentajes variables (PORCENTAJES_TRAFICO) y semilla fija
    if PORCENTAJES_TRAFICO:
        nombres = list(PORCENTAJES_TRAFICO)
        desconocidos = [k for k in nombres if k not in COLORES_TRAFICO]
        if desconocidos:
            raise SystemExit(f"Tipos de trafico desconocidos: {desconocidos}. Usa {list(COLORES_TRAFICO)}.")
        cuentas = [round(PORCENTAJES_TRAFICO[k] * n) for k in nombres[:-1]]
        cuentas.append(n - sum(cuentas))
        tipos = np.concatenate([[k] * c for k, c in zip(nombres, cuentas)])
        np.random.default_rng(SEMILLA_TRAFICO).shuffle(tipos)
        r["traffic_type"] = tipos
    else:
        r["traffic_type"] = "mixed"

    # Posiciones de los BBU pools: siempre sobre RRH reales. Se calculan solo con las reales,
    # para que no cambien al variar el numero de RRH sinteticas.
    pools, _, _ = ce.elegir_pools(P_real, N_POOLS, None, METODO_POOLS, SEMILLA)
    idx_pools = np.array(pools)
    Q = P[idx_pools]
    n_p = len(idx_pools)
    pool_ids = np.arange(1, n_p + 1)

    # Retardo y alcanzabilidad
    D = distancias(P, Q)                        # km
    RT = 2 * D * US_POR_KM                      # retardo ida y vuelta (us)
    d_max_km = RETARDO_RT_MAX_US / (2 * US_POR_KM)
    alc = RT <= RETARDO_RT_MAX_US               # RRH x pool
    n_opc = alc.sum(axis=1)
    servida = n_opc > 0
    opt = np.where(servida, np.where(alc, D, np.inf).argmin(axis=1), -1)
    r["servida"] = servida
    r["n_opciones"] = n_opc
    r["pool_optimo"] = np.where(servida, pool_ids[np.clip(opt, 0, None)], 0)
    r["dist_optimo_km"] = np.where(servida, D[np.arange(n), np.clip(opt, 0, None)], np.nan)
    r["retardo_rt_us"] = 2 * r.dist_optimo_km * US_POR_KM
    r["lista_pools"] = [";".join(map(str, pool_ids[alc[i]])) for i in range(n)]
    r["dist_al_mas_cercano_km"] = D.min(axis=1)

    print(f"RRHs: {n} | pools candidatos: {n_p} | retardo RT max: {RETARDO_RT_MAX_US:.0f} us "
          f"= {d_max_km:.1f} km (fibra {US_POR_KM} us/km)")
    print(f"RRHs servidas: {servida.sum()} | no servidas: {(~servida).sum()}")
    print(f"Opciones por RRH: min {n_opc.min()}, media {n_opc.mean():.1f}, max {n_opc.max()}")
    print(f"RRHs con una sola opcion: {(n_opc == 1).sum()}")

    # Clasificacion A/B/C por pool
    filas = []
    for j in range(n_p):
        a = (alc[:, j] & (n_opc == 1)).sum()
        b = (alc[:, j] & (opt == j) & (n_opc > 1)).sum()
        c = (alc[:, j] & (opt != j)).sum()
        filas.append({"pool_id": int(pool_ids[j]), "lat": r.lat[idx_pools[j]], "lon": r.lon[idx_pools[j]],
                          "rrh_id": int(r.rrh_id[idx_pools[j]]),
                          "solo_esta_opcion": int(a), "optimo_con_alternativas": int(b),
                          "alcanzable_no_optimo": int(c), "total_alcanzables": int(alc[:, j].sum()),
                          "rrh_optimo_total": int((opt == j).sum())})
    tabla_pools = pd.DataFrame(filas)
    tabla_pools.to_csv("fase1_pools.csv", index=False)
    print("\n", tabla_pools.to_string(index=False))

    # Numero minimo de pools por retardo (cobertura greedy) y barrido de retardo
    elegidos = cobertura_greedy(alc)
    print(f"\nPools minimos (greedy) para servir a todas las RRH alcanzables con "
          f"{RETARDO_RT_MAX_US:.0f} us: {len(elegidos)} -> pools {sorted(pool_ids[elegidos].tolist())}")
    barrido = []
    for t in RETARDOS_BARRIDO_US:
        alc_t = (2 * D * US_POR_KM) <= t
        barrido.append({"retardo_rt_us": t, "distancia_max_km": t / (2 * US_POR_KM),
                            "rrh_servidas": int(alc_t.any(axis=1).sum()),
                            "pct_servidas": 100 * alc_t.any(axis=1).mean(),
                            "pools_minimos": len(cobertura_greedy(alc_t))})
    barrido = pd.DataFrame(barrido)
    barrido.to_csv("fase1_barrido_retardo.csv", index=False)
    print("\n", barrido.to_string(index=False))

    r.to_csv("fase1_rrh.csv", index=False)

    # Que pools sobran: se retiran de uno en uno y se mira cuanto empeora la distancia / el servicio
    poda = pd.concat([poda_pools(D, t) for t in [RETARDO_RT_MAX_US, RETARDO_PODA_ESTRICTO_US]],
                     ignore_index=True)
    poda.to_csv("fase1_poda_pools.csv", index=False)
    for t, g in poda.groupby("retardo_rt_us", sort=False):
        completas = g[g.rrh_servidas == g.rrh_servidas.iloc[0]]
        print(f"\nRetardo {t:.0f} us: se pueden retirar {len(completas) - 1} de {n_p} pools sin perder "
              f"ninguna RRH servida (quedan {completas.pools_activos.min()}).")
        print("  Orden de retirada:", g.pool_retirado.iloc[1:len(completas)].astype(int).tolist())

    # Prueba de robustez: distinto numero de RRH sinteticas, con las posiciones de los pools fijas
    filas_rob = []
    for n_s in BARRIDO_SINTETICOS:
        _, P_b, _, _, _ = ce.construir_rrh(sitios, n_s, np.random.default_rng(SEMILLA), lon0, lat0)
        D_b = distancias(P_b, Q)
        alc_b = (2 * D_b * US_POR_KM) <= RETARDO_RT_MAX_US
        filas_rob.append({"rrh_sinteticas": n_s, "rrh_total": len(P_b),
                              "pct_servidas": 100 * alc_b.any(axis=1).mean(),
                              "dist_media_km": D_b.min(axis=1).mean(), "dist_max_km": D_b.min(axis=1).max(),
                              "opciones_medias": alc_b.sum(axis=1).mean(),
                              "rrh_una_opcion": int((alc_b.sum(axis=1) == 1).sum()),
                              "pools_minimos": len(cobertura_greedy(alc_b))})
    robustez = pd.DataFrame(filas_rob)
    robustez.to_csv("fase1_robustez_sinteticas.csv", index=False)
    print("\nRobustez (RRH sinteticas, pools fijos):\n", robustez.round(2).to_string(index=False))

    # ---------------- Figuras ----------------
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    cmap = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    col_pool = {int(p): cmap[i % len(cmap)] for i, p in enumerate(pool_ids)}

    # Fig 1: mapa de RRHs por trafico + pools candidatos
    fig, ax = plt.subplots(figsize=(8, 8))
    for t, c in COLORES_TRAFICO.items():
        m = (r.traffic_type == t).values
        reales = m & (r.tipo == "real").values
        otras = m & ~reales
        ax.scatter(r.lon[reales], r.lat[reales], s=6, c=c, label=t)
        ax.scatter(r.lon[otras], r.lat[otras], s=22, c=c, marker="s", edgecolors="k", linewidths=0.6)
    ax.scatter(r.lon[idx_pools], r.lat[idx_pools], s=170, c="red", marker="*", edgecolors="k", zorder=5,
               label="BBU pool candidato")
    for j, i in enumerate(idx_pools):
        ax.annotate(str(pool_ids[j]), (r.lon[i], r.lat[i]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, weight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(lat0))); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("RRHs by traffic pattern and candidate BBU pools")
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig1_mapa_trafico_pools.png", dpi=200); plt.close(fig)

    # Fig 2: mapa de asignacion al pool optimo (no servidas en negro)
    fig, ax = plt.subplots(figsize=(8, 8))
    for j in range(n_p):
        m = (r.pool_optimo == pool_ids[j]).values
        ax.scatter(r.lon[m], r.lat[m], s=7, color=col_pool[int(pool_ids[j])])
    if (~servida).any():
        ax.scatter(r.lon[~servida], r.lat[~servida], s=25, c="k", marker="x", label="Unserved")
        ax.legend(loc="upper left", fontsize=8)
    ax.scatter(r.lon[idx_pools], r.lat[idx_pools], s=170, c="red", marker="*", edgecolors="k", zorder=5)
    for j, i in enumerate(idx_pools):
        ax.annotate(str(pool_ids[j]), (r.lon[i], r.lat[i]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, weight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set_title("Assignment of each RRH to its closest BBU pool")
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig2_mapa_asignacion.png", dpi=200); plt.close(fig)

    # Fig 7: lineas entre cada RRH y su pool optimo
    fig, ax = plt.subplots(figsize=(8, 8))
    idx_serv = np.flatnonzero(servida)
    segs = [[(r.lon[i], r.lat[i]), (r.lon[idx_pools[opt[i]]], r.lat[idx_pools[opt[i]]])] for i in idx_serv]
    cols = [col_pool[int(pool_ids[opt[i]])] for i in idx_serv]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.4, alpha=0.6))
    ax.scatter(r.lon, r.lat, s=3, c="k", alpha=0.6)
    if (~servida).any():
        ax.scatter(r.lon[~servida], r.lat[~servida], s=25, c="r", marker="x", label="Unserved")
        ax.legend(loc="upper left", fontsize=8)
    ax.scatter(r.lon[idx_pools], r.lat[idx_pools], s=170, c="red", marker="*", edgecolors="k", zorder=5)
    for j, i in enumerate(idx_pools):
        ax.annotate(str(pool_ids[j]), (r.lon[i], r.lat[i]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, weight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(lat0))); ax.autoscale()
    ax.set_title("Connections between each RRH and its closest BBU pool")
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig7_mapa_lineas.png", dpi=200); plt.close(fig)

    # Fig 3: CDF de la distancia al pool mas cercano
    d = np.sort(D.min(axis=1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d, np.arange(1, n + 1) / n, lw=2)
    x_max = max(d.max() * 1.05, 1)
    if d_max_km <= x_max:
        ax.axvline(d_max_km, color="r", ls="--", label=f"Max. allowed distance ({d_max_km:.0f} km)")
    else:   # el limite queda fuera del eje: se indica en la leyenda
        ax.plot([], [], " ", label=f"Max. allowed distance: {d_max_km:.0f} km (off scale)")
    ax.set_xlabel("Distance RRH - closest BBU pool (km)"); ax.set_ylabel("CDF")
    ax.set_xlim(0, x_max); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig3_cdf_distancia.png", dpi=200); plt.close(fig)

    # Fig 4: tres categorias por pool
    x = np.arange(n_p); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - w, tabla_pools.solo_esta_opcion, w, label="Only option")
    ax.bar(x, tabla_pools.optimo_con_alternativas, w, label="Optimal, other options exist")
    ax.bar(x + w, tabla_pools.alcanzable_no_optimo, w, label="Reachable, not optimal")
    ax.set_xticks(x); ax.set_xticklabels(tabla_pools.pool_id)
    ax.set_xlabel("BBU pool ID"); ax.set_ylabel("Number of RRHs"); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig4_categorias_por_pool.png", dpi=200); plt.close(fig)

    # Fig 5: numero de pools alcanzables por RRH
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(np.arange(n), n_opc, width=1.0, color="#1f77b4")
    ax.set_xlabel("RRH ID"); ax.set_ylabel("Number of reachable BBU pools"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig5_opciones_por_rrh.png", dpi=200); plt.close(fig)

    # Fig 8: que pasa al ir retirando pools
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (t, g) in zip(axes, poda.groupby("retardo_rt_us", sort=False)):
        ax.plot(g.pools_activos, g.dist_media_km, "o-", color="#1f77b4", label="Mean distance")
        ax.plot(g.pools_activos, g.dist_max_km, "s-", color="#ff7f0e", label="Max distance")
        ax.set_xlabel("Number of operative BBU pools"); ax.set_ylabel("Distance to closest pool (km)")
        ax.invert_xaxis(); ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for _, fila in g.iloc[1:].iterrows():   # etiqueta: ID del pool retirado en ese paso
            ax.annotate(f"-{int(fila.pool_retirado)}", (fila.pools_activos, fila.dist_media_km),
                        xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7, color="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(g.pools_activos, g.pct_servidas, "^--", color="#d62728", label="RRHs served (%)")
        ax2.set_ylim(0, 102); ax2.set_ylabel("RRHs served (%)", color="#d62728")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)
        ax.set_title(f"Removing pools one by one (max RTT {t:.0f} us)")
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig8_poda_pools.png", dpi=200); plt.close(fig)

    # Fig 6: pools minimos y % de RRH servidas segun el retardo maximo
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(barrido.retardo_rt_us, barrido.pools_minimos, "o-", color="#1f77b4")
    ax.set_xlabel("Maximum round-trip delay (us)"); ax.set_ylabel("Minimum number of pools", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(barrido.retardo_rt_us, barrido.pct_servidas, "s--", color="#d62728")
    ax2.set_ylabel("RRHs served (%)", color="#d62728"); ax2.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig6_barrido_retardo.png", dpi=200); plt.close(fig)

    # ---------------- Mapa Folium ----------------
    import folium
    m = folium.Map(location=[lat0, lon0], zoom_start=10, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri").add_to(m)

    def hexcol(c):
        return "#{:02x}{:02x}{:02x}".format(*tuple(int(255 * v) for v in c))

    for i in range(n):
        if servida[i]:
            color = hexcol(col_pool[int(r.pool_optimo[i])])
            txt = (f"RRH {r.rrh_id[i]} ({r.tipo[i]}, {r.traffic_type[i]})<br>pool optimo: {r.pool_optimo[i]}"
                   f"<br>distancia: {r.dist_optimo_km[i]:.2f} km<br>opciones: {n_opc[i]}")
        else:
            color = "#000000"
            txt = f"RRH {r.rrh_id[i]}: no servida"
        folium.CircleMarker([r.lat[i], r.lon[i]], radius=3, color=color, fill=True, fill_opacity=0.8,
                            weight=1, popup=folium.Popup(txt, max_width=250)).add_to(m)
    for j, i in enumerate(idx_pools):
        folium.Marker([r.lat[i], r.lon[i]], popup=f"BBU pool {pool_ids[j]}",
                      icon=folium.Icon(color="red", icon="star")).add_to(m)
    lineas = folium.FeatureGroup(name="Conexiones RRH - pool", show=False)
    for i in range(n):
        if servida[i]:
            j = int(opt[i])
            folium.PolyLine([[r.lat[i], r.lon[i]], [r.lat[idx_pools[j]], r.lon[idx_pools[j]]]],
                            color=hexcol(col_pool[int(pool_ids[j])]), weight=1, opacity=0.5).add_to(lineas)
    lineas.add_to(m)
    folium.LayerControl().add_to(m)
    m.save("mapa_fase1.html")

    # ---------------- Configuracion reducida (pools desactivados por el usuario) ----------------
    if POOLS_DESACTIVADOS:
        quitar = sorted({int(p) for p in POOLS_DESACTIVADOS})
        act = ~np.isin(pool_ids, quitar)
        if not act.any():
            raise SystemExit("POOLS_DESACTIVADOS incluye todos los pools.")
        D_a, ids_a, idx_a = D[:, act], pool_ids[act], idx_pools[act]
        alc_a = (2 * D_a * US_POR_KM) <= RETARDO_RT_MAX_US
        serv_a = alc_a.any(axis=1)
        opt_a = np.where(serv_a, np.where(alc_a, D_a, np.inf).argmin(axis=1), -1)
        nuevo = np.where(serv_a, ids_a[np.clip(opt_a, 0, None)], 0)
        cambia = nuevo != r.pool_optimo.values
        dist_nueva = np.where(serv_a, D_a[np.arange(n), np.clip(opt_a, 0, None)], np.nan)
        red = pd.DataFrame({"rrh_id": r.rrh_id, "lon": r.lon, "lat": r.lat, "tipo": r.tipo,
                            "traffic_type": r.traffic_type, "pool_anterior": r.pool_optimo,
                            "pool_nuevo": nuevo, "cambia_de_pool": cambia, "servida": serv_a,
                            "dist_anterior_km": r.dist_optimo_km, "dist_nueva_km": dist_nueva,
                            "n_opciones": alc_a.sum(axis=1)})
        red.to_csv("fase1_rrh_reducida.csv", index=False)
        filas_red = []
        for k, pid in enumerate(ids_a):
            m_nuevo = nuevo == pid
            filas_red.append({"pool_id": int(pid), "rrh_antes": int((r.pool_optimo == pid).sum()),
                                  "rrh_despues": int(m_nuevo.sum()), "rrh_recibidas": int((m_nuevo & cambia).sum()),
                                  "dist_media_km": dist_nueva[m_nuevo].mean() if m_nuevo.any() else np.nan,
                                  "dist_max_km": dist_nueva[m_nuevo].max() if m_nuevo.any() else np.nan})
        tabla_red = pd.DataFrame(filas_red)
        tabla_red.to_csv("fase1_pools_reducida.csv", index=False)
        print(f"\nConfiguracion reducida: pools desactivados {quitar} -> quedan {int(act.sum())} pools")
        print(f"RRHs servidas: {int(serv_a.sum())} de {n} | RRHs que cambian de pool: {int(cambia.sum())}")
        print(f"Distancia media: {np.nanmean(r.dist_optimo_km):.2f} -> {np.nanmean(dist_nueva):.2f} km | "
              f"maxima: {np.nanmax(r.dist_optimo_km):.2f} -> {np.nanmax(dist_nueva):.2f} km")
        print(tabla_red.round(2).to_string(index=False))

        fig, ax = plt.subplots(figsize=(8, 8))
        for pid in ids_a:
            m = nuevo == pid
            ax.scatter(r.lon[m & ~cambia], r.lat[m & ~cambia], s=7, color=col_pool[int(pid)])
            ax.scatter(r.lon[m & cambia], r.lat[m & cambia], s=16, color=col_pool[int(pid)],
                       edgecolors="k", linewidths=0.6)
        ax.scatter([], [], s=16, c="w", edgecolors="k", label="RRH that changed pool")
        if (~serv_a).any():
            ax.scatter(r.lon[~serv_a], r.lat[~serv_a], s=25, c="k", marker="x", label="Unserved")
        ax.scatter(r.lon[idx_a], r.lat[idx_a], s=170, c="red", marker="*", edgecolors="k", zorder=5,
                   label="Operative BBU pool")
        for pid, i in zip(ids_a, idx_a):
            ax.annotate(str(pid), (r.lon[i], r.lat[i]), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, weight="bold")
        idx_off = idx_pools[~act]
        ax.scatter(r.lon[idx_off], r.lat[idx_off], s=110, c="lightgrey", marker="X", edgecolors="k",
                   zorder=4, label="Deactivated pool")
        for pid, i in zip(pool_ids[~act], idx_off):
            ax.annotate(str(pid), (r.lon[i], r.lat[i]), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color="grey")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_aspect(1 / np.cos(np.radians(lat0))); ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Assignment with {int(act.sum())} operative BBU pools (deactivated: {quitar})")
        fig.tight_layout(); fig.savefig(DIR_FIGURAS / "fase1_fig9_mapa_reducido.png", dpi=200); plt.close(fig)

        m2 = folium.Map(location=[lat0, lon0], zoom_start=10, tiles=None)
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri", name="Esri").add_to(m2)
        for i in range(n):
            if serv_a[i]:
                color = hexcol(col_pool[int(nuevo[i])])
                borde = "#000000" if cambia[i] else color
                txt = (f"RRH {r.rrh_id[i]} ({r.tipo[i]}, {r.traffic_type[i]})<br>pool: {r.pool_optimo[i]} -> "
                       f"{nuevo[i]}<br>distancia: {dist_nueva[i]:.2f} km")
            else:
                color = borde = "#000000"
                txt = f"RRH {r.rrh_id[i]}: no servida"
            folium.CircleMarker([r.lat[i], r.lon[i]], radius=4 if cambia[i] else 3, color=borde, fill=True,
                                fill_color=color, fill_opacity=0.85, weight=2 if cambia[i] else 1,
                                popup=folium.Popup(txt, max_width=250)).add_to(m2)
        for pid, i in zip(ids_a, idx_a):
            folium.Marker([r.lat[i], r.lon[i]], popup=f"BBU pool {pid}",
                          icon=folium.Icon(color="red", icon="star")).add_to(m2)
        for pid, i in zip(pool_ids[~act], idx_off):
            folium.Marker([r.lat[i], r.lon[i]], popup=f"BBU pool {pid} (desactivado)",
                          icon=folium.Icon(color="gray", icon="remove")).add_to(m2)
        m2.save("mapa_fase1_reducido.html")
    print("\nGuardado: fase1_rrh.csv, fase1_pools.csv, fase1_barrido_retardo.csv, fase1_robustez_sinteticas.csv, fase1_poda_pools.csv, mapa_fase1.html")
    print(f"Figuras guardadas en: {DIR_FIGURAS}")


if __name__ == "__main__":
    main()