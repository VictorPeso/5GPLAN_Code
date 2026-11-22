"""
fase2_load_balancing.py
Fase 2 del proyecto C-RAN: balanceo de carga entre BBU pools ("load balancing").

Parte del mismo escenario que la fase 1 (RRH de Telefonica + sinteticas, pools sobre RRH reales)
y de las listas de pools posibles que salen del retardo maximo (f2_retardo_rt_max_us).

Modelo de carga: cada RRH tiene un patron de trafico (residential / office / mixed) y su carga
en MBph a cada hora sale del perfil medio de 24 h por celda de la tesis (Fig. D.1, DL LTE).

Asignaciones que se comparan (todas respetan el retardo maximo, sin limite de capacidad):
  1) "Distance only":  cada RRH va a su pool mas cercano (salida de la fase 1).
  2) "LB traffic":     balanceo por carga (MBph) a la hora elegida.
       - 1a iteracion: las RRH con una sola opcion se asignan a su pool.
       - 2a iteracion: el resto, ordenadas por numero de opciones (menos opciones primero), a igual
         numero por mayor carga y despues por menor distancia; cada una va al pool de su lista con
         menos carga en ese momento (a igualdad, el mas cercano); se actualiza la carga y se sigue.
  3) "LB RRH count":   igual pero balanceando el numero de RRH por pool (no usa la carga).

Salidas: figuras en Figuras/fase2_fig*.png y CSV fase2_*.csv.
Todos los parametros estan en config.py (claves f2_*) y se pueden cambiar con --nombre=valor.
Dependencias: numpy, pandas, matplotlib (y las de fase1_min_delay.py)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

import cran_escenario as ce
import fase1_min_delay as f1

cfg = f1.cfg
US_POR_KM = f1.US_POR_KM
DIR_FIGURAS = f1.DIR_FIGURAS
RT_MAX = float(cfg.f2_retardo_rt_max_us)
CASOS = dict(cfg.f2_pools)                      # {"few": 10, "normal": 14, "many": 19}
CASO_PRINCIPAL = cfg.f2_caso_principal
HORAS_FIGURA = list(cfg.f2_horas_figura)
CAP_TARJETA = float(cfg.f2_capacidad_tarjeta_mbph)

ALG_DELAY, ALG_CARGA, ALG_RRH = "Distance only", "Load balancing (traffic)", "Load balancing (no. of RRHs)"
COL_ALG = {ALG_DELAY: "#7f7f7f", ALG_CARGA: "#d62728", ALG_RRH: "#9467bd"}
COL_TIPO = {"residential": "#2ca02c", "office": "#1f77b4", "real": "#8c564b"}
NOM_TIPO = {"residential": "All residential", "office": "All office", "real": "Real case"}

# Perfil medio de trafico DL LTE por celda (MBph) a las horas 1..24. Lectura aproximada (digitalizada)
# de la Fig. D.1 del Anexo D de la tesis. Se puede sustituir por datos propios con f2_perfiles_csv.
PERFILES_TESIS = {
    "residential": [420, 300, 200, 125, 100, 90, 105, 185, 230, 280, 325, 345, 365, 385, 385, 392,
                    420, 458, 453, 462, 475, 500, 510, 480],
    "office":      [255, 180, 160, 118, 100, 85, 92, 175, 300, 400, 448, 446, 440, 470, 485, 468,
                    445, 405, 360, 325, 318, 310, 307, 275],
    "mixed":       [450, 375, 275, 200, 143, 133, 138, 245, 355, 415, 465, 497, 503, 520, 540, 526,
                    525, 528, 508, 508, 515, 527, 528, 508],
}


# ---------------- Perfiles de trafico y carga ----------------
def cargar_perfiles():
    """Devuelve {tipo: array de 24 valores (MBph) para las horas 1..24}."""
    if cfg.f2_perfiles_csv:
        t = pd.read_csv(cfg.f2_perfiles_csv).sort_values("hour")
        if len(t) != 24:
            raise SystemExit("f2_perfiles_csv debe tener 24 filas (hour = 1..24).")
        return {k: t[k].to_numpy(float) for k in ["residential", "office", "mixed"]}
    return {k: np.array(v, float) for k, v in PERFILES_TESIS.items()}


def carga_rrh(tipos, hora, perfiles, factor):
    """Carga (MBph) de cada RRH a una hora 0..23 (la hora 0 equivale a la 24 del perfil)."""
    idx = (hora - 1) % 24
    base = {k: perfiles[k][idx] for k in perfiles}
    return np.array([base[t] for t in tipos]) * factor


def hora_txt(h):
    return f"{h:02d}:00"


# ---------------- Escenario ----------------
def construir_escenario():
    """Mismo escenario que la fase 1 (RRH, tipos de trafico y pools candidatos)."""
    rng = np.random.default_rng(f1.SEMILLA)
    sitios = ce.cargar_sitios()
    rrh, P, P_real, _, _ = ce.construir_rrh(sitios, f1.N_SINTETICOS, rng)
    r = rrh[["rrh_id", "lon", "lat", "n_estaciones", "operadores", "tipo"]].copy()
    n = len(r)
    if f1.PORCENTAJES_TRAFICO:
        nombres = list(f1.PORCENTAJES_TRAFICO)
        cuentas = [round(f1.PORCENTAJES_TRAFICO[k] * n) for k in nombres[:-1]]
        cuentas.append(n - sum(cuentas))
        tipos = np.concatenate([[k] * c for k, c in zip(nombres, cuentas)])
        np.random.default_rng(f1.SEMILLA_TRAFICO).shuffle(tipos)
        r["traffic_type"] = tipos
    else:
        r["traffic_type"] = "mixed"
    pools19, _, _ = ce.elegir_pools(P_real, f1.N_POOLS, None, f1.METODO_POOLS, f1.SEMILLA)
    return r, P, P_real, np.array(pools19)


def conjunto_pools(k, P, P_real, idx_cand, D_cand):
    """Indices (en P) e identificadores de los k pools de un caso.

    Si k <= pools candidatos se parte de los candidatos y se retiran pools de uno en uno
    (misma poda que la fase 1, con el retardo f2_retardo_rt_max_us). Si k es mayor, se calculan
    k posiciones nuevas con el mismo metodo de la fase 1.
    """
    if k <= len(idx_cand):
        tabla = f1.poda_pools(D_cand, RT_MAX)
        retirados = set(tabla[tabla.pools_activos >= k].pool_retirado) - {0}
        act = [j for j in range(len(idx_cand)) if (j + 1) not in retirados]
        return idx_cand[act], np.array(act) + 1
    nuevos, _, _ = ce.elegir_pools(P_real, k, None, f1.METODO_POOLS, f1.SEMILLA)
    return np.array(nuevos), np.arange(1, k + 1)


# ---------------- Algoritmos de asignacion ----------------
def asignar_distancia(D, alc):
    """Cada RRH al pool alcanzable mas cercano (-1 si no llega a ninguno)."""
    asig = np.where(alc, D, np.inf).argmin(axis=1)
    return np.where(alc.any(axis=1), asig, -1)


def asignar_balanceo(D, alc, carga, modo):
    """Balanceo de carga (modo="carga") o del numero de RRH (modo="rrh"). Devuelve pool por RRH."""
    n, k = alc.shape
    n_opc = alc.sum(axis=1)
    asig = -np.ones(n, dtype=int)
    carga_pool = np.zeros(k)
    cuenta = np.zeros(k)
    # 1a iteracion: RRH con una sola opcion
    for i in np.where(n_opc == 1)[0]:
        j = int(alc[i].argmax())
        asig[i] = j
        carga_pool[j] += carga[i]
        cuenta[j] += 1
    # 2a iteracion: el resto, de menos a mas opciones; a igualdad, mas carga y mas cerca primero
    resto = np.where(n_opc > 1)[0]
    dmin = np.where(alc, D, np.inf).min(axis=1)
    if modo == "carga":
        orden = np.lexsort((dmin[resto], -carga[resto], n_opc[resto]))
    else:
        orden = np.lexsort((dmin[resto], n_opc[resto]))
    for i in resto[orden]:
        cand = np.where(alc[i])[0]
        metrica = carga_pool[cand] if modo == "carga" else cuenta[cand]
        j = int(cand[np.lexsort((D[i, cand], metrica))[0]])
        asig[i] = j
        carga_pool[j] += carga[i]
        cuenta[j] += 1
    return asig


# ---------------- Metricas ----------------
def carga_por_pool(asig, carga, k):
    ok = asig >= 0
    return np.bincount(asig[ok], weights=carga[ok], minlength=k)


def rrh_por_pool(asig, k):
    return np.bincount(asig[asig >= 0], minlength=k)


def metricas(asig, carga, D, k):
    lp = carga_por_pool(asig, carga, k)
    cnt = rrh_por_pool(asig, k)
    ok = asig >= 0
    dist = D[np.arange(len(asig))[ok], asig[ok]]
    return dict(rrh_servidas=int(ok.sum()), carga_media_mbph=lp.mean(), carga_std_mbph=lp.std(),
                cv_carga=lp.std() / lp.mean(), carga_max_mbph=lp.max(), carga_min_mbph=lp.min(),
                max_sobre_media=lp.max() / lp.mean(),
                tarjetas_max=int(np.ceil(lp.max() / CAP_TARJETA)), tarjetas_total=int(np.ceil(lp / CAP_TARJETA).sum()),
                rrh_por_pool_min=int(cnt.min()), rrh_por_pool_max=int(cnt.max()), rrh_por_pool_std=cnt.std(),
                dist_media_km=dist.mean(), dist_max_km=dist.max(), retardo_medio_us=2 * US_POR_KM * dist.mean())


# ---------------- Calculo de un caso (numero de pools) ----------------
def ejecutar_caso(k, esc, perfiles, factor, hora):
    r, P, P_real, idx_cand, D_cand = esc
    idx, ids = conjunto_pools(k, P, P_real, idx_cand, D_cand)
    D = f1.distancias(P, P[idx])
    alc = 2 * D * US_POR_KM <= RT_MAX
    tipos = r.traffic_type.to_numpy()
    res = dict(k=k, idx=idx, ids=ids, D=D, alc=alc, servida=alc.any(axis=1))
    res["asig_dist"] = asignar_distancia(D, alc)
    res["carga"] = carga_rrh(tipos, hora, perfiles, factor)
    res["asig_carga"] = asignar_balanceo(D, alc, res["carga"], "carga")
    res["asig_rrh"] = asignar_balanceo(D, alc, res["carga"], "rrh")
    res["asig"] = {ALG_DELAY: res["asig_dist"], ALG_CARGA: res["asig_carga"], ALG_RRH: res["asig_rrh"]}
    # Variantes: todas residenciales / todas oficina (el balanceo se repite con esas cargas)
    res["tipos_alt"] = {}
    for t in ["residential", "office"]:
        c = carga_rrh(np.array([t] * len(r)), hora, perfiles, factor)
        res["tipos_alt"][t] = dict(carga=c, asig_carga=asignar_balanceo(D, alc, c, "carga"))
    return res


# ---------------- Figuras ----------------
def guardar(fig, nombre):
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / nombre, dpi=200)
    plt.close(fig)


def barras(ax, ids, series, colores, ancho=0.8):
    """Barras agrupadas por pool. series = {etiqueta: valores}."""
    n = len(series)
    x = np.arange(len(ids))
    w = ancho / n
    for i, (nombre, v) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, v, w, label=nombre, color=colores[nombre], edgecolor="white", linewidth=.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids])
    ax.set_xlabel("BBU pool identifier")
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def fig_perfiles(perfiles):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    horas = np.arange(1, 25)
    for t, c in [("residential", COL_TIPO["residential"]), ("office", COL_TIPO["office"]), ("mixed", "#ff7f0e")]:
        ax.plot(horas, perfiles[t], color=c, lw=2, label=t.capitalize())
    for h in HORAS_FIGURA:
        ax.axvline(h, color="k", ls=":", lw=.8)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Average DL traffic per cell (MBph)")
    ax.set_xlim(1, 24)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.grid(alpha=.3)
    ax.legend()
    guardar(fig, "fase2_fig0_perfiles_trafico.png")


def fig_1a(res, perfiles, factor, tipos, caso):
    for h in HORAS_FIGURA:
        c = carga_rrh(tipos, h, perfiles, factor)
        lp = carga_por_pool(res["asig_dist"], c, len(res["ids"]))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        barras(ax, res["ids"], {"Distance only": lp}, COL_ALG)
        ax.axhline(lp.mean(), color="k", ls="--", lw=1, label=f"Mean {lp.mean():,.0f} MBph")
        ax.set_ylabel("Total load per pool (MBph)")
        ax.set_title(f"Load per BBU pool at {hora_txt(h)} (delay-only assignment, {len(res['ids'])} pools)")
        ax.legend()
        guardar(fig, f"fase2_fig1a_carga_pools_{h:02d}h.png")


def fig_1b(res, perfiles, factor, hora, n_rrh):
    k = len(res["ids"])
    series = {}
    for t in ["residential", "office"]:
        c = carga_rrh(np.array([t] * n_rrh), hora, perfiles, factor)
        series[NOM_TIPO[t]] = carga_por_pool(res["asig_dist"], c, k)
    series[NOM_TIPO["real"]] = carga_por_pool(res["asig_dist"], res["carga"], k)
    cols = {NOM_TIPO[t]: COL_TIPO[t] for t in COL_TIPO}
    fig, ax = plt.subplots(figsize=(11, 4.8))
    barras(ax, res["ids"], series, cols)
    ax.set_ylabel("Total load per pool (MBph)")
    ax.set_title(f"Load per pool at {hora_txt(hora)}: all residential, all office and real case (delay-only assignment)")
    ax.legend()
    guardar(fig, "fase2_fig1b_residencial_oficina_real.png")


def fig_1c(tipos, perfiles, factor):
    horas = np.arange(0, 24)
    n = len(tipos)
    tot = {"real": [carga_rrh(tipos, h, perfiles, factor).sum() / 1000 for h in horas]}
    for t in ["residential", "office"]:
        tot[t] = [carga_rrh(np.array([t] * n), h, perfiles, factor).sum() / 1000 for h in horas]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for t, ls, lw in [("real", "-", 2.5), ("residential", "--", 1.3), ("office", "--", 1.3)]:
        ax.plot(horas, tot[t], ls=ls, lw=lw, color=COL_TIPO[t], label=NOM_TIPO[t])
    hp = int(np.argmax(tot["real"]))
    ax.plot(hp, tot["real"][hp], "o", color="k", zorder=5)
    ax.annotate(f"Peak {hora_txt(hp)}: {tot['real'][hp]:.0f} GBph", (hp, tot["real"][hp]),
                textcoords="offset points", xytext=(-95, 12), fontsize=9)
    for h in HORAS_FIGURA:
        ax.axvline(h, color="k", ls=":", lw=.8)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Overall network load (GBph)")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1.15 * max(tot["real"]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.grid(alpha=.3)
    ax.legend(loc="lower right")
    guardar(fig, "fase2_fig1c_carga_red_dia.png")


def fig_2a(res):
    k = len(res["ids"])
    cnt = {a: rrh_por_pool(res["asig"][a], k) for a in COL_ALG}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, alg, tit in [(axs[0], ALG_CARGA, "(i) Load balancing according to traffic"),
                         (axs[1], ALG_RRH, "(ii) Load balancing according to number of RRHs")]:
        barras(ax, res["ids"], {ALG_DELAY: cnt[ALG_DELAY], alg: cnt[alg]}, COL_ALG)
        ax.set_title(tit)
        ax.legend()
    axs[0].set_ylabel("Number of RRHs per pool")
    guardar(fig, "fase2_fig2a_rrh_por_pool.png")


def fig_2b(res, perfiles, factor, tipos, hora, k):
    horas = np.arange(0, 24)
    cambios, cv = [], {a: [] for a in COL_ALG}
    asig_pico = res["asig_carga"]
    for h in horas:
        c = carga_rrh(tipos, h, perfiles, factor)
        a_h = asignar_balanceo(res["D"], res["alc"], c, "carga")
        ok = (a_h >= 0) & (asig_pico >= 0)
        cambios.append(int((a_h[ok] != asig_pico[ok]).sum()))
        for alg in COL_ALG:                      # asignacion fija (la de la hora pico) evaluada cada hora
            lp = carga_por_pool(res["asig"][alg], c, k)
            cv[alg].append(lp.std() / lp.mean())
    tabla = pd.DataFrame({"hora": horas, "rrh_que_cambian_vs_hora_pico": cambios,
                          **{f"cv_carga_{a}": cv[a] for a in COL_ALG}})
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    cols = ["#d62728" if h == hora else ("#ff9896" if h in HORAS_FIGURA else "#aaaaaa") for h in horas]
    axs[0].bar(horas, cambios, color=cols)
    axs[0].set_xlabel("Hour of the day")
    axs[0].set_ylabel(f"RRHs changing pool vs. the {hora_txt(hora)} assignment")
    axs[0].set_title("Re-running the traffic balancing at each hour")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    axs[0].grid(axis="y", alpha=.3)
    for a in COL_ALG:
        axs[1].plot(horas, cv[a], lw=2, color=COL_ALG[a], label=a)
    axs[1].set_xlabel("Hour of the day")
    axs[1].set_ylabel("Load imbalance between pools (std / mean)")
    axs[1].set_title(f"Assignment fixed at {hora_txt(hora)}, load evaluated every hour")
    axs[1].set_ylim(bottom=0)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    axs[1].grid(alpha=.3)
    axs[1].legend()
    guardar(fig, "fase2_fig2b_cambios_por_hora.png")
    return tabla


def fig_2c(res):
    k = len(res["ids"])
    cols = {NOM_TIPO[t]: COL_TIPO[t] for t in COL_TIPO}
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, (tit, usar_lb) in zip(axs, [("Delay-only assignment", False), ("Load balancing according to traffic", True)]):
        s = {}
        for t in ["residential", "office"]:
            c = res["tipos_alt"][t]["carga"]
            a = res["tipos_alt"][t]["asig_carga"] if usar_lb else res["asig_dist"]
            s[NOM_TIPO[t]] = carga_por_pool(a, c, k)
        s[NOM_TIPO["real"]] = carga_por_pool(res["asig_carga"] if usar_lb else res["asig_dist"], res["carga"], k)
        barras(ax, res["ids"], s, cols)
        ax.set_title(tit)
        ax.legend()
    axs[0].set_ylabel("Total load per pool (MBph)")
    guardar(fig, "fase2_fig2c_balanceo_residencial_oficina.png")


def fig_2d(res):
    D = res["D"]
    ok = res["servida"]
    dist = {a: D[np.arange(len(D))[ok], res["asig"][a][ok]] for a in COL_ALG}
    dmax = RT_MAX / (2 * US_POR_KM)
    bordes = np.arange(0, np.ceil(dmax) + 1)
    x = np.arange(len(bordes) - 1)
    w = 0.8 / 3
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, a in enumerate(COL_ALG):
        h, _ = np.histogram(dist[a], bins=bordes)
        ax.bar(x + (i - 1) * w, 100 * h / ok.sum(), w, label=f"{a} (mean {dist[a].mean():.2f} km)",
               color=COL_ALG[a], edgecolor="white", linewidth=.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(bordes[i])}-{int(bordes[i + 1])}" for i in x])
    ax.set_xlabel("Distance RRH - assigned BBU pool (km)")
    ax.set_ylabel("Share of RRHs (%)")
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    ax.legend()
    guardar(fig, "fase2_fig2d_distancias.png")


def fig_2e(res):
    k = len(res["ids"])
    lp = {a: carga_por_pool(res["asig"][a], res["carga"], k) for a in COL_ALG}
    fig, ax = plt.subplots(figsize=(11, 4.8))
    barras(ax, res["ids"], {f"{a} (std {lp[a].std():,.0f} MBph)": lp[a] for a in COL_ALG},
           {f"{a} (std {lp[a].std():,.0f} MBph)": COL_ALG[a] for a in COL_ALG})
    ax.axhline(lp[ALG_DELAY].mean(), color="k", ls="--", lw=1, label=f"Mean {lp[ALG_DELAY].mean():,.0f} MBph")
    ax.set_ylabel("Average capacity per pool (MBph)")
    ax.legend()
    guardar(fig, "fase2_fig2e_capacidad_media_por_pool.png")


def fig_2f(res, r):
    k = len(res["ids"])
    cmap = plt.get_cmap("tab20", 20)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    P = f1.a_km(r.lon, r.lat, 2.1542, 41.3926)
    for ax, alg in zip(axs, [ALG_DELAY, ALG_CARGA]):
        a = res["asig"][alg]
        ok = a >= 0
        ax.scatter(P[ok, 0], P[ok, 1], c=[cmap(j % 20) for j in a[ok]], s=8)
        pp = P[res["idx"]]
        ax.scatter(pp[:, 0], pp[:, 1], c="k", marker="s", s=60, edgecolor="w", zorder=5)
        for j, (x, y) in enumerate(pp):
            ax.annotate(str(res["ids"][j]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, weight="bold")
        ax.set_title(alg)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
    axs[0].set_ylabel("y (km)")
    guardar(fig, "fase2_fig2f_mapa_asignacion.png")


def fig_3(resultados, perfiles, factor, hora):
    nombres = list(resultados)
    filas = []
    for caso in nombres:
        res = resultados[caso]
        k = len(res["ids"])
        for alg in COL_ALG:
            m = metricas(res["asig"][alg], res["carga"], res["D"], k)
            filas.append(dict(caso=caso, pools=k, algoritmo=alg, hora=hora, **m))
    T = pd.DataFrame(filas)
    # (a) comparacion de metricas
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, tit in [(axs[0], "cv_carga", "Load imbalance (std / mean)"),
                         (axs[1], "max_sobre_media", "Most loaded pool / mean load"),
                         (axs[2], "dist_media_km", "Mean distance RRH - pool (km)")]:
        x = np.arange(len(nombres))
        w = 0.8 / 3
        for i, alg in enumerate(COL_ALG):
            v = [T[(T.caso == c) & (T.algoritmo == alg)][col].iloc[0] for c in nombres]
            ax.bar(x + (i - 1) * w, v, w, label=alg, color=COL_ALG[alg], edgecolor="white", linewidth=.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n({resultados[c]['k']} pools)" for c in nombres])
        ax.set_title(tit)
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)
    axs[0].legend(fontsize=8)
    guardar(fig, "fase2_fig3a_comparacion_n_pools.png")
    # (b) carga por pool en cada caso
    fig, axs = plt.subplots(len(nombres), 1, figsize=(11, 3.6 * len(nombres)), sharey=True)
    for ax, caso in zip(np.atleast_1d(axs), nombres):
        res = resultados[caso]
        k = len(res["ids"])
        barras(ax, res["ids"], {a: carga_por_pool(res["asig"][a], res["carga"], k) for a in COL_ALG}, COL_ALG)
        ax.set_title(f"{caso}: {k} pools")
        ax.set_ylabel("Load per pool (MBph)")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    guardar(fig, "fase2_fig3b_carga_por_pool_casos.png")
    return T


def fig_4(res, perfiles, factor, hora):
    """Cuanto mejora el balanceo si se permite mas retardo (mismos pools, mismas cargas)."""
    D, k = res["D"], len(res["ids"])
    filas = []
    for t in cfg.f2_barrido_retardos_us:
        alc = 2 * D * US_POR_KM <= t
        serv = alc.any(axis=1)
        for alg, a in [(ALG_DELAY, asignar_distancia(D, alc)), (ALG_CARGA, asignar_balanceo(D, alc, res["carga"], "carga")),
                       (ALG_RRH, asignar_balanceo(D, alc, res["carga"], "rrh"))]:
            m = metricas(a, res["carga"], D, k)
            filas.append(dict(retardo_rt_us=t, algoritmo=alg, pct_servidas=100 * serv.mean(),
                              opciones_medias=alc.sum(axis=1).mean(), **m))
    T = pd.DataFrame(filas)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for alg in COL_ALG:
        g = T[T.algoritmo == alg]
        axs[0].plot(g.retardo_rt_us, g.cv_carga, "o-", lw=2, color=COL_ALG[alg], label=alg)
        axs[1].plot(g.retardo_rt_us, g.dist_media_km, "o-", lw=2, color=COL_ALG[alg], label=alg)
    axs[0].set_ylabel("Load imbalance between pools (std / mean)")
    axs[1].set_ylabel("Mean distance RRH - assigned pool (km)")
    for ax in axs:
        ax.set_xlabel("Maximum round-trip delay (us)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.3)
    axs[0].legend()
    axs[0].set_title(f"{k} pools at {hora_txt(hora)}")
    guardar(fig, "fase2_fig4_balanceo_vs_retardo.png")
    return T


# ---------------- Programa principal ----------------
def main():
    esc0 = construir_escenario()
    r, P, P_real, idx_cand = esc0
    D_cand = f1.distancias(P, P[idx_cand])
    esc = (r, P, P_real, idx_cand, D_cand)
    n = len(r)
    perfiles = cargar_perfiles()
    factor = np.ones(n)
    if cfg.f2_variabilidad_carga > 0:
        factor = np.random.default_rng(f1.SEMILLA_TRAFICO + 1).lognormal(0, cfg.f2_variabilidad_carga, n)
    tipos = r.traffic_type.to_numpy()

    # Hora del balanceo: la de maxima carga total (o la indicada)
    totales = [carga_rrh(tipos, h, perfiles, factor).sum() for h in range(24)]
    hora = int(np.argmax(totales)) if cfg.f2_hora == "auto" else int(cfg.f2_hora)
    print(f"RRHs: {n} | retardo RT max: {RT_MAX:.0f} us ({RT_MAX / (2 * US_POR_KM):.1f} km) | "
          f"hora del balanceo: {hora_txt(hora)} (carga total {totales[hora] / 1000:.0f} GBph)")
    print("Tipos:", r.traffic_type.value_counts().to_dict(), "| variabilidad:", cfg.f2_variabilidad_carga)

    resultados = {c: ejecutar_caso(k, esc, perfiles, factor, hora) for c, k in CASOS.items()}
    for c, res in resultados.items():
        print(f"\nCaso {c}: {res['k']} pools (ids {res['ids'].tolist()}) | RRH servidas {res['servida'].sum()} de {n}")
        print(f"  opciones por RRH: min {res['alc'].sum(1).min()}, media {res['alc'].sum(1).mean():.1f}, "
              f"con una sola opcion: {(res['alc'].sum(1) == 1).sum()}")
    princ = resultados[CASO_PRINCIPAL]
    k = len(princ["ids"])

    # Figuras del caso principal
    fig_perfiles(perfiles)
    fig_1a(princ, perfiles, factor, tipos, CASO_PRINCIPAL)
    fig_1b(princ, perfiles, factor, hora, n)
    fig_1c(tipos, perfiles, factor)
    fig_2a(princ)
    cambios = fig_2b(princ, perfiles, factor, tipos, hora, k)
    fig_2c(princ)
    fig_2d(princ)
    fig_2e(princ)
    fig_2f(princ, r)
    T = fig_3(resultados, perfiles, factor, hora)
    barrido = fig_4(princ, perfiles, factor, hora)
    barrido.to_csv("fase2_balanceo_vs_retardo.csv", index=False)

    # CSV
    pd.DataFrame({"hour": range(1, 25), **{t: perfiles[t] for t in perfiles}}).to_csv("fase2_perfiles_trafico.csv", index=False)
    ids = princ["ids"]
    asig = pd.DataFrame({"rrh_id": r.rrh_id, "lon": r.lon, "lat": r.lat, "traffic_type": r.traffic_type,
                         f"carga_{hora:02d}h_mbph": princ["carga"], "n_opciones": princ["alc"].sum(1)})
    for nom, a in [("pool_distancia", ALG_DELAY), ("pool_balanceo_carga", ALG_CARGA), ("pool_balanceo_rrh", ALG_RRH)]:
        asig[nom] = np.where(princ["asig"][a] >= 0, ids[np.clip(princ["asig"][a], 0, None)], 0)
        asig[nom.replace("pool_", "dist_") + "_km"] = np.where(
            princ["asig"][a] >= 0, princ["D"][np.arange(n), np.clip(princ["asig"][a], 0, None)], np.nan)
    asig.to_csv("fase2_rrh_asignacion.csv", index=False)
    T.to_csv("fase2_metricas_casos.csv", index=False)
    cambios.to_csv("fase2_cambios_por_hora.csv", index=False)
    carga24 = []
    for h in range(24):
        c = carga_rrh(tipos, h, perfiles, factor)
        for alg in COL_ALG:
            lp = carga_por_pool(princ["asig"][alg], c, k)
            carga24.append(pd.DataFrame({"hora": h, "algoritmo": alg, "pool_id": ids, "carga_mbph": lp}))
    pd.concat(carga24).to_csv("fase2_carga_pool_24h.csv", index=False)
    var = []
    for t in ["residential", "office"]:
        for nombre, a in [(ALG_DELAY, princ["asig_dist"]), (ALG_CARGA, princ["tipos_alt"][t]["asig_carga"])]:
            m = metricas(a, princ["tipos_alt"][t]["carga"], princ["D"], k)
            var.append(dict(caso=f"all {t}", algoritmo=nombre, **m))
    pd.DataFrame(var).to_csv("fase2_metricas_residencial_oficina.csv", index=False)

    # Resumen por consola
    pd.set_option("display.width", 220)
    cols = ["caso", "pools", "algoritmo", "carga_media_mbph", "carga_std_mbph", "cv_carga", "max_sobre_media",
            "rrh_por_pool_min", "rrh_por_pool_max", "dist_media_km", "dist_max_km"]
    print(f"\nMetricas a las {hora_txt(hora)}:\n", T[cols].round(2).to_string(index=False))
    print(f"\nRRH que cambian de pool respecto a la asignacion de las {hora_txt(hora)} (balanceo por carga):")
    print(cambios[cambios.hora.isin(HORAS_FIGURA)][["hora", "rrh_que_cambian_vs_hora_pico"]].to_string(index=False))
    print(f"\nFiguras guardadas en: {DIR_FIGURAS}")


if __name__ == "__main__":
    main()
