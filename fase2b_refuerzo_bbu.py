"""
fase2b_refuerzo_bbu.py
Fase 2 (ampliacion): refuerzo de BBU pools en las zonas sobrecargadas.

Despues del balanceo por trafico (fase2_load_balancing.py), si algun pool tiene bastante mas carga
que la media, este script:
  1) Detecta los pools sobrecargados (carga > media * (1 + f2_umbral_sobrecarga)).
  2) Prueba a anadir una BBU nueva en un emplazamiento de RRH real de la zona tensionada, a mas de
     f2_dist_min_bbu_km de cualquier otra BBU (no tiene sentido apilar BBU en el mismo sitio). Los
     candidatos se ordenan por la carga de las RRH de los pools sobrecargados que alcanzarian
     dentro del retardo maximo (trafico + delay) y se prueban los mejores (f2_candidatos_max).
     Cada prueba rehace el balanceo completo; se elige la que deja la carga mas equilibrada y, con
     peso f2_peso_retardo, la distancia (retardo) media mas corta.
  3) Si la carga mejora (menor desequilibrio y menor carga maxima) la BBU se queda; si no, se
     descarta y se para.
  4) Si la mejora es pequena (< f2_mejora_minima), se intenta ademas: (a) mover de sitio la BBU
     sobrecargada, y (b) un refinamiento que reparte RRH del pool mas cargado entre los pools de
     su alrededor que alcanzan. Se queda con lo que mas mejora.
  5) Se repite hasta f2_max_bbus_extra BBU nuevas.

Uso: python fase2b_refuerzo_bbu.py   (parametros f2_* en config.py, cambiables con --nombre=valor)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fase1_min_delay as f1
import fase2_load_balancing as f2

cfg = f2.cfg
RT, US = f2.RT_MAX, f2.US_POR_KM
D_MAX = RT / (2 * US)
UMBRAL = float(cfg.f2_umbral_sobrecarga)
MAX_EXTRA = int(cfg.f2_max_bbus_extra)
HOLGURA = float(cfg.f2_umbral_holgura)           # BBU con holgura: carga < media * (1 - HOLGURA)
MEJORA_MIN = float(cfg.f2_mejora_minima)
MAX_CAND = int(cfg.f2_candidatos_max)
DIST_MIN = float(cfg.f2_dist_min_bbu_km)          # separacion minima entre dos BBU (km)
PESO_RETARDO = float(cfg.f2_peso_retardo)         # peso de la distancia media (retardo) al elegir sitio


# ---------------- Evaluacion de una configuracion ----------------
def resumen(D, alc, asig, carga, refinado=False):
    k = D.shape[1]
    lp = f2.carga_por_pool(asig, carga, k)
    ok = asig >= 0
    dist = D[np.arange(len(asig))[ok], asig[ok]]
    return dict(D=D, alc=alc, asig=asig, lp=lp, cv=lp.std() / lp.mean(), max_sobre_media=lp.max() / lp.mean(),
                dist_media=dist.mean(), servidas=int(ok.sum()), refinado=refinado)


def evaluar(D, carga):
    """Balanceo por trafico completo para una matriz de distancias RRH x pools."""
    alc = 2 * D * US <= RT
    return resumen(D, alc, f2.asignar_balanceo(D, alc, carga, "carga"), carga)


def mejor_prueba(est, candidatos, D_full, carga, columna=None):
    """Prueba cada candidato (como pool nuevo, o sustituyendo la columna `columna`) y devuelve el mejor."""
    mejor = None
    for c in candidatos:
        if columna is None:
            D_try = np.column_stack([est["D"], D_full[:, c]])
        else:
            D_try = est["D"].copy()
            D_try[:, columna] = D_full[:, c]
        e = evaluar(D_try, carga)
        if e["servidas"] < est["servidas"]:
            continue
        clave = (e["cv"] + PESO_RETARDO * e["dist_media"] / D_MAX, e["dist_media"])
        if mejor is None or clave < mejor[0]:
            mejor = (clave, c, e)
    return None if mejor is None else (mejor[1], mejor[2])


def candidatos_zona(est, carga, D_full, idx, sobrecargados, excluir=()):
    """Sitios de RRH real ordenados por la carga de RRH de pools sobrecargados que alcanzarian."""
    en_sobre = np.isin(est["asig"], sobrecargados)
    pot = np.array([carga[en_sobre & (D_full[:, c] <= D_MAX)].sum() for c in range(D_full.shape[1])])
    pot[list(set(idx.tolist()) | set(excluir))] = 0
    for b in idx:                                        # separacion minima respecto al resto de BBU
        if b not in excluir:
            pot[D_full[b, :] < DIST_MIN] = 0
    orden = np.argsort(-pot)
    return [int(c) for c in orden[:MAX_CAND] if pot[c] > 0]


def _mover(D, alc, asig, lp, carga, techo, holgura, solo_holgura, max_mov):
    """Mueve RRH de los pools por encima de `techo` a otros pools que las alcanzan. Devuelve cuantas movio."""
    n = 0
    for _ in range(max_mov):
        mov = None
        for j in np.argsort(-lp):
            if lp[j] <= techo:
                break
            mejor = None
            for i in np.where(asig == j)[0]:
                for t in np.where(alc[i])[0]:
                    if t == j:
                        continue
                    if solo_holgura and (lp[t] >= holgura or lp[t] + carga[i] > techo):
                        continue
                    nuevo = max(lp[t] + carga[i], lp[j] - carga[i])
                    if nuevo < lp[j] - 1e-9 and (mejor is None or nuevo < mejor[0] - 1e-9
                                                or (abs(nuevo - mejor[0]) < 1e-9 and D[i, t] < D[mejor[1], mejor[2]])):
                        mejor = (nuevo, i, t)
            if mejor is not None:
                mov = (j,) + mejor
                break
        if mov is None:
            break
        j, _, i, t = mov
        lp[j] -= carga[i]
        lp[t] += carga[i]
        asig[i] = t
        n += 1
    return n


def redistribuir(est, carga, max_mov=5000):
    """Redistribuye RRH entre las BBU existentes, siempre dentro del retardo maximo, en dos fases:
      A) de los pools sobrecargados a BBU con holgura (carga < media*(1-f2_umbral_holgura)), que no pasan del
         umbral de sobrecarga al recibirlas;
      B) si siguen sobrecargados, tambien a BBU vecinas que no tienen holgura pero con las que baja la carga maxima.
    Cada movimiento baja la carga del pool de origen y deja al destino por debajo de ella; en caso de empate
    gana el de menor distancia.
    """
    D, alc, asig = est["D"], est["alc"], est["asig"].copy()
    lp = est["lp"].copy()
    media = lp.mean()
    techo, holgura = media * (1 + UMBRAL), media * (1 - HOLGURA)
    na = _mover(D, alc, asig, lp, carga, techo, holgura, True, max_mov)
    nb = _mover(D, alc, asig, lp, carga, techo, holgura, False, max_mov)
    out = resumen(D, alc, asig, carga, refinado=True)
    out["mov_holgura"], out["mov_vecinas"] = na, nb
    return out


# ---------------- Refuerzo de un caso ----------------
def reforzar(caso, res, D_full, carga, P, r):
    idx = res["idx"].copy()
    etiquetas = [str(i) for i in res["ids"]]
    est = resumen(res["D"], res["alc"], res["asig_carga"], carga)
    pasos = []

    def fila(paso, accion, e, decision, detalle=""):
        pasos.append(dict(caso=caso, paso=paso, accion=accion, pools=e["D"].shape[1], decision=decision, detalle=detalle,
                          cv_carga=e["cv"], max_sobre_media=e["max_sobre_media"], carga_max_mbph=e["lp"].max(),
                          carga_media_mbph=e["lp"].mean(), dist_media_km=e["dist_media"]))

    fila(0, "Balanceo por trafico (base)", est, "-")
    # Antes de anadir BBU: aprovechar las BBU existentes con holgura (carga por debajo de la media)
    media0 = est["lp"].mean()
    holg = np.where(est["lp"] < media0 * (1 - HOLGURA))[0]
    sobre0 = np.where(est["lp"] > media0 * (1 + UMBRAL))[0]
    en = np.isin(est["asig"], sobre0)
    llegan = int(sum(any(t in holg and t != est["asig"][i] for t in np.where(est["alc"][i])[0]) for i in np.where(en)[0]))
    txt0 = (f"BBU con holgura: {len(holg)} de {est['D'].shape[1]}; pools sobrecargados: {len(sobre0)}; "
            f"RRH de estos que alcanzan una BBU con holgura: {llegan} de {int(en.sum())}")
    e0 = redistribuir(est, carga)
    ok0 = e0["cv"] < est["cv"] - 1e-9 and e0["lp"].max() <= est["lp"].max() + 1e-9
    fila(0, "Redistribuir RRH hacia BBU con holgura (sin BBU nuevas)", e0 if ok0 else est, "aceptada" if ok0 else "sin efecto",
         txt0 + (f"; RRH movidas: {e0['mov_holgura']} a BBU con holgura + {e0['mov_vecinas']} a vecinas; "
                 f"desequilibrio {est['cv']:.3f} -> {e0['cv']:.3f}; carga maxima {est['lp'].max():,.0f} -> {e0['lp'].max():,.0f} MBph"
                 if ok0 else "; no se puede mover ninguna RRH sin superar el umbral"))
    if ok0:
        est = e0
    est0 = est
    n_nuevos = 0
    for paso in range(1, MAX_EXTRA + 1):
        media = est["lp"].mean()
        sobre = np.where(est["lp"] > media * (1 + UMBRAL))[0]
        if len(sobre) == 0:
            fila(paso, "Sin pools sobrecargados", est, "fin", f"ninguno supera la media en mas de un {UMBRAL:.0%}")
            break
        txt_sobre = "pools sobrecargados: " + ", ".join(f"{etiquetas[j]} ({est['lp'][j] / media:.2f}x)" for j in sobre)
        cands = candidatos_zona(est, carga, D_full, idx, sobre)
        prueba = mejor_prueba(est, cands, D_full, carga) if cands else None
        if prueba is None:
            fila(paso, "Anadir BBU", est, "rechazada", txt_sobre + "; sin candidatos validos")
            break
        c, e = prueba
        if est["refinado"]:
            e = redistribuir(e, carga)
        mejora = (est["cv"] - e["cv"]) / est["cv"]
        lat, lon = r.lat[c], r.lon[c]
        sitio = f"sitio RRH {int(r.rrh_id[c])} ({lat:.4f}, {lon:.4f})"
        if not (e["cv"] < est["cv"] and e["lp"].max() < est["lp"].max()):
            fila(paso, f"Anadir BBU en {sitio}", e, "rechazada (no mejora)", txt_sobre)
            break
        idx_new, et_new, e_new = np.append(idx, c), etiquetas + [f"N{n_nuevos + 1}"], e
        detalle = f"{txt_sobre}; mejora del desequilibrio {mejora:.0%}"
        if mejora < MEJORA_MIN:
            # La mejora es pequena: probar a mover de sitio la BBU mas sobrecargada y a refinar el reparto
            fila(paso, f"Anadir BBU en {sitio}", e, "mejora pequena", detalle)
            j = int(est["lp"].argmax())
            cj = candidatos_zona(e, carga, D_full, idx_new, [j], excluir=[idx_new[j]])
            reub = mejor_prueba(e, cj, D_full, carga, columna=j) if cj else None
            if reub is not None:
                cc, ee = reub
                m2 = (est["cv"] - ee["cv"]) / est["cv"]
                ok = ee["cv"] < e_new["cv"] - 1e-9
                fila(paso, "Alternativa: mover BBU sobrecargada", ee, "aceptada" if ok else "descartada",
                     f"mover la BBU {etiquetas[j]} al sitio RRH {int(r.rrh_id[cc])}: desequilibrio {ee['cv']:.3f} (mejora total {m2:.0%})")
                if ok:
                    idx_new = idx_new.copy()
                    idx_new[j] = cc
                    e_new = ee
            ee = redistribuir(e_new, carga)
            m2 = (est["cv"] - ee["cv"]) / est["cv"]
            ok = ee["cv"] < e_new["cv"] - 1e-9
            fila(paso, "Alternativa: redistribuir RRH hacia BBU con holgura", ee, "aceptada" if ok else "descartada",
                 f"desequilibrio {ee['cv']:.3f} (mejora total {m2:.0%})")
            if ok:
                e_new = ee
            fila(paso, "Resultado del paso", e_new, "aceptada",
                 f"desequilibrio {est['cv']:.3f} -> {e_new['cv']:.3f} (mejora total {(est['cv'] - e_new['cv']) / est['cv']:.0%}); "
                 f"carga maxima {est['lp'].max():,.0f} -> {e_new['lp'].max():,.0f} MBph")
        else:
            fila(paso, f"Anadir BBU en {sitio}", e, "aceptada", detalle)
        idx, etiquetas, est = idx_new, et_new, e_new
        n_nuevos += 1
    return dict(est=est, est0=est0, idx=idx, etiquetas=etiquetas, pasos=pasos, n_nuevos=n_nuevos)


# ---------------- Figuras ----------------
def fig_5a(res, ref, carga):
    base = f2.carga_por_pool(res["asig_carga"], carga, len(res["ids"]))
    final = ref["est"]["lp"]
    k0 = len(res["ids"])
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    f2.barras(axs[0], [str(i) for i in res["ids"]], {"Load balancing (traffic)": base}, {"Load balancing (traffic)": f2.COL_ALG[f2.ALG_CARGA]})
    axs[0].axhline(base.mean(), color="k", ls="--", lw=1, label=f"Mean {base.mean():,.0f} MBph")
    axs[0].set_title(f"Before: {k0} pools (imbalance {base.std() / base.mean():.2f})")
    cols = ["#d62728" if not e.startswith("N") else "#2ca02c" for e in ref["etiquetas"]]
    x = np.arange(len(final))
    axs[1].bar(x, final, color=cols, edgecolor="white", linewidth=.4)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(ref["etiquetas"])
    axs[1].axhline(final.mean(), color="k", ls="--", lw=1, label=f"Mean {final.mean():,.0f} MBph")
    axs[1].set_title(f"After: {len(final)} pools (imbalance {ref['est']['cv']:.2f}); green = new BBU")
    axs[1].set_xlabel("BBU pool identifier")
    axs[1].grid(axis="y", alpha=.3)
    axs[1].set_axisbelow(True)
    axs[0].set_ylabel("Total load per pool (MBph)")
    for ax in axs:
        ax.set_ylim(0, 1.2 * max(base.max(), final.max()))
        ax.legend(loc="upper right", ncol=2, fontsize=9)
    f2.guardar(fig, "fase2_fig5a_refuerzo_carga_pools.png")


def fig_5b(res, ref, r):
    P = f1.a_km(r.lon, r.lat, 2.1542, 41.3926)
    cmap = plt.get_cmap("tab20", 20)
    NUEVOS = ["#ff00ff", "#00e5ff", "#ffd600", "#00ff55", "#ff1744"]      # colores vivos, solo para pools nuevos
    k0 = len(res["idx"])
    antes = res["asig_carga"]
    despues = ref["est"]["asig"]

    def color(j):
        return NUEVOS[(j - k0) % len(NUEVOS)] if j >= k0 else cmap(j % 20)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    for ax, asig, idx, et, tit in [(axs[0], antes, res["idx"], [str(i) for i in res["ids"]], "Before"),
                                   (axs[1], despues, ref["idx"], ref["etiquetas"], "After adding BBUs")]:
        ok = asig >= 0
        viejo = ok & (asig < k0)
        ax.scatter(P[viejo, 0], P[viejo, 1], c=[cmap(j % 20) for j in asig[viejo]], s=8)
        for j in range(k0, len(idx)):                     # RRH de los pools nuevos: color propio, mas grandes
            m = asig == j
            ax.scatter(P[m, 0], P[m, 1], c=color(j), s=18, edgecolor="k", linewidth=.3, zorder=3)
        for j, c in enumerate(idx):
            nuevo = et[j].startswith("N")
            ax.scatter(P[c, 0], P[c, 1], c=color(j) if nuevo else "k", marker="*" if nuevo else "s",
                       s=300 if nuevo else 60, edgecolor="k" if nuevo else "w", zorder=5)
            ax.annotate(et[j], P[c], textcoords="offset points", xytext=(6, 6), fontsize=8, weight="bold")
        ax.set_title(tit)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
    axs[0].set_ylabel("y (km)")
    f2.guardar(fig, "fase2_fig5b_refuerzo_mapa.png")


def fig_5c(todos):
    T = pd.DataFrame([p for caso in todos.values() for p in caso["pasos"] if p["decision"] in ("-", "aceptada", "mejora pequena")])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for caso, g in T.groupby("caso", sort=False):
        g = g.drop_duplicates("pools", keep="last")
        axs[0].plot(g.pools, g.cv_carga, "o-", lw=2, label=caso)
        axs[1].plot(g.pools, g.max_sobre_media, "o-", lw=2, label=caso)
    axs[0].set_ylabel("Load imbalance between pools (std / mean)")
    axs[1].set_ylabel("Most loaded pool / mean load")
    for ax in axs:
        ax.set_xlabel("Number of BBU pools")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(alpha=.3)
        ax.set_ylim(bottom=0)
    axs[0].legend(title="Case")
    f2.guardar(fig, "fase2_fig5c_refuerzo_pasos.png")


# ---------------- Resumen final antes / despues ----------------
def estado(D, asig, carga):
    """Metricas de un reparto RRH -> pool: pools, desequilibrio, carga maxima, distancia y retardo (RTT)."""
    k = D.shape[1]
    lp = f2.carga_por_pool(asig, carga, k)
    ok = asig >= 0
    dist = D[np.arange(len(asig))[ok], asig[ok]]
    rtt = 2 * US * dist
    return dict(pools=k, cv_carga=lp.std() / lp.mean(), carga_max_mbph=lp.max(), carga_media_mbph=lp.mean(),
                max_sobre_media=lp.max() / lp.mean(), dist_media_km=dist.mean(), dist_max_km=dist.max(),
                retardo_medio_us=rtt.mean(), retardo_max_us=rtt.max(),
                pct_rrh_mitad_retardo=100 * (rtt <= RT / 2).mean(), pct_rrh_en_retardo=100 * (rtt <= RT + 1e-9).mean(),
                rrh_servidas=int(ok.sum()))


def tabla_resumen(todos, resultados, cargas):
    filas = []
    for caso, ref in todos.items():
        res, carga = resultados[caso], cargas[caso]
        etapas = [("1 Distance only", estado(res["D"], res["asig_dist"], carga)),
                  ("2 Traffic balancing", estado(res["D"], res["asig_carga"], carga)),
                  ("3 Rebalancing (existing BBUs)", estado(ref["est0"]["D"], ref["est0"]["asig"], carga)),
                  ("4 Rebalancing + new BBUs", estado(ref["est"]["D"], ref["est"]["asig"], carga))]
        base = etapas[1][1]
        for nombre, e in etapas:
            filas.append(dict(caso=caso, etapa=nombre, bbu_nuevas=ref["n_nuevos"] if nombre.startswith("4") else 0, **e,
                              mejora_cv_vs_balanceo_pct=100 * (base["cv_carga"] - e["cv_carga"]) / base["cv_carga"],
                              mejora_cv_por_bbu_nueva=((base["cv_carga"] - e["cv_carga"]) / ref["n_nuevos"]
                                                       if nombre.startswith("4") and ref["n_nuevos"] else np.nan)))
    return pd.DataFrame(filas)


def fig_5d(T):
    casos = list(dict.fromkeys(T.caso))
    etapas = list(dict.fromkeys(T.etapa))
    medidas = [("cv_carga", "Load imbalance (std / mean)"), ("max_sobre_media", "Most loaded pool / mean"),
               ("dist_media_km", "Mean RRH-BBU distance (km)"), ("retardo_max_us", "Maximum round-trip delay (us)")]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    w = 0.8 / len(etapas)
    for ax, (col, tit) in zip(axs, medidas):
        for i, et in enumerate(etapas):
            v = [T[(T.caso == c) & (T.etapa == et)][col].iloc[0] for c in casos]
            b = ax.bar(np.arange(len(casos)) + (i - (len(etapas) - 1) / 2) * w, v, w, label=et[2:])
            ax.bar_label(b, fmt="%.2f" if col != "retardo_max_us" else "%.0f", fontsize=6)
        ax.set_xticks(np.arange(len(casos)))
        pools = [f"{c}\n({T[(T.caso == c) & (T.etapa == etapas[0])].pools.iloc[0]} -> {T[(T.caso == c) & (T.etapa == etapas[-1])].pools.iloc[0]} pools)" for c in casos]
        ax.set_xticklabels(pools, fontsize=8)
        ax.set_title(tit, fontsize=10)
        ax.grid(axis="y", alpha=.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    axs[0].legend(fontsize=7, loc="upper right")
    f2.guardar(fig, "fase2_fig5d_refuerzo_comparativa.png")


# ---------------- Programa principal ----------------
def main():
    r, P, P_real, idx_cand = f2.construir_escenario()
    D_cand = f1.distancias(P, P[idx_cand])
    esc = (r, P, P_real, idx_cand, D_cand)
    perfiles = f2.cargar_perfiles()
    factor = np.ones(len(r))
    if cfg.f2_variabilidad_carga > 0:
        factor = np.random.default_rng(f1.SEMILLA_TRAFICO + 1).lognormal(0, cfg.f2_variabilidad_carga, len(r))
    tipos = r.traffic_type.to_numpy()
    totales = [f2.carga_rrh(tipos, h, perfiles, factor).sum() for h in range(24)]
    hora = int(np.argmax(totales)) if cfg.f2_hora == "auto" else int(cfg.f2_hora)
    D_full = f1.distancias(P, P[:len(P_real)])             # distancias a todos los sitios de RRH real
    print(f"Hora {f2.hora_txt(hora)} | retardo RT max {RT:.0f} us | umbral sobrecarga {UMBRAL:.0%} | "
          f"max BBU extra {MAX_EXTRA} | mejora minima {MEJORA_MIN:.0%} | candidatos {MAX_CAND}")
    todos, filas_pools, resultados, cargas = {}, [], {}, {}
    for caso in cfg.f2_pools:
        res = f2.ejecutar_caso(cfg.f2_pools[caso], esc, perfiles, factor, hora)
        ref = reforzar(caso, res, D_full, res["carga"], P, r)
        todos[caso] = ref
        resultados[caso], cargas[caso] = res, res["carga"]
        for j, c in enumerate(ref["idx"]):
            filas_pools.append(dict(caso=caso, pool=ref["etiquetas"][j], rrh_id=int(r.rrh_id[c]), lat=r.lat[c], lon=r.lon[c],
                                    nuevo=ref["etiquetas"][j].startswith("N"), carga_mbph=ref["est"]["lp"][j],
                                    rrh_asignadas=int((ref["est"]["asig"] == j).sum())))
        if caso == f2.CASO_PRINCIPAL:
            res_p, ref_p, carga_p = res, ref, res["carga"]
    pasos = pd.DataFrame([p for caso in todos.values() for p in caso["pasos"]])
    pasos.to_csv("fase2_refuerzo_pasos.csv", index=False)
    pd.DataFrame(filas_pools).to_csv("fase2_refuerzo_pools.csv", index=False)
    fig_5a(res_p, ref_p, carga_p)
    fig_5b(res_p, ref_p, r)
    fig_5c(todos)
    resumen_t = tabla_resumen(todos, resultados, cargas)
    resumen_t.to_csv("fase2_refuerzo_resumen.csv", index=False)
    fig_5d(resumen_t)
    a = pd.DataFrame({"rrh_id": r.rrh_id, "traffic_type": r.traffic_type, "carga_mbph": carga_p,
                      "pool_final": [ref_p["etiquetas"][j] if j >= 0 else "none" for j in ref_p["est"]["asig"]]})
    a.to_csv("fase2_refuerzo_asignacion.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(pasos[["caso", "paso", "accion", "pools", "cv_carga", "max_sobre_media", "carga_max_mbph", "dist_media_km", "decision"]]
          .round(3).to_string(index=False))
    print("\nDetalle:")
    for _, p in pasos[pasos.detalle != ""].iterrows():
        print(f"  [{p.caso} paso {p.paso}] {p.accion} -> {p.decision}: {p.detalle}")
    print("\nRESUMEN FINAL (antes / despues del refuerzo)")
    cols = ["caso", "etapa", "bbu_nuevas", "pools", "cv_carga", "max_sobre_media", "carga_max_mbph", "dist_media_km",
            "retardo_medio_us", "retardo_max_us", "pct_rrh_mitad_retardo", "mejora_cv_por_bbu_nueva"]
    print(resumen_t[cols].round(3).to_string(index=False))
    print(f"\nFiguras guardadas en: {f1.DIR_FIGURAS}")


if __name__ == "__main__":
    main()
