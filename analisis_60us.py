"""Analisis con RTT max 60 us: distribucion de distancia/retardo con 18, 14 y 10 pools.
Reutiliza la poda de fase1_min_delay.py. Uso: python analisis_60us.py [carpeta]"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fase1_min_delay as f
carp = sys.argv[1] if len(sys.argv) > 1 else "."
out = os.path.join(carp, "Figuras"); os.makedirs(out, exist_ok=True)
RT = 60.0; NS = [18, 14, 10]; UKM = f.US_POR_KM
rrh = pd.read_csv(os.path.join(carp, "fase1_rrh.csv")); pools = pd.read_csv(os.path.join(carp, "fase1_pools.csv"))
lon0, lat0 = 2.1542, 41.3926
R = f.a_km(rrh.lon, rrh.lat, lon0, lat0); P = f.a_km(pools.lon, pools.lat, lon0, lat0)
D = f.distancias(R, P)
# comprobacion: debe coincidir con el CSV
assert abs(D.min(1) - rrh.dist_al_mas_cercano_km).max() < 1e-3, "proyeccion distinta"
tabla = f.poda_pools(D, RT)
dmax = RT / (2 * UKM)
orden = tabla.pool_retirado.tolist()
def activos(n):
    ret = set(tabla[tabla.pools_activos >= n].pool_retirado) - {0}
    return [i for i in range(len(pools)) if (pools.pool_id[i]) not in ret]
res = {}; filas = []
for n in NS:
    act = activos(n); Da = D[:, act]; dm = Da.min(1); serv = dm <= dmax
    res[n] = dict(act=act, dm=dm, serv=serv, asig=np.array(act)[Da.argmin(1)])
    ds = dm[serv]; ret = 2 * UKM * ds
    carga = np.bincount(res[n]["asig"][serv], minlength=len(pools))[act]
    filas.append(dict(pools=n, ids=";".join(str(pools.pool_id[i]) for i in act),
        servidas=int(serv.sum()), no_servidas=int((~serv).sum()), pct_servidas=100*serv.mean(),
        dist_media_km=ds.mean(), dist_mediana_km=np.median(ds), dist_p90_km=np.percentile(ds, 90),
        dist_max_km=ds.max(), retardo_medio_us=ret.mean(), retardo_mediana_us=np.median(ret),
        retardo_p90_us=np.percentile(ret, 90), retardo_max_us=ret.max(),
        rrh_por_pool_min=carga.min(), rrh_por_pool_media=carga.mean(), rrh_por_pool_max=carga.max()))
T = pd.DataFrame(filas); T.to_csv(os.path.join(carp, "fase1_analisis_60us.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.drop(columns="ids").round(2).T.to_string()); print(T[["pools","ids"]].to_string(index=False))
col = {18: "#1f77b4", 14: "#ff7f0e", 10: "#2ca02c"}
# fig10: CDF de distancia + retardo
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for n in NS:
    d = np.sort(res[n]["dm"][res[n]["serv"]]); y = np.arange(1, len(d)+1) / len(rrh)
    ax[0].plot(d, y, color=col[n], lw=2, label=f"{n} pools ({100*res[n]['serv'].mean():.1f}% served)")
    ax[1].plot(2*UKM*d, y, color=col[n], lw=2, label=f"{n} pools")
ax[0].axvline(dmax, color="r", ls="--", label=f"Limit {dmax:.0f} km"); ax[1].axvline(RT, color="r", ls="--", label="Limit 60 us")
ax[0].set_xlabel("Distance RRH - closest BBU pool (km)"); ax[1].set_xlabel("Round-trip delay to closest pool (us)")
for a in ax: a.set_ylabel("CDF (fraction of all RRHs)"); a.grid(alpha=.3); a.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(out, "fase1_fig10_cdf_60us.png"), dpi=200); plt.close()
# fig11: histogramas
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True); b = np.linspace(0, dmax, 25)
for a, n in zip(ax, NS):
    d = res[n]["dm"][res[n]["serv"]]; a.hist(d, bins=b, color=col[n], edgecolor="w")
    a.axvline(d.mean(), color="k", ls="--", label=f"mean {d.mean():.2f} km"); a.axvline(np.median(d), color="k", ls=":", label=f"median {np.median(d):.2f} km")
    a.set_title(f"{n} pools - {int((~res[n]['serv']).sum())} RRH not served"); a.set_xlabel("Distance to closest pool (km)"); a.legend(); a.grid(alpha=.3)
ax[0].set_ylabel("Number of RRHs"); plt.tight_layout(); plt.savefig(os.path.join(out, "fase1_fig11_hist_60us.png"), dpi=200); plt.close()
# fig12: mapas
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
for a, n in zip(ax, NS):
    s = res[n]["serv"]; sc = a.scatter(R[s, 0], R[s, 1], c=2*UKM*res[n]["dm"][s], s=7, cmap="viridis", vmin=0, vmax=RT)
    if (~s).any(): a.scatter(R[~s, 0], R[~s, 1], c="red", marker="x", s=22, label=f"not served ({(~s).sum()})")
    a.scatter(P[res[n]["act"], 0], P[res[n]["act"], 1], c="k", marker="s", s=45, edgecolor="w", label=f"{n} pools")
    a.set_title(f"{n} pools"); a.set_aspect("equal"); a.set_xlabel("x (km)"); a.legend(loc="lower left", fontsize=8)
ax[0].set_ylabel("y (km)"); fig.colorbar(sc, ax=ax, shrink=.8, label="Round-trip delay (us)")
plt.savefig(os.path.join(out, "fase1_fig12_mapas_60us.png"), dpi=200, bbox_inches="tight"); plt.close()
