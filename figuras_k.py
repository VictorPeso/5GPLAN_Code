"""
figuras_k.py
Genera las figuras para justificar la eleccion de K y comparar k-center con k-means,
a partir de rrh_escenario.csv. No depende de cran_escenario.py. Etiquetas en ingles.

  fig_A_distance_vs_K.png      distancia maxima y media vs K (k-center), con K elegido marcado
  fig_B_marginal_gain.png      cuanto baja la distancia maxima por cada pool extra (k-center)
  fig_C_distance_cdf.png       % de RRHs dentro de una distancia dada (k-center, varios K)
  fig_D_load_per_pool.png      RRHs por pool, k-center K=K_COMPARAR frente a K=K_ELEGIDO
  fig_E_methods_distance.png   distancia maxima y media vs K: k-center frente a k-means
  fig_F_methods_balance.png    % de RRHs en el pool mayor vs K: k-center frente a k-means
  fig_G_methods_load.png       RRHs por pool con K_ELEGIDO: k-center frente a k-means
  fig_H_methods_map.png        mapa de pools y asignacion con K_ELEGIDO para cada metodo
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = "rrh_escenario.csv"
K_ELEGIDO = 10
K_COMPARAR = 5                      # el K "malo" para comparar el reparto de carga
K_LISTA = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
K_CDF = [5, 10, 15]
SEMILLA = 42

COL_KC, COL_KM = "tab:blue", "tab:green"


# ---------------- Algoritmos (copia de cran_escenario.py) ----------------
def a_km(lon, lat, lon0, lat0):
    x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * 111.32
    return np.column_stack([x, y])


def dist_a_punto(P, p):
    return np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])


def farthest_first(P, k, candidatos):
    """k-center greedy; solo los candidatos pueden ser pool."""
    idx_cand = np.flatnonzero(candidatos)
    k = min(k, len(idx_cand))
    c = P.mean(axis=0)
    primero = int(idx_cand[np.argmin(dist_a_punto(P[idx_cand], c))])
    pools = [primero]
    dmin = dist_a_punto(P, P[primero])
    asign = np.zeros(len(P), dtype=int)
    for i in range(1, k):
        nuevo = int(idx_cand[np.argmax(dmin[idx_cand])])
        pools.append(nuevo)
        d = dist_a_punto(P, P[nuevo])
        mejora = d < dmin
        asign[mejora] = i
        dmin = np.minimum(dmin, d)
    return pools, asign, dmin


def kmeans_pools(P, k, candidatos, semilla=0, reinicios=10, iteraciones=100):
    """k-means; cada pool = RRH candidato mas cercano al centro de su grupo."""
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
    for c in mejor[1]:
        d = np.hypot(P[idx_cand, 0] - c[0], P[idx_cand, 1] - c[1])
        for j in np.argsort(d):
            if int(idx_cand[j]) not in pools:
                pools.append(int(idx_cand[j]))
                break
    D = matriz(P[pools])
    return pools, D.argmin(axis=1), D.min(axis=1)


def main():
    rrh = pd.read_csv(CSV)
    real = rrh["tipo"].values == "real"
    P = a_km(rrh["lon"], rrh["lat"], rrh.loc[real, "lon"].mean(), rrh.loc[real, "lat"].mean())

    ks = sorted(set(K_LISTA + K_CDF + [K_ELEGIDO, K_COMPARAR]))
    kc = {k: farthest_first(P, k, real) for k in ks}
    km = {k: kmeans_pools(P, k, real, SEMILLA + k) for k in ks}
    dmax = np.array([kc[k][2].max() for k in ks])
    dmed = np.array([kc[k][2].mean() for k in ks])

    # ---- A: distancia vs K (k-center) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, dmax, "o-", label="Maximum distance")
    ax.plot(ks, dmed, "s--", label="Average distance")
    ax.axvline(K_ELEGIDO, color="red", ls=":", lw=1.5)
    i = ks.index(K_ELEGIDO)
    ax.annotate(f"K = {K_ELEGIDO}: {dmax[i]:.1f} km", (K_ELEGIDO, dmax[i]),
                textcoords="offset points", xytext=(15, 12), color="red",
                arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("Number of BBU pools (K)")
    ax.set_ylabel("RRH to BBU pool distance (km)")
    ax.set_title("Farthest-first k-center")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_A_distance_vs_K.png", dpi=200)
    plt.close(fig)

    # ---- B: mejora marginal (k-center) ----
    etiquetas, ganancias = [], []
    for a, b in zip(ks[:-1], ks[1:]):
        g = (dmax[ks.index(a)] - dmax[ks.index(b)]) / (b - a)
        etiquetas.append(f"{a}->{b}")
        ganancias.append(g)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colores = ["tab:red" if e.startswith(f"{K_ELEGIDO}->") else "tab:blue" for e in etiquetas]
    ax.bar(etiquetas, ganancias, color=colores)
    ax.set_xlabel("Increase in number of pools")
    ax.set_ylabel("Reduction of max distance per added pool (km)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_B_marginal_gain.png", dpi=200)
    plt.close(fig)

    # ---- C: CDF de distancias (k-center) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in K_CDF:
        d = np.sort(kc[k][2])
        ax.plot(d, np.arange(1, len(d) + 1) / len(d) * 100, label=f"K = {k}")
    ax.set_xlabel("RRH to BBU pool distance (km)")
    ax.set_ylabel("RRHs within that distance (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_C_distance_cdf.png", dpi=200)
    plt.close(fig)

    # ---- D: carga por pool (k-center, dos K) ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, k in zip(axs, [K_COMPARAR, K_ELEGIDO]):
        cuentas = np.sort(np.bincount(kc[k][1], minlength=k))[::-1]
        ax.bar(range(1, len(cuentas) + 1), cuentas)
        ax.set_title(f"k-center, K = {k}: largest pool holds {cuentas[0] / cuentas.sum() * 100:.0f}% of RRHs")
        ax.set_xlabel("BBU pool (sorted by size)")
        ax.set_xticks(range(1, len(cuentas) + 1))
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("Number of RRHs assigned")
    fig.tight_layout()
    fig.savefig("fig_D_load_per_pool.png", dpi=200)
    plt.close(fig)

    # ---- E: distancias, k-center vs k-means ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, f, titulo in zip(axs, [np.max, np.mean], ["Maximum distance", "Average distance"]):
        ax.plot(ks, [f(kc[k][2]) for k in ks], "o-", color=COL_KC, label="k-center")
        ax.plot(ks, [f(km[k][2]) for k in ks], "s--", color=COL_KM, label="k-means")
        ax.axvline(K_ELEGIDO, color="red", ls=":", lw=1)
        ax.set_xlabel("Number of BBU pools (K)")
        ax.set_ylabel(f"{titulo} (km)")
        ax.set_title(titulo)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("fig_E_methods_distance.png", dpi=200)
    plt.close(fig)

    # ---- F: equilibrio, k-center vs k-means ----
    def mayor(res, k):
        c = np.bincount(res[k][1], minlength=k)
        return c.max() / c.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [mayor(kc, k) for k in ks], "o-", color=COL_KC, label="k-center")
    ax.plot(ks, [mayor(km, k) for k in ks], "s--", color=COL_KM, label="k-means")
    ax.axvline(K_ELEGIDO, color="red", ls=":", lw=1)
    ax.set_xlabel("Number of BBU pools (K)")
    ax.set_ylabel("RRHs in the largest pool (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_F_methods_balance.png", dpi=200)
    plt.close(fig)

    # ---- G: carga por pool con K_ELEGIDO ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, res, nombre, col in zip(axs, [kc, km], ["k-center", "k-means"], [COL_KC, COL_KM]):
        cuentas = np.sort(np.bincount(res[K_ELEGIDO][1], minlength=K_ELEGIDO))[::-1]
        ax.bar(range(1, len(cuentas) + 1), cuentas, color=col)
        ax.set_title(f"{nombre}, K = {K_ELEGIDO}: largest pool {cuentas[0] / cuentas.sum() * 100:.0f}%")
        ax.set_xlabel("BBU pool (sorted by size)")
        ax.set_xticks(range(1, len(cuentas) + 1))
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("Number of RRHs assigned")
    fig.tight_layout()
    fig.savefig("fig_G_methods_load.png", dpi=200)
    plt.close(fig)

    # ---- H: mapa de pools y asignacion ----
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, res, nombre in zip(axs, [kc, km], ["k-center", "k-means"]):
        pools, asign, d = res[K_ELEGIDO]
        ax.scatter(P[:, 0], P[:, 1], c=[cmap(a % 10) for a in asign], s=6, alpha=0.7)
        ax.scatter(P[pools, 0], P[pools, 1], marker="*", s=260, c="black",
                   edgecolors="white", linewidths=1.2, zorder=5, label="BBU pool")
        ax.set_title(f"{nombre}, K = {K_ELEGIDO}  (max {d.max():.1f} km, mean {d.mean():.1f} km)")
        ax.set_xlabel("East-West (km)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    axs[0].set_ylabel("North-South (km)")
    fig.tight_layout()
    fig.savefig("fig_H_methods_map.png", dpi=200)
    plt.close(fig)

    print("Figuras guardadas: fig_A .. fig_H")
    for nombre, res in [("k-center", kc), ("k-means", km)]:
        for k in [K_COMPARAR, K_ELEGIDO]:
            print(f"{nombre:8s} K={k:2d}: dist max {res[k][2].max():5.2f} km, "
                  f"media {res[k][2].mean():4.2f} km, pool mayor {mayor(res, k):.0f}%")


if __name__ == "__main__":
    main()