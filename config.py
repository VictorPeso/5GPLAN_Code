"""
config.py
Parametros de entrada de TODOS los scripts del proyecto C-RAN, en un unico sitio.

Dos formas de cambiarlos (sin tocar el codigo de los demas scripts):
  1) Editar los valores de PARAMETROS de abajo.
  2) Sobrescribirlos al ejecutar, con --nombre=valor. Ejemplos:
         python fase1_min_delay.py --n_sinteticos=100 --n_pools=15
         python fase1_min_delay.py --operadores="['VODAFONE']" --retardo_rt_max_us=100
         python fase1_min_delay.py --porcentajes_trafico="{'mixed':0.4,'office':0.4,'residential':0.2}"
         python fase1_min_delay.py --pools_desactivados="[4,5,8]"
     Las listas y los diccionarios se escriben como en Python, entre comillas.
"""
import ast
import sys
from types import SimpleNamespace

PARAMETROS = {
    # ---------- Escenario ----------
    "csv_estaciones": "estaciones_tec.csv",   # salida de anadir_tecnologias.py
    "operadores": ["TELEFONICA"],             # texto contenido en el nombre del operador (sin importar
                                           # mayusculas ni tildes); None = todos los operadores
    "filtro_etiquetas": ["4G", "5G", "4G+5G"],  # tecnologias admitidas; None = no filtrar
    "decimales_agrupar": 5,                   # ~1 m: estaciones con igual coordenada = mismo emplazamiento (RRH)

    # ---------- RRH sinteticas ----------
    "n_sinteticos": 50,                       # numero de RRH sinteticas (0 = ninguna)
    "min_dist_rrh_km": 0.1,                   # distancia minima entre una RRH sintetica y cualquier otra
    "max_dist_a_real_km": 1.0,                # radio en el que se cuentan RRH reales vecinas
    "min_vecinos_reales": 3,                  # reales necesarias dentro de ese radio (evita mar / zonas vacias)

    # ---------- Patrones de trafico ----------
    # Porcentaje de RRH de cada tipo de zona (deben sumar 1). Se pueden cambiar los porcentajes,
    # pero los nombres deben ser mixed / office / residential.
    "porcentajes_trafico": {"mixed": 0.5, "office": 0.3, "residential": 0.2},
    "semilla_trafico": 42,

    # ---------- BBU pools ----------
    "n_pools": 19,                            # numero de BBU pools candidatos (tesis: 19)
    "metodo_pools": "k-means",                # "k-means" (distancia media) o "k-center" (distancia maxima)
    "pools_desactivados": [],                 # IDs de pools a apagar para generar el mapa reducido; [] = no

    # ---------- Retardo ----------
    "retardo_rt_max_us": 150.0,               # retardo maximo de ida y vuelta (us); tesis: 150 us = 15 km
    "us_por_km": 5.0,                         # retardo de la fibra por km, en un sentido (v = 2e8 m/s)

    # ---------- Analisis adicionales ----------
    "retardos_barrido_us": [20, 30, 40, 50, 60, 80, 100, 150, 200],
    "retardo_poda_estricto_us": 60,           # 2o retardo para el analisis de que pools sobran
    "barrido_sinteticos": [0, 50, 100, 300, 600],  # prueba de robustez: cuantas sinteticas probar

    # ---------- Primera entrega (cran_escenario.py) ----------
    "k_lista": [2, 3, 4, 5, 6, 8, 10, 15, 20],   # valores de K a comparar
    "k_elegido": 10,                          # K usado para el mapa y la asignacion final

    # ---------- Fase 2: load balancing (fase2_load_balancing.py) ----------
    "f2_retardo_rt_max_us": 60.0,             # retardo maximo de ida y vuelta usado para las listas de pools posibles
    "f2_pools": {"few": 10, "normal": 14, "many": 19},   # numero de pools de los 3 casos a comparar
    "f2_caso_principal": "normal",            # caso con el que se dibujan las figuras detalladas
    "f2_hora": "auto",                        # hora del balanceo (0-23); "auto" = hora de maxima carga total
    "f2_barrido_retardos_us": [40, 60, 80, 100, 150],   # retardos para ver cuanto ayuda un limite mas holgado
    "f2_horas_figura": [8, 15, 21],           # horas para la carga por pool antes del algoritmo (fig 1a)
    "f2_perfiles_csv": None,                  # CSV propio con columnas hour,residential,office,mixed (MBph); None = tesis
    "f2_variabilidad_carga": 0.0,             # desv. tipica (lognormal) de un factor aleatorio por RRH; 0 = sin variacion
    "f2_capacidad_tarjeta_mbph": 5000.0,      # SUPUESTO: capacidad de una tarjeta de procesado (para contar tarjetas)
    "f2_umbral_sobrecarga": 0.25,             # un pool esta sobrecargado si supera la media en mas de este %
    "f2_max_bbus_extra": 2,                   # maximo de BBU nuevas que se prueban en el refuerzo
    "f2_umbral_holgura": 0.0,                 # una BBU tiene holgura (puede recibir RRH) si su carga < media*(1-este %); 0 = por debajo de la media
    "f2_mejora_minima": 0.15,                 # mejora relativa del desequilibrio por debajo de la cual se prueban alternativas
    "f2_candidatos_max": 150,                 # sitios de RRH real que se prueban como BBU nueva (los de mas carga a aliviar)
    "f2_dist_min_bbu_km": 2.0,                # separacion minima entre dos BBU (km); evita apilar BBU nuevas junto a otras
    "f2_peso_retardo": 0.2,                   # peso de la distancia media (retardo) frente al equilibrio al elegir el sitio

    # ---------- General ----------
    "semilla": 42,
    "dir_figuras": "Figuras",                 # carpeta (junto al script) donde se guardan las figuras
}


def cargar_parametros(argv=None):
    """Devuelve los parametros como objeto (p.ej. cfg.n_pools), aplicando los --nombre=valor."""
    valores = dict(PARAMETROS)
    for arg in (sys.argv[1:] if argv is None else argv):
        if not arg.startswith("--") or "=" not in arg:
            continue
        clave, _, texto = arg[2:].partition("=")
        if clave not in valores:
            raise SystemExit(f"Parametro desconocido: --{clave}. Disponibles: {', '.join(valores)}")
        try:
            valores[clave] = ast.literal_eval(texto)
        except (ValueError, SyntaxError):
            valores[clave] = texto           # texto sin comillas (p. ej. k-means)
    p = valores["porcentajes_trafico"]
    if p and abs(sum(p.values()) - 1) > 1e-6:
        raise SystemExit(f"porcentajes_trafico debe sumar 1 (suma {sum(p.values()):.3f}).")
    return SimpleNamespace(**valores)