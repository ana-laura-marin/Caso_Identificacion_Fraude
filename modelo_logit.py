import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

# 1. Datos: periodo con etiqueta madura
# La extracción llega a sep-2026 y el 100 % de los fraudes cierra en <= 232 días,
# así que los reclamos reportados antes de abr-2026 ya tienen la etiqueta definitiva.
reclamos = pd.read_csv("reclamos.csv", parse_dates=["fecha_reporte"])
df = reclamos[reclamos["fecha_reporte"] < "2026-04-01"].copy()
assert df["dias_hasta_cierre"].notna().all(), "hay reclamos abiertos en el periodo modelado"

# 2. Variables: solo información disponible el día que entra el reclamo.
# Fuera por fuga de información: investigacion_abierta, dias_hasta_cierre, monto_pagado.
df["robo_total"] = (df["tipo_siniestro"] == "robo_total").astype(int)
df["vidrios"] = (df["tipo_siniestro"] == "vidrios").astype(int)
df["canal_taller"] = (df["canal_reporte"] == "taller").astype(int)
df["ratio_monto"] = df["monto_reclamado"] / df["suma_asegurada"]
df["log_monto"] = np.log(df["monto_reclamado"])            # mínimo 3.000, sin ceros
df["log_antig"] = np.log(df["antiguedad_poliza_dias"])     # mínimo 1 día, sin ceros
# fecha_siniestro falta en el 100 % de los reclamos reportados entre ago y oct-2025 y en
# ningún otro mes: es una falla de extracción, no un rasgo del reclamo. Un indicador de
# "sin fecha" sería una dummy de calendario (y es colineal en train, donde no hay faltantes).
# Se imputa con la mediana de los meses sin falla, conocida antes del periodo de validación.
faltante = df.groupby(df["fecha_reporte"].dt.to_period("M"))["dias_reporte"].apply(lambda s: s.isna().mean())
print("Meses con fecha_siniestro faltante:", {str(k): round(v, 2) for k, v in faltante[faltante > 0].items()})
MEDIANA_DIAS = df.loc[df["fecha_reporte"] < "2025-04-01", "dias_reporte"].median()
df["dias_reporte_imp"] = df["dias_reporte"].fillna(MEDIANA_DIAS)
df["noche"] = ((df["hora_siniestro"] >= 22) | (df["hora_siniestro"] <= 5)).astype(int)
df["previos"] = df["num_reclamos_previos_24m"].clip(upper=3)

# 3. Partición temporal en tres tramos
#   train: estimar y elegir variables | valid: comparar modelos | test: se mira una sola vez
train = df[df["fecha_reporte"] < "2025-04-01"].copy()
valid = df[(df["fecha_reporte"] >= "2025-04-01") & (df["fecha_reporte"] < "2025-10-01")].copy()
test = df[df["fecha_reporte"] >= "2025-10-01"].copy()
for nombre, parte in [("train", train), ("valid", valid), ("test", test)]:
    print(f"{nombre}: {len(parte):,} reclamos, {parte['marca_fraude'].sum()} fraudes")

# 4. Historial del taller sin fuga: solo cuentan los reclamos del mismo taller que ya
# estaban CERRADOS el día del reporte. Se suaviza hacia la tasa base de train para que un
# taller con 3 reclamos y 1 fraude no aparezca con 33 %. Robo total no pasa por taller:
# queda en la tasa base y el efecto lo captura la dummy robo_total.
TASA_BASE = train["marca_fraude"].mean()
K_SUAVIZADO = 50

def historial_taller(datos, k=K_SUAVIZADO, base=TASA_BASE):
    datos = datos.copy()
    datos["fecha_cierre"] = datos["fecha_reporte"] + pd.to_timedelta(datos["dias_hasta_cierre"], unit="D")
    tasa = pd.Series(base, index=datos.index)
    for _, g in datos.dropna(subset=["taller_id"]).groupby("taller_id"):
        cierres = g.sort_values("fecha_cierre")
        fechas_cierre = cierres["fecha_cierre"].to_numpy()
        fraudes_acum = np.concatenate([[0], cierres["marca_fraude"].cumsum().to_numpy()])
        n_previos = np.searchsorted(fechas_cierre, g["fecha_reporte"].to_numpy(), side="left")
        tasa.loc[g.index] = (fraudes_acum[n_previos] + k * base) / (n_previos + k)
    return tasa * 100   # en puntos porcentuales: el OR se lee "por cada punto de tasa del taller"

df["tasa_taller"] = historial_taller(df)
for parte in (train, valid, test):
    parte["tasa_taller"] = df.loc[parte.index, "tasa_taller"]

# 5. Ajuste con chequeo de convergencia
def ajustar(formula_derecha, datos):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.logit("marca_fraude ~ " + formula_derecha, datos).fit(disp=0, maxiter=200)
    if not m.mle_retvals["converged"]:
        raise RuntimeError(f"no convergió: {formula_derecha}")
    return m

# 6. Candidatas por bloque (una categórica = un bloque con varios coeficientes)
bloques = {
    "tipo": "C(tipo_siniestro, Treatment('colision'))",
    "canal": "C(canal_reporte, Treatment('app'))",
    "provincia": "C(provincia, Treatment('San Jose'))",
    "ratio": "ratio_monto",
    "monto": "log_monto",
    "antiguedad_poliza": "log_antig",
    "dias_reporte": "dias_reporte_imp",
    "noche": "noche",
    "edad_vehiculo": "edad_vehiculo",
    "previos": "previos",
    "parte_policial": "hubo_parte_policial",
    "testigos": "hubo_testigos",
    "taller": "tasa_taller",
}

# 6a. Colinealidad entre las numéricas candidatas (VIF > 5 sería preocupante)
numericas = ["ratio_monto", "log_monto", "log_antig", "dias_reporte_imp", "edad_vehiculo",
             "previos", "tasa_taller"]
X = train[numericas].assign(const=1.0)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(len(numericas))],
                index=numericas, name="VIF")
print("\nVIF (train)\n", vif.round(2))

# 6b. Razón de verosimilitud: sacar un bloque a la vez del modelo completo (solo train)
completo = ajustar(" + ".join(bloques.values()), train)
filas = []
for nombre in bloques:
    sin = ajustar(" + ".join(v for k, v in bloques.items() if k != nombre), train)
    lr = 2 * (completo.llf - sin.llf)
    gl = completo.df_model - sin.df_model
    filas.append({"bloque": nombre, "gl": int(gl), "LR": lr, "p": chi2.sf(lr, gl),
                  "dAIC_al_sacar": sin.aic - completo.aic,
                  "dBIC_al_sacar": sin.bic - completo.bic})
print("\nAporte de cada bloque (train)\n", pd.DataFrame(filas).sort_values("p").round(3))

# 7. Especificaciones candidatas
base_m2 = ("robo_total + vidrios + canal_taller + log_antig + dias_reporte_imp"
           " + noche + previos + hubo_parte_policial + hubo_testigos")
modelos = {
    "M0 completo": " + ".join(bloques.values()),
    "M1 sin provincia/edad/monto": ("C(tipo_siniestro, Treatment('colision')) + C(canal_reporte, Treatment('app'))"
                                    " + ratio_monto + log_antig + dias_reporte_imp"
                                    " + noche + previos + hubo_parte_policial + hubo_testigos + tasa_taller"),
    "M2 categorías agrupadas": base_m2 + " + ratio_monto + tasa_taller",
    "M3 = M2 sin ratio": base_m2 + " + tasa_taller",
    "M4 = M3 sin taller": base_m2,
}

# Métrica de negocio: de los K reclamos con mayor puntaje cada mes, cuántos son fraude.
# K = investigaciones promedio por mes en train (capacidad real de los 4 analistas).
CAPACIDAD = round(train["investigacion_abierta"].sum() / train["fecha_reporte"].dt.to_period("M").nunique())

def acierto_capacidad(datos, puntaje, k=CAPACIDAD):
    d = datos.assign(puntaje=puntaje, mes=datos["fecha_reporte"].dt.to_period("M"))
    top = d.sort_values("puntaje", ascending=False).groupby("mes").head(k)
    return pd.Series({"acierto_topK": top["marca_fraude"].mean(),
                      "fraudes_capturados_pct": top["marca_fraude"].sum() / d["marca_fraude"].sum(),
                      "colones_capturados_pct": (top["monto_reclamado"] * top["marca_fraude"]).sum()
                                                / (d["monto_reclamado"] * d["marca_fraude"]).sum()})

ajustes = {k: ajustar(f, train) for k, f in modelos.items()}
comparacion = pd.DataFrame({
    k: {"parametros": int(m.df_model) + 1, "BIC_train": m.bic,
        "AUC_valid": roc_auc_score(valid["marca_fraude"], m.predict(valid)),
        "PR_AUC_valid": average_precision_score(valid["marca_fraude"], m.predict(valid)),
        **acierto_capacidad(valid, m.predict(valid)).to_dict()}
    for k, m in ajustes.items()}).T
print(f"\nComparación (capacidad = {CAPACIDAD} reclamos/mes; test todavía sin mirar)\n", comparacion.round(3))

def lr_anidados(grande, chico):
    a, b = ajustes[grande], ajustes[chico]
    lr = 2 * (a.llf - b.llf); gl = a.df_model - b.df_model
    print(f"{grande} vs {chico}: LR={lr:.2f}, gl={int(gl)}, p={chi2.sf(lr, gl):.3f}")

print()
lr_anidados("M0 completo", "M1 sin provincia/edad/monto")
lr_anidados("M1 sin provincia/edad/monto", "M2 categorías agrupadas")
lr_anidados("M2 categorías agrupadas", "M3 = M2 sin ratio")
lr_anidados("M3 = M2 sin ratio", "M4 = M3 sin taller")

# 8. Elección: el de menor BIC en train (parsimonia), confirmado en valid
elegido = comparacion["BIC_train"].astype(float).idxmin()
print(f"\nModelo elegido: {elegido}")

# 9. Reestimar con train + valid y evaluar UNA vez en test
train_valid = pd.concat([train, valid])
final = ajustar(modelos[elegido], train_valid)
p_test = final.predict(test)
print(f"Fraudes por parámetro (EPV) en el ajuste final: "
      f"{train_valid['marca_fraude'].sum() / (final.df_model + 1):.1f}")

meses_test = test["fecha_reporte"].dt.to_period("M")
investigados = test[test["investigacion_abierta"] == 1]
actual = pd.Series({
    "acierto_topK": investigados["marca_fraude"].mean(),
    "fraudes_capturados_pct": 1.0,   # por construcción: solo hay fraude donde se investigó
    "colones_capturados_pct": 1.0,
})
print("\nTest (oct-2025 a mar-2026)")
print(f"AUC = {roc_auc_score(test['marca_fraude'], p_test):.3f} | "
      f"PR-AUC = {average_precision_score(test['marca_fraude'], p_test):.3f} | "
      f"tasa base = {test['marca_fraude'].mean():.3f}")
print(pd.DataFrame({"modelo_topK": acierto_capacidad(test, p_test),
                    "analistas_hoy": actual}).round(3))
print(f"Analistas hoy investigan {len(investigados) / meses_test.nunique():.0f} reclamos/mes en test.")

# La comparación anterior castiga al modelo: un reclamo que nadie investigó tiene
# marca_fraude = 0 aunque haya sido fraude, así que el acierto del modelo es un PISO.
# Dos lecturas justas, usando solo reclamos con etiqueta real (los investigados):
d_test = test.assign(p=p_test, mes=meses_test)
top = d_test.sort_values("p", ascending=False).groupby("mes").head(CAPACIDAD)
top_inv = top[top["investigacion_abierta"] == 1]
print(f"Del top-{CAPACIDAD} mensual del modelo, {len(top_inv) / len(top):.0%} ya lo investigaban los analistas; "
      f"en esos, el acierto es {top_inv['marca_fraude'].mean():.1%} (vs {investigados['marca_fraude'].mean():.1%} "
      f"de todo lo investigado).")
print(f"AUC dentro de lo investigado: {roc_auc_score(investigados['marca_fraude'], p_test[investigados.index]):.3f} "
      f"(> 0,5 = el modelo ordena mejor que el azar incluso entre lo que los analistas ya eligieron)")
print(f"Reclamos del top-{CAPACIDAD} que nadie investigó: {len(top) - len(top_inv)} en 6 meses "
      f"(su etiqueta es desconocida, no 'limpia').")

# Calibración: probabilidad media predicha vs. tasa observada por decil de puntaje
cal = (test.assign(p=p_test, decil=pd.qcut(p_test, 10, labels=False, duplicates="drop") + 1)
           .groupby("decil").agg(reclamos=("p", "size"), p_media=("p", "mean"),
                                 tasa_obs=("marca_fraude", "mean")))
print("\nCalibración en test\n", cal.round(3))

# 10. Odds ratios con IC 95 % (ajuste final, train + valid).
# Ojo: los p-valores después de seleccionar variables son optimistas; leer como descriptivos.
ic = np.exp(final.conf_int())
print("\nOdds ratios del modelo final\n",
      pd.DataFrame({"OR": np.exp(final.params), "IC95_inf": ic[0],
                    "IC95_sup": ic[1], "p": final.pvalues}).round(3))
