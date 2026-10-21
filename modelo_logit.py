import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

pd.set_option("display.max_columns", None)

# 1. Datos: periodo con etiqueta madura y variables construidas
reclamos = pd.read_csv("reclamos.csv", parse_dates=["fecha_reporte"])
df = reclamos[reclamos["fecha_reporte"] < "2026-04-01"].copy()

df["robo_total"] = (df["tipo_siniestro"] == "robo_total").astype(int)
df["vidrios"] = (df["tipo_siniestro"] == "vidrios").astype(int)
df["canal_taller"] = (df["canal_reporte"] == "taller").astype(int)
df["ratio_monto"] = df["monto_reclamado"] / df["suma_asegurada"]
df["log_monto"] = np.log(df["monto_reclamado"])
df["log_antig"] = np.log(df["antiguedad_poliza_dias"])
df["sin_fecha_siniestro"] = df["dias_reporte"].isna().astype(int)
df["dias_reporte_imp"] = df["dias_reporte"].fillna(0)
df["noche"] = ((df["hora_siniestro"] >= 22) | (df["hora_siniestro"] <= 5)).astype(int)
df["previos"] = df["num_reclamos_previos_24m"].clip(upper=3)

# 2. Partición temporal: se selecciona con train; test solo se mira al final
train = df[df["fecha_reporte"] < "2025-10-01"]
test = df[df["fecha_reporte"] >= "2025-10-01"]

# 3. Candidatas por bloque (una categórica = un bloque con varios coeficientes)
bloques = {
    "tipo": "C(tipo_siniestro, Treatment('colision'))",
    "canal": "C(canal_reporte, Treatment('app'))",
    "provincia": "C(provincia, Treatment('San Jose'))",
    "ratio": "ratio_monto",
    "monto": "log_monto",
    "antiguedad_poliza": "log_antig",
    "dias_reporte": "dias_reporte_imp + sin_fecha_siniestro",
    "noche": "noche",
    "edad_vehiculo": "edad_vehiculo",
    "previos": "previos",
    "parte_policial": "hubo_parte_policial",
    "testigos": "hubo_testigos",
}

def ajustar(formula_derecha, datos=train):
    return smf.logit("marca_fraude ~ " + formula_derecha, datos).fit(disp=0)

# 4. Test de razón de verosimilitud: sacar un bloque a la vez del modelo completo
completo = ajustar(" + ".join(bloques.values()))
filas = []
for nombre in bloques:
    sin = ajustar(" + ".join(v for k, v in bloques.items() if k != nombre))
    lr = 2 * (completo.llf - sin.llf)
    gl = completo.df_model - sin.df_model
    filas.append({"bloque": nombre, "gl": int(gl), "LR": lr, "p": chi2.sf(lr, gl),
                  "dAIC_al_sacar": sin.aic - completo.aic,
                  "dBIC_al_sacar": sin.bic - completo.bic})
print(pd.DataFrame(filas).sort_values("p").round(3))

# 5. Comparar especificaciones candidatas
modelos = {
    "M0 completo": " + ".join(bloques.values()),
    "M1 sin provincia/edad/monto": ("C(tipo_siniestro, Treatment('colision')) + C(canal_reporte, Treatment('app'))"
                                    " + ratio_monto + log_antig + dias_reporte_imp + sin_fecha_siniestro"
                                    " + noche + previos + hubo_parte_policial + hubo_testigos"),
    "M2 categorías agrupadas": ("robo_total + vidrios + canal_taller + ratio_monto + log_antig"
                                " + dias_reporte_imp + sin_fecha_siniestro + noche + previos"
                                " + hubo_parte_policial + hubo_testigos"),
    "M3 = M2 sin ratio": ("robo_total + vidrios + canal_taller + log_antig + dias_reporte_imp"
                          " + sin_fecha_siniestro + noche + previos + hubo_parte_policial + hubo_testigos"),
}
ajustes = {k: ajustar(f) for k, f in modelos.items()}
print(pd.DataFrame({k: {"parametros": int(m.df_model) + 1, "AIC": m.aic, "BIC": m.bic,
                        "AUC_test": roc_auc_score(test["marca_fraude"], m.predict(test))}
                    for k, m in ajustes.items()}).T.round(3))

def lr_anidados(grande, chico):
    a, b = ajustes[grande], ajustes[chico]
    lr = 2 * (a.llf - b.llf); gl = a.df_model - b.df_model
    print(f"{grande} vs {chico}: LR={lr:.2f}, gl={int(gl)}, p={chi2.sf(lr, gl):.3f}")

lr_anidados("M0 completo", "M1 sin provincia/edad/monto")
lr_anidados("M1 sin provincia/edad/monto", "M2 categorías agrupadas")
lr_anidados("M2 categorías agrupadas", "M3 = M2 sin ratio")

# 6. Odds ratios con intervalo de confianza al 95 %
final = ajustes["M3 = M2 sin ratio"]
ic = np.exp(final.conf_int())
print(pd.DataFrame({"OR": np.exp(final.params), "IC95_inf": ic[0],
                    "IC95_sup": ic[1], "p": final.pvalues}).round(3))
