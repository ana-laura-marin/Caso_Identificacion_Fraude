import pandas as pd
from scipy.stats import binom

pd.set_option("display.max_columns", None)  # mostrar todas las columnas al imprimir

# 1. Cargar datos y quedarnos con el periodo de etiqueta madura
reclamos = pd.read_csv("reclamos.csv", parse_dates=["fecha_reporte"])

# Un fraude tarda ~81 días en cerrarse: los reclamos de abr-2026 en adelante
# todavía no tienen la etiqueta definitiva, así que los dejamos fuera.
CORTE = "2026-04-01"
df = reclamos[reclamos["fecha_reporte"] < CORTE].copy()

# Columna auxiliar: monto reclamado solo si el reclamo cerró como fraude
df["monto_fraude"] = df["monto_reclamado"] * df["marca_fraude"]

# 2. Tamaño del problema
meses = df["fecha_reporte"].dt.to_period("M").nunique()
kpis = pd.Series({
    "reclamos": len(df),
    "meses": meses,
    "reclamado_millones": df["monto_reclamado"].sum() / 1e6,
    "pagado_millones": df["monto_pagado"].sum() / 1e6,
    "fraude_detectado_millones": df["monto_fraude"].sum() / 1e6,
    "tasa_fraude_pct": df["marca_fraude"].mean() * 100,
    "expedientes_por_analista_mes": df["investigacion_abierta"].sum() / meses / 4,
    "acierto_investigacion_pct": df["marca_fraude"].sum() / df["investigacion_abierta"].sum() * 100,
})
print(kpis.round(1))

# 3. Una función para resumir cualquier dimensión
def resumen(datos, columna):
    """Reclamos, tasa de fraude y colones de fraude por categoría."""
    tabla = datos.groupby(columna, observed=True).agg(
        reclamos=("id_reclamo", "size"),
        tasa_fraude=("marca_fraude", "mean"),
        fraude_millones=("monto_fraude", "sum"),
    )
    tabla["fraude_millones"] /= 1e6
    tabla["pct_reclamos"] = tabla["reclamos"] / tabla["reclamos"].sum()
    tabla["pct_fraude"] = tabla["fraude_millones"] / tabla["fraude_millones"].sum()
    return tabla.sort_values("fraude_millones", ascending=False).round(3)

print(resumen(df, "provincia"))
print(resumen(df, "tipo_siniestro"))

df["edad_tramo"] = pd.cut(df["edad_vehiculo"], bins=[-1, 4, 8, 12, 25],
                          labels=["0-4", "5-8", "9-12", "13+"])
edad = resumen(df, "edad_tramo")
pagado = df.groupby("edad_tramo", observed=True)[["monto_pagado", "monto_reclamado"]].sum()
edad["pagado_sobre_reclamado"] = (pagado["monto_pagado"] / pagado["monto_reclamado"]).round(3)
print(edad.sort_index())

# 4. Talleres: ¿el volumen se asocia con más fraude?
talleres = df.groupby("taller_id").agg(
    reclamos=("id_reclamo", "size"),
    fraudes=("marca_fraude", "sum"),
)
talleres["tasa_fraude"] = talleres["fraudes"] / talleres["reclamos"]

talleres["tramo"] = pd.cut(talleres["reclamos"], bins=[0, 30, 100, 500, 5000],
                           labels=["<30", "30-100", "100-500", "500+"])
por_tramo = talleres.groupby("tramo", observed=True)[["reclamos", "fraudes"]].sum()
por_tramo["talleres"] = talleres.groupby("tramo", observed=True).size()
por_tramo["tasa_fraude"] = (por_tramo["fraudes"] / por_tramo["reclamos"]).round(3)
print(por_tramo)

# 5. Talleres para revisar: ¿cuán raro sería ver tantos fraudes si el taller
# tuviera la tasa promedio? (prueba binomial de una cola)
tasa_base = df["marca_fraude"].mean()
talleres["prob_azar"] = binom.sf(talleres["fraudes"] - 1, talleres["reclamos"], tasa_base)

a_revisar = (talleres[talleres["reclamos"] >= 20]
             .query("prob_azar < 0.05")
             .sort_values("prob_azar"))
print(a_revisar.round(4))

# 6. Gráficos (se guardan en graficos/ porque el script corre sin pantalla)
import os
import matplotlib.pyplot as plt

os.makedirs("graficos", exist_ok=True)

# 6.1 Tasa de fraude por provincia
prov = resumen(df, "provincia")
prov["tasa_fraude"].sort_values().plot(kind="barh", figsize=(8, 5), title="Tasa de fraude por provincia")
plt.xlabel("Tasa de fraude")
plt.tight_layout()
plt.savefig("graficos/1_fraude_provincia.png", dpi=120)
plt.show()
plt.close()

# 6.2 Tasa de fraude por edad del vehículo
edad.sort_index()["tasa_fraude"].plot(kind="bar", figsize=(6, 4), title="Tasa de fraude por edad del vehículo")
plt.ylabel("Tasa de fraude")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficos/2_fraude_edad_vehiculo.png", dpi=120)
plt.show()
plt.close()

# 6.3 Tasa de fraude según volumen del taller
por_tramo["tasa_fraude"].plot(kind="bar", figsize=(6, 4), title="Tasa de fraude según volumen del taller")
plt.ylabel("Tasa de fraude")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficos/3_fraude_volumen_taller.png", dpi=120)
plt.show()
plt.close()

# 6.4 Fraude detectado por mes
mensual = df.groupby(df["fecha_reporte"].dt.to_period("M"))["monto_fraude"].sum() / 1e6
mensual.plot(figsize=(9, 4), marker="o", title="Fraude detectado por mes (millones)")
plt.ylabel("Millones")
plt.tight_layout()
plt.savefig("graficos/4_fraude_mensual.png", dpi=120)
plt.show()
plt.close()
