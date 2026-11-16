# %% [markdown]
# # Análisis exploratorio de Predict Future Sales
#
# **Product Development · Proyecto 1 · Universidad Galileo**
#
# **Integrantes:** Luis Zarate y Samuel Gómez.
#
# Se estudia el historial diario para preparar el pronóstico de la cantidad mensual
# vendida por tienda y producto en noviembre de 2015. Este notebook desarrolla los
# diez apartados del enunciado y conserva las decisiones y limitaciones del análisis.
# Los datos originales permanecen en `data/raw/`. Las interpretaciones que contienen
# cifras se calculan al ejecutar las celdas; no se entrenan modelos en esta entrega.
#
# ## 1. Estructura del proyecto y reproducibilidad
# La estructura se generó con Cookiecutter Data Science v2; su origen y revisión
# están registrados en `docs/plantilla.json`. El repositorio excluye los CSV crudos.
# La publicación en GitHub y las capturas reales del pull se completan por separado.

# %%
from pathlib import Path
import hashlib
import json
import re
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "data/raw/sales_train.csv").is_file())
RAW = ROOT / "data/raw"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 18)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
sns.set_theme(style="whitegrid", palette="colorblind", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.family": "DejaVu Sans"})

def table(df, name, n=15):
    df.to_csv(TABLES / f"{name}.csv", index=True, encoding="utf-8-sig")
    display(df.head(n))

def figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.show()
    plt.close()

def note(text):
    display(Markdown(text))

print("Python:", sys.version.split()[0], "| pandas:", pd.__version__)
print("Proyecto:", ROOT.name)

# %% [markdown]
# ## 2. Comprensión inicial del dataset
#
# Se cargan los cinco archivos del problema y `sample_submission.csv`, que solo
# ilustra el formato final. El 0,5 de ese archivo no es una etiqueta real de ventas.
# Los identificadores son categorías aunque se almacenen como números.

# %%
frames = {p.stem: pd.read_csv(p) for p in sorted(RAW.glob("*.csv"))}
required = {"sales_train", "items", "item_categories", "shops", "test", "sample_submission"}
assert required.issubset(frames), f"Faltan archivos: {required - set(frames)}"
inventory = pd.DataFrame([
    {"archivo": name + ".csv", "filas": len(df), "columnas": df.shape[1],
     "MB": (RAW / f"{name}.csv").stat().st_size / 1e6}
    for name, df in frames.items()
]).set_index("archivo")
table(inventory, "inventario_archivos")
for name, df in frames.items():
    note(f"**Primeras filas de `{name}.csv`**")
    display(df.head(3))

# %% [markdown]
# | Campo | Significado y papel |
# | --- | --- |
# | date | Fecha del registro diario, originalmente dd.mm.aaaa. |
# | date_block_num | Mes consecutivo: 0 corresponde a enero de 2013. |
# | shop_id / shop_name | Identificador y nombre de la tienda. |
# | item_id / item_name | Identificador y nombre del producto. |
# | item_category_id / item_category_name | Identificador y nombre de su categoría. |
# | item_price | Precio unitario en la unidad monetaria del conjunto. |
# | item_cnt_day | Cantidad diaria; los negativos representan devoluciones. |
# | ID | Identificador del par tienda-producto solicitado en test. |
# | item_cnt_month | Objetivo final: suma mensual de item_cnt_day por tienda-producto. |
# | ingresos | Variable derivada: item_price × item_cnt_day. |
#
# La tabla no acredita un ticket individual: una fecha, tienda y producto pueden
# aparecer varias veces, por ejemplo, con distinto precio. No se eliminarán esos
# registros solo por compartir esa combinación.

# %%
raw = frames["sales_train"]
items, categories, shops, test = [frames[n] for n in ("items", "item_categories", "shops", "test")]
raw_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob("*.csv"))}
original_types = raw.dtypes.astype(str).rename("tipo_original").to_frame()
sales = raw.copy()
sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y", errors="coerce")
original_types["tipo_analisis"] = sales[original_types.index].dtypes.astype(str)
table(original_types, "tipos_ventas")
first, last = sales.date.min(), sales.date.max()
MONTHS = pd.period_range(first, last, freq="M")
DAYS = pd.date_range(first, last, freq="D")
note(f"El histórico va de **{first:%d/%m/%Y} a {last:%d/%m/%Y}**, con **{sales.date_block_num.nunique()} meses**. "
     "La predicción de noviembre queda fuera de este intervalo.")

# %% [markdown]
# ## 3. Validación de estructura y frecuencia
#
# Se verifican fechas, continuidad, correspondencia entre fecha y número de mes,
# claves de los catálogos y estructura de test. Una ausencia de registro significa
# ausencia de ventas observadas; por sí sola no prueba que una tienda estuviera abierta.

# %%
expected_blocks = (sales.date.dt.year - 2013) * 12 + sales.date.dt.month - 1
missing_days = DAYS.difference(pd.DatetimeIndex(sales.date.unique()))
checks = pd.Series({
    "fechas_no_interpretables": sales.date.isna().sum(),
    "meses_inconsistentes": expected_blocks.ne(sales.date_block_num).sum(),
    "dias_sin_registros_globales": len(missing_days),
    "productos_fuera_catalogo": (~sales.item_id.isin(items.item_id)).sum(),
    "tiendas_fuera_catalogo": (~sales.shop_id.isin(shops.shop_id)).sum(),
    "categorias_fuera_catalogo": (~items.item_category_id.isin(categories.item_category_id)).sum(),
    "IDs_test_duplicados": test.ID.duplicated().sum(),
    "pares_test_duplicados": test.duplicated(["shop_id", "item_id"]).sum(),
    "cantidades_no_enteras": (~np.isclose(sales.item_cnt_day, sales.item_cnt_day.round())).sum(),
}, name="conteo").to_frame()
table(checks, "validacion_estructura")
assert checks.loc[["fechas_no_interpretables", "meses_inconsistentes", "productos_fuera_catalogo", "tiendas_fuera_catalogo", "categorias_fuera_catalogo", "IDs_test_duplicados", "pares_test_duplicados"], "conteo"].eq(0).all()
assert items.item_id.is_unique and categories.item_category_id.is_unique and shops.shop_id.is_unique
assert test.ID.equals(frames["sample_submission"].ID)
note(f"Se encontraron **{len(missing_days)} días sin registros globales** y **{int(checks.loc['meses_inconsistentes','conteo'])} inconsistencias de mes**. "
     "Se adopta la frecuencia **mensual** para el objetivo; la diaria se conserva para explorar ciclos semanales. "
     "Los 34 meses limitan la evidencia sobre ciclos anuales.")

# %% [markdown]
# ## 4. Faltantes, duplicados y valores no válidos
#
# Se separan tres situaciones: nulos, precios no positivos y devoluciones. Las
# devoluciones son observaciones válidas para las cantidades netas. Los precios no
# positivos se excluyen solo de los cálculos monetarios; su cantidad se conserva.
# Para las seis columnas originales, una fila idéntica se trata como duplicada en
# la vista de análisis. No existe un ID de ticket que permita confirmar su origen,
# por lo que también se mide el efecto de esa decisión.

# %%
nulls = pd.concat([
    pd.DataFrame({"archivo": name, "columna": df.columns,
                  "nulos": df.isna().sum().values, "porcentaje": df.isna().mean().values * 100})
    for name, df in frames.items()
], ignore_index=True)
table(nulls, "valores_faltantes", n=40)
duplicate_mask = sales.duplicated(subset=list(raw.columns), keep="first")
duplicates_n = int(duplicate_mask.sum())
duplicate_units = float(sales.loc[duplicate_mask, "item_cnt_day"].sum())
table(sales.loc[duplicate_mask, list(raw.columns)], "duplicados_exactos")
sales = sales.loc[~duplicate_mask].copy()
sales["precio_valido"] = sales.item_price.gt(0)
sales["ingresos"] = (sales.item_price * sales.item_cnt_day).where(sales.precio_valido)
sales["unidades_positivas"] = sales.item_cnt_day.clip(lower=0)
sales["unidades_devueltas"] = -sales.item_cnt_day.clip(upper=0)
quality = pd.Series({"filas_originales": len(raw), "duplicados_excluidos": duplicates_n,
    "unidades_en_duplicados": duplicate_units, "filas_analizadas": len(sales),
    "precios_no_positivos": (~sales.precio_valido).sum(),
    "filas_devoluciones": sales.item_cnt_day.lt(0).sum(),
    "unidades_devueltas": sales.unidades_devueltas.sum()}, name="valor").to_frame()
table(quality, "calidad_ventas")
table(sales.loc[~sales.precio_valido, list(raw.columns)], "precios_invalidos")
note(f"No hay nulos en ventas. Se excluyen **{duplicates_n} duplicados exactos**, equivalentes a "
     f"**{duplicate_units:,.0f} unidades** ({duplicate_units/raw.item_cnt_day.sum()*100:.5f}% del total original). "
     f"Se conservan **{int(sales.item_cnt_day.lt(0).sum()):,} registros de devoluciones**. "
     "El precio no positivo no se imputa: se mantiene señalado para revisión, sin calcular ingreso en esa fila.")

# %%
def normal_shop_name(name):
    value = name.casefold().replace("!", "").replace("?", "").replace("²", "")
    value = re.sub(r"\s+фран\s*$", "", value)
    return re.sub(r"\s+", " ", value).strip()

shops_review = shops.assign(nombre_normalizado=shops.shop_name.map(normal_shop_name))
exact_shop_duplicates = shops.loc[shops.shop_name.duplicated(keep=False)]
candidates = shops_review.loc[shops_review.nombre_normalizado.duplicated(keep=False)].sort_values("nombre_normalizado")
table(exact_shop_duplicates, "tiendas_nombre_exacto_repetido")
table(candidates, "tiendas_posibles_alias")
note(f"Hay **{len(exact_shop_duplicates)} filas con nombre exactamente repetido**, pero la normalización detecta "
     f"**{candidates.nombre_normalizado.nunique()} grupos candidatos a alias**. Se conservan los IDs originales: "
     "el parecido del nombre no confirma la identidad física. Unificar tiendas requiere validación y cambiaría el historial y la cobertura de test.")

# %% [markdown]
# ## 5. Análisis descriptivo general
#
# Se distinguen unidades positivas, devoluciones y cantidades netas. Los ingresos
# son netos de devoluciones en las filas con precio válido. No se interpreta una
# asociación observada como un efecto causal.

# %%
sales = (sales.merge(items[["item_id", "item_category_id"]], on="item_id", how="left", validate="many_to_one")
         .merge(categories, on="item_category_id", how="left", validate="many_to_one")
         .merge(shops, on="shop_id", how="left", validate="many_to_one"))
assert len(sales) == len(raw) - duplicates_n
sales["mes"] = sales.date.dt.to_period("M")
sales["anio"] = sales.date.dt.year
sales["mes_calendario"] = sales.date.dt.month
descriptive = sales[["item_price", "item_cnt_day", "ingresos"]].describe(percentiles=[.01, .25, .5, .75, .99, .999]).T
descriptive.loc["item_price"] = sales.loc[sales.precio_valido, "item_price"].describe(percentiles=[.01, .25, .5, .75, .99, .999])
table(descriptive, "estadisticas_descriptivas")
AGG = {"item_cnt_day": "sum", "unidades_positivas": "sum", "unidades_devueltas": "sum", "ingresos": "sum"}
by_shop = sales.groupby(["shop_id", "shop_name"], observed=True).agg(AGG).sort_values("item_cnt_day", ascending=False)
by_item = sales.groupby("item_id", observed=True).agg(AGG).sort_values("item_cnt_day", ascending=False)
by_category = sales.groupby(["item_category_id", "item_category_name"], observed=True).agg(AGG).sort_values("item_cnt_day", ascending=False)
daily = sales.groupby("date").agg(AGG).reindex(DAYS, fill_value=0)
monthly = sales.groupby("mes").agg(AGG).reindex(MONTHS, fill_value=0)
weekly = daily.resample("W-SUN").sum()
annual = daily.resample("YS").sum()
for name, df in [("ventas_tiendas", by_shop), ("ventas_productos", by_item), ("ventas_categorias", by_category),
                 ("ventas_diarias", daily), ("ventas_semanales", weekly), ("ventas_mensuales", monthly), ("ventas_anuales", annual)]:
    table(df, name, 8)
note(f"Las cantidades netas suman **{sales.item_cnt_day.sum():,.0f} unidades**. "
     f"Las devoluciones representan **{sales.unidades_devueltas.sum()/sales.unidades_positivas.sum()*100:.3f}%** de las unidades positivas. "
     "El año 2015 termina en octubre y no debe compararse directamente con años completos.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(axes, ["item_price", "item_cnt_day", "ingresos"], ["Precio unitario", "Cantidad diaria", "Ingreso por registro"]):
    values = sales[col].dropna()
    if col == "item_price": values = values[values > 0]
    transformed = np.sign(values) * np.log1p(np.abs(values))
    ax.hist(transformed, bins=55, color="#246a91", alpha=.85)
    ax.set(title=title, xlabel="signo(x) × log(1 + |x|)", ylabel="Registros")
figure("01_distribuciones")
note("La escala logarítmica con signo permite visualizar colas y devoluciones sin borrarlas. "
     "Es una transformación de visualización; las estadísticas y agregaciones conservan la escala original.")

# %% [markdown]
# ## 6. Exploración temporal
#
# Se estudian tendencia, ciclos y cambios en el número de tiendas y productos con
# actividad. Se usan promedios por día para comparar días de semana y meses con
# distinta exposición. Estos patrones son descriptivos: no prueban por sí solos
# estacionalidad estable ni explican las causas de los cambios.

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
axes[0].plot(daily.index, daily.item_cnt_day, lw=.6, alpha=.45, label="Diario")
axes[0].plot(daily.index, daily.item_cnt_day.rolling(28, min_periods=28).mean(), color="#b94e25", lw=2, label="Media móvil 28 días")
axes[0].set(title="Ventas netas diarias y tendencia suavizada", ylabel="Unidades")
axes[0].legend()
axes[1].plot(monthly.index.to_timestamp(), monthly.item_cnt_day, marker="o", label="Unidades mensuales")
axes[1].set(title="Ventas netas mensuales", ylabel="Unidades", xlabel="Mes")
figure("02_series_temporales")
changes = monthly.item_cnt_day.pct_change().mul(100).rename("variacion_porcentual").to_frame()
table(changes.sort_values("variacion_porcentual").dropna().iloc[np.r_[0:3, -3:0]], "cambios_mensuales_extremos")
comparable = sales.loc[sales.mes_calendario.le(10)].groupby("anio").agg(AGG)
table(comparable, "comparacion_enero_octubre")
note(f"El máximo mensual se observa en **{monthly.item_cnt_day.idxmax()}**, con "
     f"**{monthly.item_cnt_day.max():,.0f} unidades netas**; el mínimo en **{monthly.item_cnt_day.idxmin()}**, "
     f"con **{monthly.item_cnt_day.min():,.0f}**. La tabla enero-octubre compara ventanas equivalentes entre años.")

# %%
weekday_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
weekday = daily.assign(dia_semana=daily.index.dayofweek).groupby("dia_semana").item_cnt_day.agg(["mean", "median", "count"])
calendar_month = daily.assign(mes_calendario=daily.index.month).groupby("mes_calendario").item_cnt_day.agg(["mean", "median", "count"])
table(weekday, "ciclos_dia_semana")
table(calendar_month, "ciclos_mes_calendario")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].bar(weekday_names, weekday["mean"], color="#246a91")
axes[0].tick_params(axis="x", rotation=40)
axes[0].set(title="Unidades netas promedio por día de semana", ylabel="Unidades por día")
axes[1].bar(calendar_month.index, calendar_month["mean"], color="#d67b3c")
axes[1].set(title="Unidades netas promedio por mes calendario", xlabel="Mes del año", ylabel="Unidades por día")
axes[1].set_xticks(range(1, 13))
figure("03_ciclos_calendario")
note(f"El promedio diario más alto corresponde a **{weekday_names[int(weekday['mean'].idxmax())]}**. "
     "Las diferencias mensuales mezclan tendencia y años distintos: noviembre y diciembre tienen menos años observados que enero-octubre.")

# %%
activity = sales.groupby("mes").agg(tiendas_activas=("shop_id", "nunique"), productos_activos=("item_id", "nunique"), registros=("item_id", "size"))
table(activity, "actividad_mensual", 34)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
activity.tiendas_activas.plot(ax=axes[0], marker="o", color="#246a91", title="Tiendas con registros por mes")
activity.productos_activos.plot(ax=axes[1], color="#d67b3c", title="Productos con registros por mes")
for ax in axes: ax.set_xlabel("Mes")
figure("04_actividad_mensual")
note("Los cambios de cobertura comercial pueden contribuir a los cambios del total agregado. "
     "No hay inventarios ni fechas de apertura/cierre para distinguir falta de demanda de falta de disponibilidad.")

# %% [markdown]
# ## 7. Comparación entre sucursales, productos y anomalías
#
# Los extremos se identifican, no se eliminan de forma automática. Un precio alto
# puede corresponder a un producto distinto y una cantidad alta a una operación de
# volumen. Se presenta sensibilidad al recorte como contraste, sin aplicarlo al EDA principal.

# %%
table(by_item.head(10).join(items.set_index("item_id")[["item_name"]]), "productos_mas_vendidos", 10)
table(by_item.tail(10).sort_values("item_cnt_day").join(items.set_index("item_id")[["item_name"]]), "productos_menos_vendidos", 10)
table(by_category.tail(10).sort_values("item_cnt_day"), "categorias_menos_vendidas", 10)
table(by_shop.tail(10).sort_values("item_cnt_day"), "tiendas_menor_venta", 10)
unsold_items = items.loc[~items.item_id.isin(sales.item_id)]
table(unsold_items, "catalogo_sin_ventas_observadas", 8)
table(sales.nlargest(10, "item_cnt_day")[list(raw.columns) + ["item_category_id", "shop_name"]], "cantidades_extremas", 10)
table(sales.nlargest(10, "item_price")[list(raw.columns) + ["item_category_id", "shop_name"]], "precios_extremos", 10)
q_price, q_units = sales.item_price.quantile(.999), sales.item_cnt_day.quantile(.999)
sales["extremo"] = sales.item_price.gt(q_price) | sales.item_cnt_day.abs().gt(q_units)
anomalies = sales.groupby("shop_id").agg(registros=("item_id", "size"), extremos=("extremo", "sum"))
anomalies["porcentaje_extremos"] = anomalies.extremos / anomalies.registros * 100
table(anomalies.sort_values("extremos", ascending=False), "extremos_por_tienda", 10)
sensitivity = pd.DataFrame({"original": monthly.item_cnt_day,
    "recorte_superior_p999": sales.assign(cantidad_recortada=sales.item_cnt_day.clip(upper=q_units)).groupby("mes").cantidad_recortada.sum()})
table(sensitivity, "sensibilidad_cantidades_extremas", 34)
note(f"Los umbrales exploratorios p99,9 son **{q_price:,.2f}** para precio y **{q_units:,.2f}** para cantidad. "
     "No son reglas de fraude ni justifican borrar observaciones. Los productos de menor venta incluyen devoluciones netas; "
     "los productos del catálogo sin historial se muestran aparte de los productos con ventas observadas.")

# %% [markdown]
# ## 8. Visualizaciones obligatorias
# La serie de ventas totales se presentó en el apartado 6. Aquí se añaden todas las
# sucursales, productos top 10, categorías y matrices de comportamiento temporal.
# Se usan IDs en los ejes para que los nombres largos no impidan leer las gráficas;
# las tablas conservan los nombres originales.

# %%
fig, ax = plt.subplots(figsize=(15, 5))
shop_order = by_shop.reset_index()
ax.bar(shop_order.shop_id.astype(str), shop_order.item_cnt_day, color="#246a91")
ax.set(title="Ventas netas por sucursal en todo el histórico", xlabel="shop_id (orden por ventas)", ylabel="Unidades")
ax.tick_params(axis="x", labelsize=8, rotation=90)
figure("05_ventas_todas_sucursales")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
by_item.head(10).item_cnt_day.sort_values().plot.barh(ax=axes[0], color="#246a91")
axes[0].set(title="Diez productos con más unidades netas", xlabel="Unidades", ylabel="item_id")
cat_order = by_category.reset_index()
cat_plot = cat_order.head(15).set_index("item_category_id").item_cnt_day
cat_plot.index = cat_plot.index.astype(str)
cat_plot.loc["Otras categorías"] = cat_order.iloc[15:].item_cnt_day.sum()
cat_plot.sort_values().plot.barh(ax=axes[1], color="#d67b3c")
axes[1].set(title="15 categorías principales y resto", xlabel="Unidades netas", ylabel="item_category_id")
figure("06_productos_y_categorias")

# %%
shop_matrix = sales.pivot_table(index="shop_id", columns="mes", values="item_cnt_day", aggfunc="sum", fill_value=0).reindex(columns=MONTHS, fill_value=0)
category_matrix = sales.pivot_table(index="item_category_id", columns="mes", values="item_cnt_day", aggfunc="sum", fill_value=0).reindex(columns=MONTHS, fill_value=0)
fig, axes = plt.subplots(2, 1, figsize=(15, 14))
for ax, matrix, title in [(axes[0], shop_matrix, "Unidades netas por tienda y mes"), (axes[1], category_matrix, "Unidades netas por categoría y mes")]:
    shown = np.sign(matrix) * np.log1p(np.abs(matrix))
    sns.heatmap(shown, ax=ax, cmap="vlag", center=0, xticklabels=matrix.columns.astype(str), yticklabels=2,
                cbar_kws={"label": "signo(x) × log(1 + |unidades|)"})
    ax.set(title=title, xlabel="Mes")
figure("07_matrices_temporales")
note("En estas matrices, cero significa ausencia de unidades netas registradas en la combinación, "
     "sin acreditar que hubiera stock o que la tienda estuviera abierta. La escala de color es logarítmica con signo.")

# %% [markdown]
# ## 9. Autocorrelación y autocorrelación parcial
#
# Se analizan el total mensual y diario, y dos tiendas, productos y pares de alta
# venta. La selección es ilustrativa y favorece series de mayor volumen; no representa
# al azar todo el catálogo. Las series mensuales se alinean a los 34 meses, usando
# cero cuando no se registra cantidad en ese mes. Esto no resuelve la disponibilidad
# comercial desconocida. Las bandas del 95% son aproximadas, no se corrigen por
# comparar múltiples rezagos y se interpretan con prudencia.

# %%
selected_shops = by_shop.head(2).reset_index().shop_id.tolist()
selected_items = by_item.head(2).index.tolist()
by_pair = sales.groupby(["shop_id", "item_id"]).item_cnt_day.sum().sort_values(ascending=False)
selected_pairs = by_pair.head(2).index.tolist()
series = {"Total mensual": monthly.item_cnt_day, "Total diario": daily.item_cnt_day}
selection = []
for shop in selected_shops:
    values = shop_matrix.loc[shop]
    series[f"Tienda {shop}"] = values
    selection.append({"nivel": "tienda", "shop_id": shop, "item_id": None, "meses_con_registro": int((sales.loc[sales.shop_id.eq(shop)].mes.nunique()))})
for item in selected_items:
    part = sales.loc[sales.item_id.eq(item)]
    series[f"Producto {item}"] = part.groupby("mes").item_cnt_day.sum().reindex(MONTHS, fill_value=0)
    selection.append({"nivel": "producto", "shop_id": None, "item_id": item, "meses_con_registro": int(part.mes.nunique())})
for shop, item in selected_pairs:
    part = sales.loc[sales.shop_id.eq(shop) & sales.item_id.eq(item)]
    series[f"Tienda {shop} / producto {item}"] = part.groupby("mes").item_cnt_day.sum().reindex(MONTHS, fill_value=0)
    selection.append({"nivel": "par", "shop_id": shop, "item_id": item, "meses_con_registro": int(part.mes.nunique())})
table(pd.DataFrame(selection), "series_seleccionadas", 10)
correlations = []
for pos, (label, values) in enumerate(series.items()):
    n = len(values)
    lags = min(35 if label == "Total diario" else 12, n // 2 - 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    axes[0].plot(np.arange(n), values.to_numpy(), lw=1.2)
    axes[0].set(title=label, xlabel="Día" if label == "Total diario" else "Mes (0 = enero 2013)", ylabel="Unidades netas")
    if values.nunique() > 1:
        plot_acf(values, lags=lags, zero=False, alpha=.05, ax=axes[1], title="ACF")
        plot_pacf(values, lags=lags, zero=False, alpha=.05, method="ywm", ax=axes[2], title="PACF")
        a, a_ci = acf(values, nlags=lags, alpha=.05, fft=True)
        p, p_ci = pacf(values, nlags=lags, alpha=.05, method="ywm")
        for lag in range(1, lags + 1):
            correlations.append({"serie": label, "n": n, "rezago": lag, "acf": a[lag], "pacf": p[lag],
                "acf_significativa": bool(a_ci[lag, 0] > 0 or a_ci[lag, 1] < 0),
                "pacf_significativa": bool(p_ci[lag, 0] > 0 or p_ci[lag, 1] < 0)})
        sig = [str(k) for k in range(1, lags + 1) if a_ci[k, 0] > 0 or a_ci[k, 1] < 0]
        sig_p = [str(k) for k in range(1, lags + 1) if p_ci[k, 0] > 0 or p_ci[k, 1] < 0]
        note(f"**{label}**: {n} observaciones. Rezagos ACF fuera de las bandas aproximadas: "
             + (", ".join(sig) if sig else "ninguno") + ". Rezagos PACF: "
             + (", ".join(sig_p) if sig_p else "ninguno")
             + ". ACF describe persistencia total; PACF contrasta la asociación restante al controlar rezagos intermedios.")
    else:
        note(f"**{label}** es constante; no se calcula autocorrelación.")
    figure(f"08_acf_pacf_{pos:02}")
correlations = pd.DataFrame(correlations)
table(correlations, "autocorrelaciones", 15)

# %%
delta = monthly.item_cnt_day.diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(delta, lags=12, zero=False, ax=axes[0], title="ACF del cambio mensual")
plot_pacf(delta, lags=12, zero=False, method="ywm", ax=axes[1], title="PACF del cambio mensual")
figure("09_autocorrelacion_cambio_mensual")
selected_lags = correlations.loc[((correlations.serie.eq("Total diario")) & correlations.rezago.isin([7, 14, 21, 28])) |
                                ((correlations.serie.eq("Total mensual")) & correlations.rezago.isin([1, 12]))]
table(selected_lags, "rezagos_calendario")
a7 = selected_lags.loc[selected_lags.serie.eq("Total diario") & selected_lags.rezago.eq(7)].iloc[0]
a12 = selected_lags.loc[selected_lags.serie.eq("Total mensual") & selected_lags.rezago.eq(12)].iloc[0]
note(f"El total diario presenta ACF a 7 días de **{a7.acf:.3f}**, "
     + ("fuera" if a7.acf_significativa else "dentro") + " de las bandas aproximadas; los rezagos semanales repetidos "
     "y la gráfica por día de semana son compatibles con un ciclo semanal. "
     f"En el total mensual, la ACF a 12 meses es **{a12.acf:.3f}**, "
     + ("fuera" if a12.acf_significativa else "dentro") + " de las bandas. "
     "Esta muestra corta no permite afirmar una estacionalidad anual estable ni descartar que exista.")
note("Se contrasta el nivel con el cambio mensual para reconocer que la tendencia puede producir dependencia persistente. "
     "Los rezagos 7, 14 y 28 días orientan la revisión de un posible ciclo semanal. El rezago 12 meses tiene "
     "muy pocos ciclos anuales disponibles. Una ACF significativa aislada no prueba estacionalidad y la PACF "
     "no determina por sí sola el orden final de un modelo: harán falta validación temporal y diagnóstico de residuos.")

# %% [markdown]
# ## 10. Hallazgos y preparación para forecasting
#
# La cobertura se calcula con los IDs originales y pares únicos. Un par con historial
# puede llevar meses inactivo; por ello también se mide presencia en octubre de 2015.
# No tener historial del par no equivale a que ni la tienda ni el producto sean nuevos.

# %%
history_pairs = pd.MultiIndex.from_frame(sales[["shop_id", "item_id"]].drop_duplicates())
test_pairs = pd.MultiIndex.from_frame(test[["shop_id", "item_id"]])
last_pairs = pd.MultiIndex.from_frame(sales.loc[sales.mes.eq(MONTHS[-1]), ["shop_id", "item_id"]].drop_duplicates())
coverage = test.copy()
coverage["par_con_historial"] = test_pairs.isin(history_pairs)
coverage["par_con_registro_octubre"] = test_pairs.isin(last_pairs)
coverage["producto_con_historial"] = test.item_id.isin(sales.item_id)
coverage["tienda_con_historial"] = test.shop_id.isin(sales.shop_id)
coverage_summary = coverage.iloc[:, 3:].agg(["sum", "mean"]).T.rename(columns={"sum": "pares", "mean": "proporcion"})
coverage_summary["porcentaje"] = coverage_summary.proporcion * 100
table(coverage_summary, "cobertura_test")
table(coverage.groupby(["tienda_con_historial", "producto_con_historial", "par_con_historial"]).size().rename("pares").to_frame(), "tipos_historial_test")
coverage.to_csv(ROOT / "data/processed/cobertura_por_par_test.csv", index=False)
coverage_pct = coverage.par_con_historial.mean() * 100
shop_share = by_shop.head(5).item_cnt_day.sum() / sales.item_cnt_day.sum() * 100
category_share = by_category.head(5).item_cnt_day.sum() / sales.item_cnt_day.sum() * 100
note(f"""### Conclusiones basadas en la ejecución

- **Cobertura del objetivo:** {coverage.par_con_historial.sum():,} de {len(test):,} pares de test tienen historial: **{coverage_pct:.2f}%**.
  El resto requiere resolver combinaciones sin historia; no basta con un modelo separado para cada par.
- **Recencia:** solo **{coverage.par_con_registro_octubre.mean()*100:.2f}%** de los pares tiene registros en octubre.
  Se debe distinguir historia remota, ausencia reciente y producto nuevo.
- **Concentración:** las cinco tiendas de mayor volumen reúnen **{shop_share:.2f}%** de las unidades netas y las cinco categorías principales **{category_share:.2f}%**.
  Conviene examinar errores por tienda y categoría, además del error total.
- **Variables candidatas:** calendario, identificación de tienda/producto/categoría, rezagos de cantidades y estadísticas móviles.
  Los precios contemporáneos de noviembre no se conocen en test; no deben introducirse como si estuvieran disponibles.
- **Calidad:** fechas y claves son consistentes y no hay nulos originales; se documentan los duplicados, devoluciones,
  el precio inválido, candidatos a alias y extremos. La falta de nulos no significa ausencia de problemas de negocio.
- **Temporalidad:** los patrones del total mezclan demanda y cambios de cobertura. Los ciclos sugeridos por ACF/PACF
  deben contrastarse fuera de muestra; 34 meses no bastan para afirmar un patrón anual estable para todas las series.
- **Siguiente etapa:** construir una tabla mensual tienda-producto; definir cómo representar meses sin registro y el universo de pares;
  crear rezagos mediante desplazamiento; separar entrenamiento y validación por fecha, por ejemplo entrenar hasta septiembre
  y validar octubre, seguido de varias ventanas retrospectivas. Ajustar transformaciones usando solo cada entrenamiento.

La exploración de todo el histórico es descriptiva. La selección de variables y transformaciones para un modelo se repetirá
dentro de sus ventanas de entrenamiento para evitar fuga de información. No se han atribuido causas a picos o caídas
que el conjunto no documenta.
""")

# %%
summary = {"filas_originales": len(raw), "filas_analisis": len(sales), "meses": len(MONTHS),
           "duplicados": duplicates_n, "precios_invalidos": int((~sales.precio_valido).sum()),
           "registros_devolucion": int(sales.item_cnt_day.lt(0).sum()),
           "unidades_netas": float(sales.item_cnt_day.sum()), "pares_test": len(test),
           "pares_test_con_historial": int(coverage.par_con_historial.sum()), "cobertura_test_pct": coverage_pct,
           "python": sys.version.split()[0], "raw_sha256": raw_hashes}
(ROOT / "reports/resumen_eda.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
assert all(hashlib.sha256((RAW / name).read_bytes()).hexdigest() == value for name, value in raw_hashes.items())
note("**Ejecución terminada.** Las tablas y gráficas se guardaron en `reports/`. Los CSV originales conservan sus hashes. "
     "Faltan las acciones externas del equipo: publicar en GitHub y adjuntar capturas reales del pull.")
