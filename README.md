# Predict Future Sales

Proyecto 1 de Product Development, Universidad Galileo. El objetivo de esta entrega es explorar ventas diarias para preparar un pronóstico mensual por tienda-producto para noviembre de 2015.

## Integrantes

- Luis Zarate
- Samuel Gómez

Repositorio: [samgom13/PRODUCT-DEVELOPMENT](https://github.com/samgom13/PRODUCT-DEVELOPMENT).

## Abrir el análisis

Abrir la carpeta `Proyectos` en VS Code y el archivo `notebooks/01_exploratory_data_analysis.ipynb`. El notebook conserva sus resultados, gráficas e interpretaciones. Para repetirlo, elegir el intérprete de Python correspondiente y ejecutar todas las celdas.

En este equipo se ejecutó con Anaconda, Python 3.13.9. Para reproducir el entorno desde Anaconda Prompt, situado en esta carpeta:

```powershell
conda env create -f environment.yml
conda activate predict_future_sales
```

Seleccionar después ese entorno como kernel del notebook. Las dependencias analíticas también se documentan en `requirements.txt`.

## Datos

Descargar los seis CSV de [Predict Future Sales en Kaggle](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data) y colocarlos en `data/raw/`: `sales_train.csv`, `items.csv`, `item_categories.csv`, `shops.csv`, `test.csv` y `sample_submission.csv`.

Los datos crudos no se incluyen en Git. El notebook verifica sus hashes antes y después de la ejecución. `sample_submission.csv` ilustra el formato; sus valores no son ventas reales.

## Estructura

```text
Proyectos/
├── data/                 # raw, interim, processed y external; datos locales
├── docs/                 # requisitos, decisiones y procedencia de la plantilla
├── models/               # reservado para las etapas posteriores
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 01_exploratory_data_analysis.py
├── predict_future_sales/ # paquete reservado para evolución del proyecto
├── references/
├── reports/
│   ├── figures/          # gráficas exportadas
│   ├── tables/           # tablas agregadas del análisis
│   └── evidencias/       # capturas reales de Git por integrante
├── environment.yml
├── requirements.txt
└── Proyecto_1_EDA_Predict_Future_Sales.pdf
```

La estructura se generó con la [plantilla oficial Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/). El commit exacto de la plantilla consta en `docs/plantilla.json`. El archivo `.py` es una versión con celdas de VS Code del mismo análisis.

## Alcance y criterios

El notebook recorre los diez apartados: comprensión, calidad, descripción, exploración temporal, comparaciones, visualizaciones, ACF/PACF y conclusiones. `docs/criterios_eda.md` relaciona requisitos y evidencias. `docs/decisiones_analisis.md` explica el manejo de duplicados, devoluciones, precios inválidos, extremos, posibles alias y validación temporal futura.

Los resultados numéricos quedan en `reports/resumen_eda.json`. El detalle por cada par de test se guarda solo localmente en `data/processed/cobertura_por_par_test.csv`. No se entrenan modelos en esta fase.

## Antes de entregar

Revisar y discutir el notebook; comprobar su publicación en el repositorio indicado; realizar un pull real por integrante y guardar sus capturas en `reports/evidencias/`. El repositorio local por sí solo no demuestra publicación ni colaboración en GitHub.

El enunciado permite IA y menciona específicamente Claude Code dentro de VS Code. El equipo debe poder explicar y verificar las decisiones y resultados de su entrega.
