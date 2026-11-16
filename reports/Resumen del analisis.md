# Resultados iniciales del análisis de ventas

El historial contiene 2.935.849 registros entre enero de 2013 y octubre de 2015. El análisis utiliza 2.935.843 registros después de excluir seis duplicados exactos y conserva las devoluciones para calcular cantidades netas.

## Calidad de los datos

No se encontraron nulos originales, fechas no interpretables ni inconsistencias entre las fechas y el número de mes. Se detectó un precio no positivo y 7.356 registros con cantidades negativas. Las cantidades de la fila con precio inválido se conservan; su ingreso no se calcula. Los extremos se revisan sin eliminarlos automáticamente.

Los nombres sugieren tres grupos de posibles alias de tiendas: 0/57, 1/58 y 10/11. Se documentan como candidatos y se mantienen los identificadores originales hasta confirmar que corresponden a la misma sucursal.

## Implicaciones para el pronóstico

- De los 214.200 pares tienda-producto de test, 111.404 tienen historial: **52,01 %**. Las combinaciones sin historial necesitan una estrategia de respaldo.
- Solo 28.680 pares tienen registros en octubre de 2015: **13,39 %**. Tener historia en algún momento no equivale a actividad reciente.
- El **92,88 %** de los pares corresponde a un producto con historia y todas las tiendas de test aparecen en el entrenamiento. La ausencia del par no siempre significa un producto o una tienda nuevos.
- La ACF diaria a siete días es aproximadamente **0,636**, compatible con la exploración de un ciclo semanal. La ACF mensual a doce meses es aproximadamente **0,259** y queda dentro de las bandas aproximadas del análisis en niveles. Solo hay 34 observaciones mensuales; no se afirma una estacionalidad anual estable.
- Las cantidades de enero-octubre disminuyen de 1.249.381 en 2013 a 1.034.285 en 2014 y 764.584 en 2015. También cambia la cantidad de tiendas y productos con registros, por lo que no se atribuye esa reducción exclusivamente a demanda.

## Próximos pasos

Revisar las decisiones y las gráficas del notebook, confirmar los integrantes, publicar el repositorio y aportar las capturas reales del pull. Para la siguiente fase se propone una tabla mensual, rezagos y validación por ventanas temporales. No se han entrenado modelos en esta entrega.

Los cálculos completos, sus limitaciones y las interpretaciones están en `notebooks/01_exploratory_data_analysis.ipynb`. Las cifras verificables se conservan en `reports/resumen_eda.json` y `reports/tables/`.
