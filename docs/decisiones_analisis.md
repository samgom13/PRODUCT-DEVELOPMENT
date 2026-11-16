# Decisiones del análisis exploratorio

Las decisiones son explícitas y pueden discutirse antes de modelar. Los originales no se editan; al terminar se comprueban los hashes SHA-256 de los seis CSV.

1. **Objetivo.** Cantidad mensual neta por tienda-producto. El ingreso es una variable descriptiva adicional y no sustituye al objetivo.
2. **Duplicados.** Se excluyen de la vista de análisis las filas idénticas en las seis columnas originales. No se elimina una fila solo por compartir fecha, tienda y producto. Se cuantifica la diferencia y se advierte que no hay un ID de ticket para confirmar el origen.
3. **Devoluciones.** Se mantienen las cantidades negativas. Se calculan además unidades positivas y devueltas para evitar confundir ingresos brutos y netos.
4. **Precios inválidos.** Las cantidades de una fila con precio no positivo se conservan; sus ingresos quedan ausentes y el precio no se utiliza en las estadísticas de precios válidos. No se inventa un precio de sustitución.
5. **Tiendas similares.** Se detectan posibles alias normalizando nombres, pero no se fusionan los IDs sin confirmación. La cobertura usa siempre identificadores originales.
6. **Extremos.** Se listan precios y cantidades extremos; p99,9 es una señal exploratoria. El recorte se usa solo como contraste. No se retiran ventas de volumen sin evidencia de error.
7. **Tiempo.** Los agregados mensuales se alinean al horizonte observado. Un mes sin registros se representa con cero unidades observadas, sin concluir que hubiera stock o actividad comercial. La exposición diaria se tiene en cuenta al comparar días de semana o meses calendario.
8. **Autocorrelación.** Se ilustran series de alta venta y se informa su cobertura. Las bandas son aproximadas y los rezagos no determinan automáticamente el modelo. Hay solo 34 meses; una tendencia o cambio de cobertura puede inducir autocorrelación.
9. **Validación futura.** El EDA es descriptivo. El entrenamiento y ajuste de transformaciones deberán respetar ventanas temporales, y los rezagos deben desplazarse para que no utilicen el objetivo futuro.
10. **Interpretación.** Se describen asociaciones y patrones, sin atribuir causalidad ni inventar explicaciones de eventos que los datos no documentan.
