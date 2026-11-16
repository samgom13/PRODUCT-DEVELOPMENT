# Publicación y evidencias de GitHub

El proyecto ya tiene un repositorio Git local en esta carpeta. El repositorio del equipo es https://github.com/samgom13/PRODUCT-DEVELOPMENT y los integrantes son Luis Zarate y Samuel Gómez. La cuenta samgom13 está autenticada y el primer envío a main se verificó correctamente.

## Publicación realizada y revisión local

Desde una terminal abierta en `Proyectos`, revisar los cambios y confirmar que ningún CSV de `data/raw/` esté incluido:

```powershell
git status
git diff --cached --stat
git ls-files data
```

En `data` solo deben aparecer archivos `.gitkeep`. El notebook ejecutado, las gráficas y las tablas agregadas sí forman parte de los resultados.

Usar la identidad local de Samuel Gómez, comprobar que `origin` corresponde al repositorio del equipo y publicar la rama `main`. Si el remoto ya contiene trabajo, primero se debe revisar e integrar sin sobrescribir su historial. No incluir contraseñas ni tokens en el código o en capturas.

## Captura de cada integrante

Una vez publicado el repositorio y descargado en la computadora de cada integrante, ejecutar desde su copia local:

```powershell
git pull origin main
git status
```

Tomar una captura donde se vean la carpeta del proyecto, el comando y su resultado, y guardarla en `reports/evidencias/`. La evidencia debe ser una ejecución real de cada integrante; no basta una imagen de ejemplo o una salida redactada.

Adjuntar el enlace de GitHub y el notebook ejecutado según la plataforma del curso. Las capturas pendientes no se consideran completas por haber creado esta guía.
