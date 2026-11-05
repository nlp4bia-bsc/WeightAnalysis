# Organización del repositorio y migración de utilidades

Todo el código de producción vive en `src/`. La raíz conserva configuración, documentación,
pruebas, los adaptadores de batch y el Parquet experimental ignorado. Los resultados locales
y datos privados no se eliminan como parte de la limpieza.

```text
src/
  weight_analysis/            # API científica independiente
    core/ metrics/ analyses/ comparisons/ statistics/ results/ visualization/
    execution/ experiments/  # unidades y manifiestos; envío explícito separado de cálculo
    maintenance/             # inspección, exportación, snapshots y entorno de pruebas
    examples/                # ejemplos sintéticos ejecutables como módulos
  weightspace/               # compatibilidad con imports y formatos históricos
    reports/                 # informes exploratorios antiguos, opt-in y parametrizados
scripts/picasso/              # adaptadores de ejecución externos al cálculo
tests/                       # regresiones, oráculos sintéticos y contratos de instalación
docs/history/                # material histórico identificado como no revalidado
```

## Correspondencia de comandos

Instalar con `python -m pip install -e ".[dev,visualization]"` en un entorno Python adecuado.
En Picasso usar el módulo CUDA y el procedimiento de pruebas de la guía específica.

| Entrada anterior | Entrada actual |
| --- | --- |
| `python weightspace.py ...` | `python -m weightspace ...` o `weightspace ...` |
| `python weightspace_parallel.py ...` | `python -m weightspace.parallel ...` o `weightspace-parallel ...` |
| `check_weightspace_grid.py` | `python -m weight_analysis.maintenance check-grid --originals RUTA --trained RUTA` |
| `snapshot_finished_checkpoints.py` | `python -m weight_analysis.maintenance snapshot --trained RUTA --snapshot RUTA [--exclude idioma/entidad/modelo] [--expected N]` |
| `exportar_revision_460.py` | `python -m weight_analysis.maintenance export-review --input-dir RUTA --output-dir RUTA [--legacy-encoder-pairs]` |
| `recover_05_shards.py` | `python -m weightspace.recovery --run-dir RUTA --shard N --piece N --pieces N` |
| `correlacion_f1_weightspace.py` | `python -m weightspace.reports correlate --data-dir RUTA --performance F1.parquet --output-dir RUTA` |
| `validacion_f1_weightspace/validar_f1_weightspace.py` | `python -m weightspace.reports validate --data-dir RUTA --performance F1.parquet --output-dir RUTA` |
| `validacion_f1_weightspace/auditar_datos.py` | `python -m weightspace.reports audit --data-dir RUTA --performance F1.parquet --output-dir RUTA` |
| `validacion_f1_weightspace/analizar_capas_f1.py` | `python -m weightspace.reports layers --data-dir RUTA --performance F1.parquet --output-dir RUTA` |
| `validacion_f1_weightspace/sensibilidad_backbones.py` | `python -m weightspace.reports sensitivity --output-dir RUTA [--input rankings_f1.csv]` |
| `python -m weightspace.benchmark ...` | Sigue disponible; también `python -m weight_analysis.maintenance benchmark ...` |
| `python examples/independent_analysis.py` | `python -m weight_analysis.examples.independent_analysis` |
| `python examples/comparative_geometry.py` | `python -m weight_analysis.examples.comparative_geometry` |

Los informes históricos conservan fórmulas, medianas, permutaciones y salidas CSV, pero sus
rutas de entrada/salida ahora son explícitas. Se mantienen `WS_DATA`/`WS_F1` como alternativas
para informes. `--seed` y `--permutations` reproducen sus opciones aleatorias. La validación
histórica requiere el extra `statistics` por statsmodels. No se ejecutan informes al importar
un módulo ni se activan sus pruebas exploratorias desde el flujo científico normalizado.
Sus p/q siguen siendo exploratorios, no inferencia confirmatoria. La biblioteca nueva conserva
sus agregaciones por energía y sus restricciones de pseudorreplicación.

La exportación de revisión admite directorios anidados, columnas escalares, CSV gzip,
inventario y ZIP. `--legacy-encoder-pairs` reproduce el filtro histórico de 03/04/05 a encoder;
no se aplica implícitamente a los datos nuevos. Se exige una salida nueva y fuera del árbol
original para proteger fuentes y revisiones existentes. El snapshot conserva una selección
mediante symlinks; no copia ni hace inmutables los contenidos de los checkpoints.

## Retirado y conservado

Se eliminan los siete `.py` de la raíz, el directorio de informes ejecutables de la raíz y
sus tres logs generados. El resumen histórico se conserva en
[history/legacy_f1_exploration.md](history/legacy_f1_exploration.md), sin revalidar sus conclusiones.
Los imports `weightspace.*` y sus CLIs se mantienen en `src/weightspace`; no se pierde la lectura
de formatos antiguos ni el cálculo/finalización de shards históricos. Sus cachés siguen
vinculadas al hash del código: mantener el checkout original al recuperar un run antiguo;
la migración no relaja sus validaciones ni mezcla versiones. Los launchers de raíz
quedan sustituidos por módulos y comandos instalados.

`tests/test_slurm_orchestration.py` exigía cinco scripts de shell que ya estaban ausentes:
su baseline tenía 2 fallos y 13 errores. Se retira con ese adaptador obsoleto. Las garantías
aplicables se comprueban en `test_picasso_experiments.py`: recursos DGX/CPU, nombres de jobs,
mutación de entradas, reintentos, formatos de IDs, idempotencia y cuarentena de una respuesta
exitosa cuyo ID no se puede reconocer. La política nueva usa `afterany` y no conserva la antigua
prohibición de dependencias. No se han marcado aquellas pruebas ausentes como aprobadas.

`tests/fixtures/legacy_reports.json` contiene oráculos de datos sintéticos obtenidos de los
scripts sin modificar del commit `1662f39`, con siete permutaciones para que sean rápidos.
Las regresiones comparan todas las tablas numéricas con tolerancia para redondeo. No contienen
resultados NER privados ni pretenden validar hipótesis empíricas.
