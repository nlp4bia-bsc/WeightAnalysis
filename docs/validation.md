# Validación y límites

## Suite actual

Desde un entorno con la distribución instalada y los extras de desarrollo:

```console
python -m pytest -q
python -m pytest -m gpu -v -rs
python -m build
python -m weight_analysis.examples.independent_analysis
python -m weight_analysis.examples.comparative_geometry
```

Las pruebas de CUDA real se omiten cuando `torch.cuda.is_available()` es False. Los oráculos
externos de LLM-Biometrics se omiten si no se aporta un checkout autoral auditado. Los ejemplos
portátiles usan CPU explícita. No hay entrenamiento, inferencia ni descarga de modelos en las
pruebas matemáticas. Las entradas privadas de investigación no forman parte de las fixtures.

Las comprobaciones incluyen referencias analíticas/NumPy/SciPy, invariancias, casos degenerados,
correlaciones, espectros, cuantiles, subespacios, proyecciones y centroides. Se contrastan las
fórmulas históricas, los filtros antes de materializar pesos, la equivalencia entre fragmentos y
cálculo conjunto, claves de fold/semilla, referencias, procedencia, persistencia atómica, caché,
continuación tras fallos y visualización sin agregaciones implícitas. Las pruebas de packaging
comprueban imports y CLI fuera del checkout, sin PYTHONPATH y sin importar cómputo al listar análisis.

La limpieza no modifica las fórmulas. Los informes históricos movidos a `src/weightspace/reports`
se comparan contra tablas sintéticas calculadas con el código previo del commit `1662f39`.
También se comprueban exportación de revisión, snapshots y compatibilidad de imports históricos.
El detalle de traslados y retiradas está en [repository_layout.md](repository_layout.md).

## Ejecución observada en Picasso

El log aportado por el usuario del job **3538733** muestra Python 3.12.3, PyTorch 2.10.0+cu128,
PyArrow 24.0.0 y una **NVIDIA B200**: **359 passed, 11 skipped, 5 failed**. Los cinco fallos
corresponden a subprocesos CLI que no encontraron `weight_analysis` fuera del checkout.
No fueron fallos de las fórmulas ni de detección CUDA. Los mensajes de shell
`not a valid test operator` no impidieron arrancar PyTorch con esa GPU; su origen exacto en la
configuración del módulo de Picasso no está disponible aquí y no se afirma haberlos corregido.

Se mantiene la exigencia de que las pruebas de instalación eliminen PYTHONPATH. Para satisfacerla,
`submit_tests.sh` prepara un wheel en login; el batch crea un venv temporal sin descargar dependencias,
hereda las bibliotecas del módulo y usa pip para instalar ese wheel. Comprueba la procedencia de
ambos paquetes, ejecuta el CLI desde un directorio ajeno al checkout y valida CUDA en el intérprete
que ejecutará pytest. La nueva prueba local reproduce ese contrato con un wheel real, no con
un parche de sys.path ni omitiendo las pruebas fallidas.

La nueva ejecución GPU en Picasso queda pendiente del usuario. Las pruebas locales usan CPU;
no se extrapolan tiempos, VRAM ni número de omisiones de este entorno al cluster.
Véase [picasso_experiments.md](picasso_experiments.md) para el envío actualizado.

## Baselines y retirada del flujo antiguo

Antes de la limpieza, la suite independiente dio **295 passed, 80 skipped**, excluyendo el
módulo SLURM histórico roto. Tras la limpieza, la suite completa y su ejecución desde un wheel instalado dieron
**313 passed, 80 skipped**, ya sin ese módulo obsoleto ni filtros de exclusión. El runner
instalado emitió un aviso de deprecación de un plugin anyio del entorno, sin fallos.
Las omisiones locales son CUDA real y los nueve oráculos externos no instalados.

El módulo retirado requería cinco scripts ausentes desde el checkout de referencia y producía
2 fallos y 13 errores. Sus condiciones relevantes se han trasladado a las pruebas del adaptador
actual, incluidos formatos de IDs e idempotencia; su antigua política de no usar dependencias
fue sustituida por `afterany`. Esta retirada no convierte sus antiguos fallos en pruebas aprobadas.

Ruff se aplica a la biblioteca canónica y a código/pruebas nuevos. Los módulos históricos
conservados tienen convenciones anteriores; no se reformatean sus fórmulas como parte de una
limpieza de rutas. `pip check`, la construcción de sdist/wheel y la instalación fuera del repo
forman parte de la validación de entrega.

## Límites científicos

El Parquet NER real tiene 459 agregados de cuatro folds; los pesos proporcionados por el usuario
son fold 1, con semilla numérica no especificada. La auditoría no inventa datos ausentes ni
convierte esos agregados en observaciones independientes. Sus datos locales se conservan fuera
de Git. No se dispone aquí del filesystem ni del scheduler de Picasso, del paper objetivo ni
de hipótesis preregistradas. Las pruebas matemáticas no confirman por sí solas resultados empíricos.
