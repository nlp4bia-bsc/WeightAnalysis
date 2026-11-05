# Validación de la modularización

## Entorno y baseline

Baseline obtenido antes de cambiar algoritmos: commit
`285cd8bec45f0cfb30807519ca2e539a3acdc4df`, 51 pruebas CPU aprobadas.
Referencia arquitectónica consultada fuera del proyecto y sin modificar:
`nlp4bia-bsc/ner_lab`, commit `8640d4797b081a5f6b66d9b45c11269542a79a26`.

La validación usa Python 3.12.14, CPU, dos hilos por biblioteca numérica,
fixtures sintéticas y ningún checkpoint descargado. No se ha ejecutado inferencia.
Las funciones históricas conservan sus nombres y cálculos a través de fachadas.
Las correcciones numéricas se exponen en funciones nuevas, documentadas en
`mathematical_audit.md`, sin sobrescribir resultados históricos.

## Resultados de la modularización inicial

| Objetivo | Resultado |
| --- | --- |
| Suite completa, PyTorch 2.15.0+cpu | **221 passed, 9 skipped, 2 failed, 13 errors** |
| Suite CPU sin el módulo SLURM ya roto, PyTorch 2.10.0+cpu | **221 passed, 9 skipped** |
| Ruff sobre src, ejemplo y nuevas pruebas | Aprobado |
| Instalación editable y pip check | Aprobados |
| Build sdist y wheel desde sdist | Aprobado |
| Wheel instalado fuera del checkout, imports públicos e históricos | Aprobados |
| Ejemplo Python ejecutado contra wheel instalado | 4 filas drift, 4 fingerprints; equivalencia fragmentos/conjunto, merge y Parquet correctos |
| Referencia ner_lab | Checkout de lectura sin cambios; no integración ni dependencia añadida |
| Restricción de archivos | Ningún .sh ni .sbatch nuevo |

La comprobación con torch 2.10 usa un intérprete temporal separado para que API y
subprocesos CLI carguen la misma versión. Se reutilizan las demás dependencias del
entorno; esto valida esa versión de torch, no toda la combinación de dependencias de
ner_lab ni su integración. Los identificadores incluyen versiones numéricas, por lo
que distintas versiones pueden producir identificadores distintos aun con datos iguales.

Los fallos de la suite completa son previos: `tests/test_slurm_orchestration.py`
necesita cinco archivos ausentes del commit original:

- `submit_weightspace_parallel.sh`
- `advance_weightspace_parallel.sh`
- `run_weightspace_parallel.sh`
- `retry_weightspace_parallel.sh`
- `slurm_jobid.sh`

No se han recreado, porque la tarea prohíbe añadir scripts e infraestructura SLURM.
Esas pruebas se ejecutaron y sus fallos se mantienen visibles. No se han alterado
tests históricos ni marcado como aprobados. Las nueve omisiones corresponden a
oráculos optativos de LLM-Biometrics: su código upstream auditado no está instalado.

## Revisión previa al commit

La revisión independiente detectó que un intermedio calculado con unos originales podía
aceptarse junto a otros originales declarados en una llamada posterior. Ahora se comparan
también los hashes del original declarado con los registrados en el intermedio. Una copia
idéntica en otra ruta sigue siendo válida. Cinco nuevas regresiones cubren espectros base,
espectros delta y drift, además de la reubicación sin cambios de contenido.

La comprobación final en el entorno restaurado, con PyTorch 2.15.0+cpu y excluyendo únicamente
el módulo SLURM ya roto, dio **226 passed, 9 skipped**. Ruff, pip check y build de sdist/wheel
también aprobaron. Los fallos históricos SLURM descritos arriba siguen pendientes.

## Qué comprueban las pruebas nuevas

Referencias NumPy/SciPy y ejemplos analíticos contrastan normas, normalizaciones,
Pearson, espectros, entropías, rangos, cuantiles y subespacios. Se cubren ceros,
no finitos, formas incompatibles, invariancias, empates, bases de rango deficiente
y cancelación numérica. Los tests históricos también contrastan las fórmulas
paper-inspired; no se presentan como ejecución de código autoral externo.

Las pruebas de integración comprueban lectura segura de formatos y shards,
prefijos HF, filtros antes de materializar parámetros, ausencia de SVD no solicitada,
dependencias por fuente, aislamiento de fold/seed, diseños factoriales, centralidad,
persistencia atómica, fallos durante escritura, caché corrupta/obsoleta, identidad
float64 exacta y equivalencia entre unidades disjuntas y cálculo conjunto. El CLI
usa la API real desde un directorio temporal; listar análisis no importa torch.
La visualización utiliza Agg y resultados ya calculados, sin recomputar pesos.

## Ampliación CUDA

La API solicita GPU por defecto; las fixtures CPU y el ejemplo portátil ahora pasan
`device="cpu"` explícitamente. Tras ampliar las rutas de cálculo, el flujo independiente dio
**246 passed, 75 skipped** en este entorno CPU: nueve omisiones son los oráculos previos y
66 son pruebas de ejecución CUDA real sin hardware disponible. Los tests CPU contrastan
además alias/índices, error sin fallback, interpolación, cuantiles, overlap y el algoritmo
truncado contra una SVD exacta, sin modificar el RNG global. La velocidad y la equivalencia en
hardware CUDA siguen pendientes de ejecutar esas pruebas en destino; véase `gpu.md`.
El marcador `gpu` permite seleccionar solo CUDA real con `python -m pytest -m gpu -v -rs`;
incluye métricas individuales, los ocho análisis, formatos/precisión de tensores, RNG,
residencia de kernels, caché, CLI y varias GPU cuando están disponibles.

## Reproducción

Instalación de desarrollo sin GPU, con un Python compatible:

```console
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -e ".[dev,visualization]"
python -m pytest -q
```

La última orden incluye los fallos SLURM documentados. Para ejecutar solo el flujo
independiente disponible:

```console
python -m pytest -q --ignore=tests/test_slurm_orchestration.py
python examples/independent_analysis.py
python -m build
```

No hay estado global de jobs ni servicios que iniciar. Los checkpoints deben ser
inmutables durante cada cálculo; los hashes identifican sus contenidos al iniciar
la unidad y permiten invalidar reutilizaciones posteriores. La procedencia no prueba
por sí sola que el original declarado sea el usado durante el entrenamiento.

Las conclusiones empíricas y la equivalencia con resultados reales siguen pendientes:
no se han suministrado el paper objetivo, sus hipótesis, los checkpoints ni las tablas F1.

## Extensión de geometría comparativa fina

La extensión añade pruebas independientes en `tests/test_comparative_extension.py`:
selección determinista por F1 y empates, ámbito checkpoint/agregado, política explícita de cruce,
bloques de coordenadas, energías/cosenos/Pearson históricos, centroides, proyecciones,
residuos, dispersión y LOO con oráculos NumPy no collineales. Se comprueban invariancia al
origen, referencias alternativas, evidencia de vocabulario, reubicación de checkpoints,
equivalencia de fragmentos de pares/capas/métricas, particiones Parquet y digest, reanudación,
caché corrupta, dependencia espectral reutilizada, incidencias, prints y fallos del log.
Las figuras se contrastan con Agg y rechazan agregaciones implícitas. Las asociaciones F1
registran checkpoints/scores compartidos sin valores p iid.

Se corrigió la reducción del perfil de adaptación cuando recibe un dispositivo explícito:
la rama tensorial usa PyTorch en su dispositivo en lugar de pasar por NumPy. Una regresión
independiente contrasta tanto la rama histórica como la nueva con la misma fórmula NumPy.
No se cambiaron definiciones matemáticas históricas. El lector Parquet lee el archivo concreto
para impedir que Arrow infiera columnas extra de los directorios Hive de partición.

La suite independiente final dio **283 passed, 79 skipped** con PyTorch CPU en este entorno;
las omisiones son nueve oráculos externos y 70 casos CUDA sin hardware. De estos últimos,
cuatro cubren las nuevas rutas directa/intrínseca y grupos por entidad/idioma, con comparación
CPU/CUDA de resultados detallados y agregados. No se afirma haber probado kernels o velocidad
en una GPU real. Los fallos históricos del módulo SLURM descritos arriba siguen pendientes y
ese módulo se excluyó explícitamente de este recuento; no se han creado scripts para resolverlo.

Ruff sobre `src/`, el test nuevo y el ejemplo nuevo, `pip check` y build de sdist/wheel pasan.
Una ejecución de Ruff sobre todos los tests históricos también señala 23 incidencias de estilo
previas; no se han reformateado ni alterado esos tests como parte de la extensión.
El ejemplo `python examples/comparative_geometry.py` completa cálculo, reunión, agregación y
lectura/escritura Parquet sin red. La guía matemática, esquemas y límites están en
[comparative_geometry.md](comparative_geometry.md).

## Parquet experimental y adaptadores Picasso

Se inspeccionó el Parquet experimental recibido: 459 filas, 16 columnas, 22 backbones,
siete idiomas y tres entidades. La auditoría local conserva esquema completo y hash; el
original y los derivados locales están excluidos de Git. No hay nulos ni duplicados por
idioma/entidad/modelo; faltan las tres celdas neerlandesas de un backbone. Se respetan los
cuatro folds agregados del F1 y la declaración del usuario de pesos de fold 1, sin inventar
la semilla numérica. No se pudo inspeccionar el filesystem remoto de Picasso.

La validación independiente con el adaptador nuevo da **295 passed, 80 skipped**. La suite
`tests/test_picasso_experiments.py` aporta doce casos CPU y un caso CUDA omitido sin GPU.
Cubre auditoría, cobertura completa de macros, rankings deterministas, manifiestos,
F1 ausente/inválido, ejecución de arrays con pesos sintéticos, unión de ambos extremos,
centroides por backbone, reanudación, señal/presupuesto, modificación de entradas, marcadores
corruptos, figuras desde tablas y finalización que falla si intenta leer pesos.

Los scripts pasan `bash -n`. El envío se prueba con sbatch simulado y sus IDs/dependencias;
no se han enviado jobs ni se han medido tiempos/VRAM en Picasso. La reserva de 8 horas y una
GPU por elemento no garantiza que un shard grande termine en un único turno. El run conserva
unidades completadas y publica los índices que necesitan reanudación. Ruff sobre `src/` y el
test nuevo, `pip check` y build de sdist/wheel pasan. El módulo SLURM **histórico** con scripts
raíz ausentes sigue excluido explícitamente, como en las regresiones anteriores; los nuevos
adaptadores tienen su propia cobertura y no simulan compatibilidad con aquel flujo antiguo.
Véase [picasso_experiments.md](picasso_experiments.md) para comandos, esquemas y límites.
