# Referencia arquitectónica: ner_lab

## Alcance y versión inspeccionada

Se consultó el repositorio oficial [nlp4bia-bsc/ner_lab](https://github.com/nlp4bia-bsc/ner_lab),
commit [`8640d4797b081a5f6b66d9b45c11269542a79a26`](https://github.com/nlp4bia-bsc/ner_lab/tree/8640d4797b081a5f6b66d9b45c11269542a79a26).
La copia de consulta se descargó en `/tmp/ner-lab-reference`, fuera de WeightAnalysis.
No se ha modificado ni incorporado código de ese proyecto. Este informe describe ese commit,
no promete equivalencia con versiones posteriores. WeightAnalysis sigue siendo una distribución
independiente, sin imports ni dependencia obligatoria de `lab` o de `ner_lab`.

La revisión incluye `README.md`, `pyproject.toml`, `uv.lock`, `docs/dev/{DESIGN,CODE_STYLE,DECISIONS}.md`,
`docs/{core,ner,nel}/`, `src/lab/{core,ner,nel}/`, `src/lab/cli.py` y `verification/`.
Se examinaron firmas, exports, contratos, lectores/escritores, procedencia y verificaciones;
no se ejecutó entrenamiento ni la verificación completa de ner_lab.

## Convenciones observadas y decisiones para WeightAnalysis

| Aspecto | Evidencia concreta en ner_lab | Aplicación en WeightAnalysis |
| --- | --- | --- |
| Paquetes | `src/lab`; `core`, `ner`, `nel`; paquete instalado, no imports accidentales desde cwd | Paquete independiente `weight_analysis` en `src/`; compatibilidad histórica separada |
| Modularización | `docs/dev/CODE_STYLE.md`, reglas 1–5; `ner/training/{arguments,devices,tracking,trainer,assessment}.py` | Dividir por responsabilidad científica y artefacto, evitando módulos triviales o jerarquías de managers |
| API | `README.md`, `docs/dev/DESIGN.md`: funciones con argumentos nombrados explícitos; cada paso invocable | Métricas atómicas importables, análisis seleccionables y composición mediante resultados explícitos |
| Configuración | `DESIGN.md`: no `config: dict` opaco; sí objeto tipado y documentado de una sola responsabilidad; `core/tasks.py`, `cli.py` despachan kwargs | Parámetros explícitos y configuración tipada; el núcleo científico no recibe argumentos de argparse |
| Resultados | `core/dataset.py:PreparedDataset`, `core/split.py:SplitResult`, `ner/training/assessment.py` | Resultados estructurados; metadatos y unidades de cómputo explícitos |
| Tipado | `from __future__ import annotations`, `str \| Path`, `Literal`, dataclasses; `src/lab/py.typed`; `ner/__init__.pyi` | Anotaciones modernas, contratos de datos y `py.typed`; no ocultar la API detrás de `Any` |
| Exports | `core/__init__.py` declara `__all__`; `ner/__init__.py` resuelve exports costosos de forma lazy | Exportar solo operaciones de entrada y contratos útiles; evitar cargar visualización o pipelines al importar una primitiva |
| Nombres | `CODE_STYLE.md`: funciones `read_*`/`write_*`, modelos `load_*`/`save_*`, rutas `_path`/`_dir`, DataFrames `_df`; `random_state` | Mantener vocabulario estable y semántica explícita de pesos finales frente a actualizaciones |
| Capas | `DESIGN.md`, `verification/verify_layering.py`: flujo de imports unidireccional; core no importa tareas; tareas hermanas no se importan | Lectura/contratos y métricas no dependen del CLI, persistencia ni scheduling; composición de análisis en el borde |
| Datos | `core/corpus.py` declara columnas y dtypes, valida claves y offsets; `core/spans.py` define otra tabla canónica | Esquemas de resultados y metadatos científicos validados; claves inequívocas para checkpoint, fold y semilla |
| I/O | `core/io.py`: DataFrames en memoria, Parquet con ZSTD, rutas explícitas; `core/provenance.py`: hashes y JSON | Separar cálculo de escritura; persistencia solicitada explícitamente, atómica y con procedencia |
| Side effects | `DESIGN.md`: ruta pura sin escritura obligatoria; orquestación devuelve también el objeto resultado | Ejecutar una métrica no dispara otros análisis, escritura ni configuración de logging global |
| Logging | `nel/retrieval/faiss_index.py` usa `logging.getLogger(__name__)`; `ner/training/tracking.py` adapta callbacks de terceros | Loggers de módulo; configuración de handlers y verbosidad en la aplicación del usuario o el CLI |
| Dependencias | `pyproject.toml`: hatchling, Python >=3.11, extras `torch`, `ner`, `nel`; `uv.lock` registra resolución | Dependencias declaradas del dominio científico; extras para desarrollo/visualización; ninguna dependencia en ner_lab |
| Estilo | Ruff: 100 columnas, target py311, imports ordenados; docstrings en prosa | Funciones pequeñas, errores tempranos y mensajes concretos; fórmulas y decisiones extensas en documentos científicos |
| Verificación | `verification/README.md`, `fixtures.py`, `_harness.py`, `verify_layering.py` | Fixtures sintéticas CPU, oráculos independientes, regresión, controles de imports y omisiones explícitas |
| Documentación | README breve enlaza índices por subdominio y `docs/dev/`; decisiones justificadas separadas | README y AGENTS breves; catálogo, API, auditoría y plan de integración fuera de ellos |

ner_lab **no usa pytest** en el commit inspeccionado: `verification/README.md` y la decisión D8
describen scripts ejecutables de comprobación, fuera de `src/`, sin suite pytest ni CI.
WeightAnalysis usa pytest porque lo exige su objetivo; adopta el principio de pruebas sintéticas,
regresión y layering, sin atribuir a la referencia una convención que no tiene.

No se copia la política de que `lab.core` nunca importe torch literalmente a todo `weight_analysis.core`:
leer tensores PyTorch es parte explícita del dominio de esta biblioteca. Sí se conserva la separación
entre contratos ligeros, lectura, matemáticas y orquestación, y se evita importar herramientas de
entrenamiento o visualización en rutas que no las necesitan.

## Límites de la evidencia científica

No se suministró el paper objetivo, su protocolo de auditoría, sus hipótesis preregistradas, ni los
checkpoints y tablas originales. El código histórico cita varias fuentes y distingue derivados propios
de fórmulas atribuidas a artículos. Esas citas no identifican por sí solas el paper del usuario.
Por tanto, puede comprobarse equivalencia matemática local y regresión sobre fixtures, pero no afirmar
que una métrica confirma una hipótesis real del paper ni reproducir resultados empíricos ausentes.

El archivo histórico `validacion_f1_weightspace/resultados/resumen_ejecutivo.md` describe resultados
exploratorios y reconoce la dependencia entre pares. No se toman sus recuentos, correlaciones o
valores p como resultados revalidados: los datos necesarios no están en el checkout.

Las prioridades siguientes son una recomendación condicionada a las preguntas científicas indicadas,
no una autorización implícita para ejecutar todos los análisis nuevos.

## Recomendación de análisis P0–P3

| Prioridad | Pregunta o propuesta | Justificación y criterio de entrada |
| --- | --- | --- |
| P0 | Separar ejecución de drift, fingerprints, comparaciones y agregación ya existentes | Necesario para validar cualquier afirmación posterior; baseline y equivalencia antes de cambiar definiciones |
| P0 | Controles deterministas de identidad, escala, rotación ortogonal y actualizaciones sintéticas | Detectan errores de normalización, W frente a delta W y alineación sin datos privados; son tests, no evidencia empírica del paper |
| P0 | Procedencia, claves fold/semilla y selección explícita de coordenadas | Impide reutilizar resultados de otros pesos o mezclar réplicas; hashes y configuraciones deben identificar el cálculo |
| P1 | Geometría de actualizaciones y localización por profundidad | Ya hay drift, cosenos y niveles tensor/capa/componente; exponer y reutilizar sus resultados aporta una prueba directa de dónde cambia el modelo sin SVD innecesaria |
| P1 | Alineación de subespacios en coordenadas compartidas | Complementa normas y cosenos cuando la pregunta exige direcciones; reportar rango, truncamiento y condición de ortonormalidad; nunca comparar pesos sin alineación entre arquitecturas |
| P1 | Descomposición factorial language/entity | Ya implementada y contrastable mediante identidad de sumas de cuadrados; describe un diseño completo de celdas, sin réplicas no estima error residual inferencial |
| P1 | Estructura de similitudes y centralidad descriptiva | Puede reutilizar distancias existentes; comprobar unicidad y cobertura del grafo; no confundir cercanía intrínseca con equivalencia funcional |
| P2 | Relación geometría–F1 con control de backbone y dominio | Requiere F1 emparejado por checkpoint/fold/semilla y protocolo de dependencia; medias de varios folds no son réplicas del único checkpoint guardado |
| P2 | Estabilidad entre folds y semillas | Requiere checkpoints de al menos dos réplicas reales por configuración; no inventar folds desde tablas agregadas ni reportar estabilidad con un fold |
| P2 | Controles negativos empíricos y contrastes de sensibilidad | Permutaciones de etiquetas o identidades deben preservar la estructura de dependencia relevante; decidir el contraste antes de mirar significación |
| P3 | Similitud funcional | Exige entradas, tokenizer, configuración de inferencia y predicciones compatibles; no puede inferirse de pesos ni distancias de fingerprints solamente |
| P3 | Interpolación y barreras de pérdida entre checkpoints | Exige arquitectura y parametrización alineadas, modelo ejecutable, tarea y datos de evaluación; coste y supuestos exceden el análisis weight-only actual |

La selección inicial recomendada es P0 y exposición de las operaciones P1 **ya presentes**.
P2 y P3 se documentan, pero no se añaden automáticamente. Si se proporciona el paper y los datos,
la siguiente decisión consiste en mapear cada hipótesis a una unidad de observación, un contraste,
un control negativo y un criterio de falsación antes de ampliar el catálogo.

## Plan de integración futura, sin ejecutarla

1. Estabilizar ahora los contratos de entrada (checkpoint y origen), selección, configuración y
   tablas de salida. Comprobar que no hay imports `lab`/`ner_lab`, rutas HPC obligatorias ni
   dependencias de scheduling.
2. Hacer que un consumidor externo entregue rutas y metadatos, invoque una operación Python y reciba
   un resultado. El consumidor controla cuándo persiste o compone; no es necesario introducir
   un gestor de jobs para distribuir estas llamadas.
3. Cuando se autorice la integración, acordar el destino con los mantenedores. El repositorio
   inspeccionado tiene namespace `lab` en `src/lab`, no un paquete importable llamado `ner_lab`.
   La ruta solicitada `ner_lab/analysis/` expresa la intención del usuario; una posible ubicación
   real sería `src/lab/analysis`, pero **no existe ni se ha creado** por esta tarea.
4. Añadir entonces un adaptador externo que transforme manifiestos de entrenamiento de ner_lab
   en contratos de WeightAnalysis. No hacer que el lector de pesos conozca `lab.ner.training`.
   Verificar especialmente la distinción observada en la referencia entre `base_model` y el
   checkpoint entrenado, y no deducir identidad del original a partir de formas tensoriales.
5. Resolver el empaquetado en esa tarea: dependencia opcional o migración controlada del namespace,
   compatibilidad de versiones de torch y adaptadores de esquemas. ner_lab fija torch 2.10.0;
   una prueba local con otra versión no demuestra compatibilidad con esa pin exacta.
6. Registrar únicamente las entradas de tarea autorizadas en el dispatcher de ner_lab y probar
   equivalencia Python/CLI, import layering y procedencia. La visualización consume resultados;
   no ejecuta métricas. Ningún paso exige copiar scheduling de Picasso a la biblioteca.
7. Mantener regresión contra resultados anteriores y documentar cualquier cambio deliberado de
   nombre, normalización, cobertura, definición estadística o esquema antes de retirar adaptadores.

Este plan no modifica ni integra ner_lab. La independencia actual y los contratos explícitos reducen
el trabajo posterior, pero no garantizan integración sin cambios: versiones, datos y APIs futuras
deben comprobarse en el momento de llevarla a cabo.
