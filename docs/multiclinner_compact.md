# MultiClinNER: ejecución compacta en Picasso

El backend `weight_analysis.experiments.compact` crea un **run nuevo**. No migra, mezcla,
abre como escribible ni elimina datos de `run-002`. Los módulos históricos y su backend
`picasso-1` conservan sus interfaces. No se ha enviado ningún trabajo real durante desarrollo.

## Entradas y decisiones experimentales

* Fine-tuning: `/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weights/<idioma>/<entidad>/<backbone>/`.
* Pretraining original confirmado por el usuario: `$HOME/ner_lab/TFG_Aziz/models/<backbone>/`.
* F1 original, intacto e ignorado por Git: `WeightAnalysis/results_aggregated.parquet`.
* Salida: `/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weigh_analysis_final/`.
* Pesos disponibles: fold 1; semilla numérica desconocida (`null`, nunca 42 por suposición).
* El archivo recibido tiene 459 configuraciones, 7 idiomas, 3 entidades, 22 backbones,
  F1 estricto medio y SD de **cuatro folds**. No contiene las puntuaciones de la fold 1.
  SHA256: `f9d9aac50a7540c63fe3c38f77962297571c9e051e5af5f8fd5fb47afa5d4fac`.
* Emparejamiento exacto por idioma, entidad y backbone; hashes completos identifican los pesos.
  La declaración de procedencia del usuario se conserva como declaración, sin presentarla como
  evidencia de entrenamiento verificada independientemente. Arquitectura y formas se comprueban
  además. Folds, semillas e inicializaciones diferentes se mantienen en dominios separados.

Se inspecciona el esquema real de F1 antes del ranking. Se preservan media, SD, número de folds,
origen y nivel de agregación. Orden: F1 descendente y SHA256 de identidad del modelo ascendente
para empates; rango denso y orden de selección son columnas distintas. Best/worst y top/bottom
son filtros de una única tabla de pares, no copias de observaciones geométricas. La selección en
los mismos resultados analizados y todas las asociaciones entre pares son **descriptivas**.
No se calculan p-values iid ni se atribuye el F1 agregado al checkpoint físico de fold 1.

## Organización y cantidad de archivos

```text
run-003-packed/
  manifest/plan.json                 # definición, identidades, selección y coste
  manifest/checkpoint_scores.parquet # una correspondencia verificable por checkpoint
  manifest/submissions/              # mapas, IDs y recibos del scheduler
  manifest/datasets.json             # publicación atómica de los archivos vigentes
  manifest/views.sql                 # vistas DuckDB, sin duplicar datos
  logs/                             # stdout, stderr y JSONL centralizados
  state/                            # un SQLite + estado/bloqueo por tarea
  parquet/atomic/                    # hechos tensoriales, estadísticos, fingerprints
  parquet/aggregated/                # valores calculados desde estadísticas suficientes
  parquet/f1/                        # scores, rankings y pares canónicos
  parquet/intrinsic/                 # perfiles y pares entre backbones
  figures/
  reports/                          # cobertura, exclusiones, errores, asociaciones y coste
```

**No hay un archivo, directorio, manifest ni sidecar por tensor o métrica.** Cada tarea usa
un diario SQLite transaccional; los payloads temporales son Arrow IPC con ZSTD. Al confirmar
Parquet se libera el payload del diario. Las unidades lógicas pueden ser miles: sus estados
son filas en SQLite, no inodos. Los archivos Parquet usan ZSTD, diccionarios, estadísticas y
row groups de 32.768 filas. El tamaño objetivo predeterminado de una parte es 64 MiB de datos
Arrow sin comprimir; `--part-mib` permite aumentarlo.

Durante el cálculo existen partes por tarea. La compactación incremental reúne partes pequeñas.
Al completar todas las tareas, una consolidación final reúne los hechos **por estudio/familia**
en `parquet/atomic/task-studies/<familia>--part-*.parquet`. Se publica el manifest nuevo antes
de recuperar las partes temporales de ese mismo run. Se conservan todas las filas y sus IDs.
La repetición de `finalize` o de un worker sobre un run sellado no recalcula los pesos.

Para la rejilla actual, el objetivo operativo es del orden de **100–300 Parquet finales**,
según dimensiones, cobertura y tamaño elegido; no es una promesa sobre un volumen aún no medido.
200–300 archivos por estudios son perfectamente aceptables. El límite preventivo del run
completo es 8.000 archivos incluyendo logs/estado; excederlo detiene la escritura de forma
reanudable, sin omitir resultados para cumplir una cuota. `reports/cost.json` contiene la
estimación previa y el manifest permite contar las partes reales. No se generan copias por
idioma, entidad, métrica o figura: son filtros sobre columnas y vistas.

El esquema canónico comparte campos tipados entre partes: `record_id`, `unit_id`, `kind`,
`family`, `artifact_a/b`, `group_id`, `backbone`, `language/entity` y extremos, `fold_a/b`,
`seed_a/b`, `reference_id/kind`, `source`, `scope`, `layer`, `component`, `embedding_type`,
`tensor`, `relative_depth`, `metric`, `value`, `status`, `exactness`, `execution_id` y `n`.
`value` es double nullable; los arrays, estadísticas suficientes y alias se conservan en
`payload` JSON. La procedencia común (run, versión, hashes, selección, configuración y
fuente F1) vive en el manifest, sin copiarla millones de veces en cada observación.

## Cálculos, reutilización y definiciones

El recorrido es tensor → checkpoints compatibles → bloques. Los lectores se mantienen abiertos
durante la tarea. El pretraining se comparte entre checkpoints. Si el tensor de todos los
modelos cabe en `--resident-gib`, se transfiere una vez a GPU y se reutiliza para reducciones,
cuantiles y SVD. Para embeddings grandes se mantiene un recorrido acotado por bloques y una
materialización por checkpoint cuando su fingerprint necesita la matriz completa; no se repite
esa operación por pareja. `resident-gib` acota la residencia planificada, no toda la memoria
de trabajo de SVD/cuántiles. Un OOM se registra para la unidad y permite reintentarla.
Si las estadísticas prueban igualdad exacta de coordenadas (máximo de diferencias cero),
los fingerprints se comparten mediante un alias verificable. No se repite la SVD ni se
duplica su espectro por cada checkpoint idéntico. La vista `weight_fingerprints` resuelve
esos alias; los Parquet conservan tanto la identidad del modelo como la referencia canónica.

Sea `x_i` un peso final y `b` el pretraining común, `d_i=x_i-b`. Se guardan Gram de `d_i`,
medias, covarianzas centradas, diferencias cuadráticas/L1/máximo y contadores de signos.
El Gram final se reconstruye con
`<x_i,x_j>=<d_i,d_j>+<d_i,b>+<d_j,b>+||b||²`.
Las diferencias se acumulan **directamente**, evitando cancelación de diagonales de Gram.
Covarianzas y medias se combinan con Welford paralelo. Las reducciones usan float64; no se usa
TF32 para ellas. Sign agreement conserva el umbral histórico `1e-8`. `VectorMoments` conserva
las definiciones históricas de L2, RMSE, coseno, Pearson, magnitud relativa y normas.

* Drift respecto a pretraining: energía, RMS, L2 relativa, coseno base/final, normas, MAE,
  máximo y desplazamiento medio. L1/MAE y máximo complementan L2 para localizar drift concentrado.
* Todos los pares del mismo backbone/inicialización/replica: mismo idioma entre entidades,
  misma entidad entre idiomas y cruces idioma–entidad. Coseno, Pearson, L2 normalizada y signos
  se distinguen para pesos finales y deltas. L2, RMSE, L1 y máximo son invariantes a un origen
  compartido: se calculan/guardan una vez, en la fuente `final`.
* Adaptación compartida: grupos completos de entidades por idioma, idiomas por entidad y
  todas las celdas por backbone. Media `m=Σd_i/N`, energía de `m`, distancia a `m`, energía
  proyectada `(<d_i,m>²/||m||²)/||d_i||²`, coseno y residuo ortogonal. Leave-one-out reutiliza
  `m_-i=(Nm-d_i)/(N-1)` y `||d_i-m_-i||=N/(N-1)||d_i-m||`.
  La dispersión es la media explícita de distancias cuadráticas al centroide. No son pruebas
  causales de especialización. Dirección nula y denominadores nulos son valores indefinidos.
* Perfiles por tensor, capa, componente, capa×componente, embeddings y encoder. Las normas
  se reconstruyen sumando energías; Pearson usa covarianzas suficientes; los espectros no se
  promedian para inventar un espectro del encoder. Se conserva cobertura de coordenadas.
* SVD y cuantiles: una llamada por matriz/origen devuelve también valores singulares para
  persistencia. Fuentes final, delta y pretraining, este último una vez por tensor/referencia.
  Se guardan curvas de energía de 64 puntos, rangos, entropías y 101 cuantiles normalizados.
  El modo predeterminado **`exact`** calcula el espectro completo; `auto` es una elección
  explícita del usuario, usa el umbral declarado y marca los truncamientos como
  `topk_approx`. Nunca presenta entropía/rango efectivo de un espectro truncado como exactos.
* Backbones: comparación **intrínseca** mediante RMS por profundidad/componente, TRACE,
  curvas espectrales normalizadas, cuantiles y rango efectivo relativo. No se restan matrices
  de arquitecturas distintas. Se utiliza una rejilla relativa de 32 puntos; no alinea neuronas.
  RMS suma energía y número de parámetros. Los perfiles de características adimensionales
  declaran pesos iguales por tensor y por posición de la rejilla. W1 de cuantiles utiliza
  cuadratura trapezoidal: es una aproximación de distribución explícita.
  No se interpolan perfiles con fingerprints pendientes; TRACE requiere tres componentes
  válidas. Las figuras consumen estos valores ya calculados, sin agregarlos.

Embeddings: se verifica la correspondencia token→índice (incluyendo tokens añadidos) o una
identidad verificable del artefacto tokenizer. Igualdad de formas o nombre del modelo no
constituyen evidencia. Si falta correspondencia, solo se excluyen esas coordenadas para
ese checkpoint; el motivo queda persistido y el resto del encoder continúa. Un grupo
incompleto no se convierte en una media de las entidades o idiomas supervivientes.

## Comandos en Picasso

Las pruebas GPU existentes siguen disponibles con `bash scripts/picasso/submit_tests.sh`.
Los scripts siguientes son nuevos y no envían nada hasta ejecutarlos explícitamente.

```bash
cd ~/WeightAnalysis
export WEIGHT_ANALYSIS_PROJECT_DIR="$PWD"
export RUN_DIR=/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weigh_analysis_final/run-003-packed

# Desde login: consultar cuota real. No entra en el contenedor PyTorch.
bash scripts/picasso/compact/quota.sh --root "$RUN_DIR" \
  --output "$PWD/local_results/picasso-quota.json"

# CPU, 8 h. Mantener el Parquet original en la raíz del repositorio.
bash scripts/picasso/compact/prepare.sh "$RUN_DIR" \
  --performance "$PWD/results_aggregated.parquet" \
  --trained /mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weights \
  --originals "$HOME/ner_lab/TFG_Aziz/models" \
  --checkpoint-fold 1 --layers-per-task 4 --part-mib 64 \
  --quota-file "$PWD/local_results/picasso-quota.json"

# Cuando prepare termine correctamente y exista manifest/plan.json:
bash scripts/picasso/compact/submit.sh --run-dir "$RUN_DIR" --concurrency 16
squeue -u "$USER"
```

Se envían arrays de hasta 500 índices, un finalizador CPU `afterany` y figuras CPU `afterany`.
Cada elemento reserva **1 GPU dgx, 8 CPU, 48 GB y 8 h**. Concurrencia 16 significa hasta
16 GPU por array; usar 1 para una sola GPU simultánea. No se inventan particiones ni cuentas.
`--no-figures` evita enviar las figuras; `--partition`, `--account`, `--constraint` son explícitos.
Todos los jobs usan `module load pytorch/2.10.0`. El envío usa `/usr/bin/python3` del host
(biblioteca estándar, Python >=3.6), porque el contenedor de PyTorch no contiene `sbatch`.
El parser elimina colores ANSI y exige un único ID inequívoco entre stdout/stderr.

`quota.sh` soporta salidas numéricas de `lfs quota` y `quota`. Si Picasso usa otra herramienta,
consultar sus cifras reales y crear el snapshot con **ambos** valores, sin inventarlos:

```bash
bash scripts/picasso/compact/quota.sh --root "$RUN_DIR" \
  --output "$PWD/local_results/picasso-quota.json" \
  --remaining-gib SALDO_REAL_GIB --remaining-inodes SALDO_REAL_INODOS
```

El snapshot caduca a las 24 h y puede actualizarse en la misma ruta sin cambiar la definición
científica. Los workers comprueban también espacio e inodos del filesystem. `--allow-unknown-quota`
es una excepción explícita para quien haya comprobado la cuota por otro medio; no convierte
el espacio libre del filesystem en saldo de usuario. El preflight reserva como margen lógico
8 GiB y 4.000 inodos, además del suelo libre. Es una comprobación, no una reserva real frente
a escrituras de otros jobs del mismo usuario. EDQUOT/ENOSPC detienen la tarea de forma reanudable.
La ocupación anterior de `run-002` sigue contando en la cuota: estos scripts no borran esos datos.

Comprobación (solo lee datos; exportar PYTHONPATH también al contenedor):

```bash
module load pytorch/2.10.0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH" APPTAINERENV_PYTHONPATH="$PYTHONPATH"
python -m weight_analysis.experiments.compact check --run-dir "$RUN_DIR" --verify
```

Consolidación o figuras manuales, sin volver a cargar pesos:

```bash
sbatch --chdir="$PWD" --output="$RUN_DIR/logs/finalize-%j.out" \
  --error="$RUN_DIR/logs/finalize-%j.err" scripts/picasso/compact/finalize.sbatch "$RUN_DIR"
sbatch --chdir="$PWD" --output="$RUN_DIR/logs/figures-%j.out" \
  --error="$RUN_DIR/logs/figures-%j.err" scripts/picasso/compact/figures.sbatch "$RUN_DIR"
```

Reintentar únicamente tareas incompletas tras el finalizador:

```bash
bash scripts/picasso/compact/retry.sh --run-dir "$RUN_DIR" --concurrency 16
```

Una respuesta exitosa no interpretable conserva el recibo y bloquea repeticiones; verificar
la cola antes de `submit.sh ... --recover`. El controlador conserva intención de envío antes
de llamar a sbatch, IDs de cada array/finalizador/figuras y ledgers de retry deterministas.
No se duplica un array si solo falló enviar el finalizador. Repetir un retry ya enviado es
idempotente. Si un proceso muere después de enviar pero sin recibo, conserva el bloqueo:
no se adivina que el job no existe.

SIGUSR1 se reenvía al worker cinco minutos antes del límite. El presupuesto de cálculo es
7h30. Se persisten bloques de estadísticas, espectros completados y unidades confirmadas;
los errores matemáticos/IO ordinarios se aíslan, mientras que presión de almacenamiento
detiene la tarea para no agravar la cuota. La finalización conserva tareas pendientes,
errores y exclusiones; no califica como completo un experimento parcial.

## Consulta y exportación de tablas específicas

PyArrow no requiere DuckDB ni los checkpoints:

```python
from weight_analysis.experiments.compact.query import read_geometry, joined_geometry

table = read_geometry(run_dir, filters={
    "family": "same_language_cross_entity", "backbone": "FacebookAI--xlm-roberta-base",
    "scope": "layer", "metric": "cosine", "source": "delta",
})
frame = joined_geometry(run_dir, filters={"scope": "encoder", "metric": "l2_distance"})
# Exportación deliberada del subconjunto de interés, no una vista duplicada por defecto:
frame.to_parquet("selected_entity_comparisons.parquet", compression="zstd", index=False)
```

DuckDB es opcional (`pip install '.[query]'` en un entorno apropiado, antes de los jobs):

```python
from weight_analysis.experiments.compact.query import duckdb_connection
db = duckdb_connection(run_dir)
db.sql("""SELECT backbone_a, language_a, entity_a, metric, value,
                 strict_f1_mean_a, strict_f1_std_a, strict_f1_mean_b, strict_f1_std_b, abs_f1_gap
          FROM weight_geometry_f1_joined
          WHERE scope='layer' AND source='delta' AND metric='cosine'""").show()
```

La misma consulta se puede ejecutar con `manifest/views.sql`. Las vistas por capa,
componente, embeddings y profundidad, y las selecciones best/worst, referencian los mismos
Parquet. Los payloads de estadísticas/fingerprints conservan arrays y valores singulares en
JSON verificable, accesible mediante `json_extract` de DuckDB. La columna `exactness` y el
estado del fingerprint distinguen valores exactos, truncados e indefinidos.
Si se copia el run a otra máquina/ruta, `write_views(nueva_ruta)` del módulo `query`
regenera las rutas de las vistas sin abrir checkpoints; las rutas del manifest son relativas.

## Coste y límites de validación

Lectura principal: aproximadamente una visita por tensor/checkpoint de cada tarea; ningún
par de modelos vuelve a abrir y recorrer pesos completos. Pares: O(M²P) reducciones para M
checkpoints compatibles y P coordenadas; Gram de actualización comparte trabajo entre
comparaciones, centroides y proyecciones. SVD exacta cuesta O(rc·min(r,c)) por matriz; los
valores singulares se reutilizan en todas sus métricas. El modo/umbral y su coste superior
se declaran antes del envío. Una reserva de ocho horas no garantiza que termine una tarea
con embeddings enormes o SVD exacta; sí permite conservar y reanudar sus unidades.

Las pruebas usan oráculos NumPy/directos para centroides, proyecciones, residuos, distancias,
Welford, invariancia de origen, fragmentos y regresión espectral. Incluyen interrupción entre
rename y commit, recuperación de bloques, repetición idempotente, corrupción Parquet,
integridad, conservación de filas, evidencia de vocabulario y límites de archivos/cuota.
Las pruebas CUDA requieren hardware real y se omiten explícitamente si no existe.

Validación local de esta entrega: **356 passed, 81 skipped**, suite completa en CPU;
Ruff pasa para `src` y las pruebas nuevas. Se comprobaron también el wheel instalado fuera
del checkout y el envío desde `/usr/bin/python3 -S` con scheduler simulado. Los jobs reales
y el hardware CUDA de Picasso no se han ejecutado desde este entorno.
