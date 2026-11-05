# Experimentos reproducibles en Picasso

El flujo se ejecuta desde WeightAnalysis, sin modificar `ner_lab`. Los adaptadores SLURM de
`scripts/picasso/` están autorizados por la petición posterior del usuario. El núcleo científico
no importa SLURM ni envía jobs. No se ha lanzado ningún job real desde este entorno.

## Archivo recibido y auditoría

Se ha leído `results_aggregated.parquet` completo: 459 filas, 16 columnas, siete idiomas,
tres entidades y 22 backbones. SHA256:
`f9d9aac50a7540c63fe3c38f77962297571c9e051e5af5f8fd5fb47afa5d4fac`.
El archivo original está excluido de Git junto con `local_results/`; no se ha cambiado su contenido.

Las columnas reales son `language`, `entity`, `model`, `n_folds`, `strict_f1_mean`,
`strict_f1_std`, `strict_precision_mean`, `strict_precision_std`, `strict_recall_mean`,
`strict_recall_std`, `char_f1_mean`, `char_f1_std`, `char_precision_mean`,
`char_precision_std`, `char_recall_mean`, `char_recall_std`.
`model` corresponde al nombre exacto del directorio del backbone. No se aplica fuzzy matching,
conversión de nombres HF, cambios de mayúsculas ni sustituciones automáticas de alias.

No hay nulos, filas repetidas ni duplicados de `(language, entity, model)`. `n_folds=4` en todas
las filas. Faltan las tres entidades de `alexyalunin--RuBioRoBERTa` para `nl`; no se crean filas
ni puntuaciones para ellas. Los tres F1 medios iguales a cero se conservan como resultados
observados. El archivo no suministra IDs de fold, semilla, configuración de entrenamiento ni
protocolo de evaluación. El informe JSON incluye esquema Arrow y metadatos completos, columnas
utilizadas, nulos, duplicados, cobertura, rangos, hash, ruta y versión de cálculo.

El usuario confirma que `models/` contiene **pretraining original** y `weights/` contiene **fold 1**.
La semilla se describe como «por defecto», sin número: se registra `seed=null`. Si se conoce,
pasar `--checkpoint-seed N` al preparar; nunca se supone que sea 42. El F1 mantiene el ámbito
`configuration_aggregate_unknown_fold_ids`: no son cuatro observaciones de los pesos de fold 1.
Las asociaciones son descriptivas y no producen valores p iid para pares dependientes.

La relación esperada por directorios se comprobará **en Picasso**. Este entorno no monta
`/mnt/fscratch/...`: no se afirma haber emparejado ni examinado los pesos reales. La auditoría
local está en `local_results/ner_source/source_audit.json`; los recuentos de correspondencias
quedan pendientes hasta `prepare` en destino.

## Preparar y lanzar

Desde la raíz del repositorio, enviar primero las pruebas como un archivo de batch:

```bash
sbatch scripts/picasso/tests.sbatch
```

El wrapper de sbatch de Picasso rechazó el envío por stdin en la prueba del usuario;
usar los archivos `.sbatch` del repositorio. Tanto las pruebas como el array GPU incluyen
`#SBATCH --constraint=dgx` y `#SBATCH --gres=gpu:1`, según la configuración confirmada
por el usuario. El job de pruebas reserva una hora y verifica CUDA antes de ejecutar pytest;
los logs son `wa-tests-<JOBID>.out` y `.err`. Continuar cuando termine sin fallos.
La preparación y el finalizador declaran `--constraint=cpu` y no reservan GPU.

Desde la raíz del checkout de WeightAnalysis en Picasso, colocar el Parquet original en esa raíz.
Mantener el mismo código durante todo el run. El directorio de salida pedido es **weigh_analysis_final**:

```bash
module load pytorch/2.10.0
export WEIGHT_ANALYSIS_PROJECT_DIR="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR=/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weigh_analysis_final/run-001

sbatch scripts/picasso/prepare.sbatch \
  --performance "$PWD/results_aggregated.parquet" \
  --trained /mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weights \
  --originals "$HOME/ner_lab/TFG_Aziz/models" \
  --run-dir "$RUN_DIR" --checkpoint-fold 1 --layers-per-task 4
```

La preparación usa CPU, hasta 8 horas, 8 CPU y 48 GB. Comprueba los archivos de cada modelo,
hashes completos, nombres, formas y configuración; registra tokenizer, `encoding.json` y evidencia.
Genera un manifiesto inmutable y **no envía arrays automáticamente desde un nodo de cálculo**.
Después de terminar ese job, revisar `plan.json`, `ner_source/source_audit.json` y los logs.
Si un modelo falta, se registra la incidencia y se preparan las unidades independientes disponibles.

Desde el nodo de login, tras existir `plan.json`:

```bash
bash scripts/picasso/submit.sh --run-dir "$RUN_DIR" --concurrency 1
```

Se envía un array GPU y un finalizador CPU con dependencia `afterany`. Cada elemento pide
**una GPU, 8 CPU, 48 GB y 08:00:00**. `--concurrency 1` ejecuta un elemento a la vez;
puede aumentarse si se quieren varias GPU simultáneas, una por elemento. Para restricciones
locales, añadir `--partition NOMBRE`, `--account CUENTA` o `--constraint TIPO_GPU`.
El constraint GPU por defecto es `dgx`; un argumento explícito puede sustituirlo.
No se inventa una partición de Picasso. La partición/constraint GPU no se reutiliza en el job CPU.
Para una cola CPU especial, lanzar `finalize.sbatch` manualmente con las opciones de esa cola.

Los arrays con más de 500 tareas se dividen y usan mapas ordinales propios para evitar depender
específicamente de `MaxArraySize`. El límite de concurrencia se aplica **por array**. En el diseño
actual, con 22 backbones de unas 12–24 capas, se espera normalmente un único array; el número
exacto se descubre en destino, sin suponer cuántos checkpoints existen.

Las dependencias Python deben estar disponibles en el módulo/entorno del job: torch >=2.10,
numpy >=2, pandas >=2.2,<3, scipy >=1.13, pyarrow >=17, safetensors >=0.4. Matplotlib >=3.8
solo se requiere para figuras. Los scripts añaden `src/` a PYTHONPATH, incluida su propagación
Singularity/Apptainer, y no instalan paquetes ni descargan modelos desde nodos de cálculo.
No se ha probado aquí el módulo específico de Picasso; la validación local usa PyTorch CPU.

## Qué se calcula

Cada tarea corresponde a un backbone y un bloque de capas. Los embeddings y parámetros
sin capa se separan para evitar repetirlos en cada fragmento. Los ítems del manifiesto incluyen:

| Familia | Universo y resultado |
| --- | --- |
| `checkpoint_vs_pretraining` | Cada checkpoint contra su original: drift, magnitudes, energías y fingerprints |
| `same_language_cross_entity` | Todas las parejas de entidades del mismo backbone/idioma |
| `same_entity_cross_language` | Todas las parejas de idiomas del mismo backbone/entidad |
| `cross_language_cross_entity` | Pares restantes del mismo backbone, etiquetados como cruce de tareas |
| `entity_mean_by_language` | Centroide de las tres entidades para cada backbone/idioma |
| `language_mean_by_entity` | Adaptación de cada entidad, compartida entre los siete idiomas, para cada backbone |
| `mean_by_backbone` | Centroide sobre las 21 celdas idioma × entidad del backbone |
| `cross_backbone_intrinsic` | Comparaciones entre backbones dentro de idioma/entidad a partir de fingerprints persistidos |

Los centroides describen actualizaciones, no generan modelos fusionados. Se obtienen norma y
energía de la dirección media, distancia a la media, fracción proyectada, coseno con signo,
residuo ortogonal, dispersión y leave-one-out. Cada celda tiene igual peso; una media de tres
entidades exige las tres y una media de 21 celdas exige las 21. La ausencia de F1 no impide
geometría de un checkpoint existente, pero no se inventa su rendimiento.

Los grupos incompletos o incompatibles se registran en `exclusions`; no se calculan como medias
de las celdas supervivientes. La ausencia de datos puede requerir corregir la entrada y preparar
un run nuevo; reenviar el mismo índice no crea un modelo ausente.

Los pares compatibles usan L2, RMSE, coseno, L2 normalizada, Pearson y acuerdo de signo.
Se calculan fuentes final y delta cuando procede. L2/RMSE invariantes a un origen común solo
se calculan una vez desde pesos finales. Drift usa energía, RMS, norma base/final y magnitud relativa.
Los fingerprints incluyen Frobenius, rangos y curvas/cuantiles normalizados. Se reutilizan
primitivas históricas y acumuladores de energía; nunca se promedian normas de tensores.

Los backbones distintos no se restan directamente. Se comparan TRACE, curvas espectrales,
cuantiles y perfiles de adaptación sobre datos ya calculados, con desgloses por componente y
profundidad relativa. La rejilla relativa no alinea neuronas. Se exige cobertura completa de
los fragmentos del checkpoint antes de usarlo en estos perfiles; no se extrapolan capas que
faltan por una tarea fallida. Las matrices truncadas no se presentan como espectros exactos;
las métricas indefinidas conservan su estado y el mínimo de componentes explícito.

Las referencias actuales son todas pretraining. «Baseline» y «pretraining» designan aquí el
mismo archivo: no se duplica un análisis equivalente. Una baseline fine-tuned distinta exige
otra referencia y procedencia explícita; la API de geometría la admite, pero no se inventa una.

El entrenamiento de cada modelo se vincula por la declaración del usuario y el layout exacto.
`ancestry_status` conserva que esa procedencia no se ha verificado independientemente. Un
manifiesto de evidencia opcional (`--evidence archivo.json`, lista de filas con language/entity/
backbone, fold, seed, initialization_id, provenance_id, tokenizer_id y datos de referencia)
permite registrar evidencia más precisa. Los hashes prueban qué archivos se leyeron, no reconstruyen
la historia de entrenamiento. Arquitectura, parametrización y vocabulario se comprueban además.

## Reanudación e incidencias

El límite de 8 horas es una **reserva máxima**, no una promesa de terminar todas las comparaciones
ni un tiempo mínimo. El worker deja de iniciar cálculos después de 7h30 y atiende la señal
USR1 de SLURM cinco minutos antes del límite. Las escrituras terminan dentro del margen disponible.
Se guardan unidades completadas en caché antes de pasar a la siguiente. Una interrupción fatal
solo puede preservar lo que ya se haya escrito atómicamente.

Errores ordinarios de archivo, tensor, memoria GPU o persistencia quedan registrados con
traceback y no cancelan los ítems independientes siguientes. Cambios de tamaño/mtime/ctime,
metadatos o directorio se detectan entre unidades, además de los hashes completos al validar
los checkpoints. No editar pesos, originales ni código durante el run.

El finalizador se ejecuta aun cuando haya fallado un elemento del array. Nunca marca como completo
un resultado parcial; conserva cobertura, incidencias e índices pendientes en
`derived/finalization.json`. Para reanudar desde login:

```bash
bash scripts/picasso/submit.sh --run-dir "$RUN_DIR" --concurrency 1 --retry
```

Los ítems completados se validan y omiten; los fallidos reutilizan las unidades de caché válidas.
Las comparaciones intrínsecas CPU tienen también marcadores con hashes de dependencias y salidas.
Si el finalizador alcanza su límite, puede repetirse sin recargar los pesos:

```bash
sbatch scripts/picasso/finalize.sbatch "$RUN_DIR"
```

Errores de configuración/SVD permanentes requieren corregir su causa. Cambiar tamaños de bloque,
modo espectral, capas o código exige otro run; no se mezclan resultados de distintas definiciones.
Seleccionar un tipo de GPU homogéneo mediante `--constraint` evita fragmentos de distintos
entornos. Si se mezclan dispositivos, el finalizador conserva agregados separados por ejecución
y señala cobertura parcial, sin afirmar que uno de ellos sea el encoder entero.

Si el Parquet obligatorio falta o cambia, se detienen **solo** las etapas basadas en F1.
La preparación/finalización geométrica independiente puede continuar y el informe declara el bloqueo.
`read_joined` rechaza vistas F1 bloqueadas para no consumir resultados obsoletos de una finalización anterior.

## Parquet y trazabilidad

```text
run-001/
  plan.json
  ner_source/                  # auditoría, rankings, score_checkpoint_links.parquet
  submissions/                 # mapas de array e IDs reales de sbatch
  logs/                        # stdout/stderr por job/array y prepare.jsonl
  tasks/00000-<backbone>/
    events.jsonl
    status.json
    cache/*.parquet
    work/<comparación>/<ID>/dataset/{tablas,manifest.json}
    work/<comparación>/<ID>/completion.json
  derived/
    finalization.json
    geometry/<comparación>/<ID>/{tablas,manifest.json}
    f1_*.parquet
    intrinsic_inputs.parquet
    weight_geometry_f1_joined/part-*.parquet
    weight_geometry_f1_by_layer/part-*.parquet
    weight_geometry_f1_by_component/part-*.parquet
    weight_geometry_f1_by_language/language-<valor>/part-*.parquet
    weight_geometry_f1_by_entity/entity-<valor>/part-*.parquet
    weight_geometry_f1_by_backbone/backbone-<valor>/part-*.parquet
  figures/
```

Los seis productos de geometría–F1 son datasets de múltiples Parquet, para no crear un archivo
monolítico. Las vistas por idioma/entidad/backbone organizan las mismas observaciones con sus
identidades de ambos extremos; no introducen promedios científicos implícitos. Un par que cruza
idiomas conserva sus dos idiomas y una etiqueta de cruce en la dimensión común.

Las tablas finales conservan modelo lógico, artefacto físico, hash/score ID, backbone, idioma,
entidad, fold y semilla, F1 medio/SD y ámbito, ambos extremos de pares, gap absoluto, comparación,
referencia, fuente, métrica, capa, componente, tensor, ámbito geométrico y cobertura.
El manifiesto conserva ruta/hash/esquema del original, correspondencias exactas, número de modelos
emparejados y sin correspondencia, opciones, código y evidencia. Las estadísticas suficientes y
de grupos se conservan para recalcular agregaciones desde Parquet, sin leer tensores.

Los rankings observados producen:

| Archivo | Filas en el archivo recibido |
| --- | ---: |
| `f1_model_ranking.parquet` | 459 |
| `f1_ranking_by_language.parquet` | 153 |
| `f1_ranking_by_entity.parquet` | 63 |
| `f1_ranking_by_backbone.parquet` | 21 |
| `f1_best_worst_pairs.parquet` | 21 |
| `f1_top_bottom_pairs.parquet` | 189, k=3 |
| `f1_all_valid_pairs.parquet` | 4.788 |

El ranking de cada idioma/entidad usa el F1 original. Los rankings globales de idioma y entidad
son **macro medias de tareas completas**, con pesos iguales. El ranking global de backbone exige
las 21 celdas. No se calcula la SD de esa macro media a partir de SD marginales: falta la covarianza
entre folds/tareas. Las SD originales se conservan junto con sus score IDs. El modelo con celdas
neerlandesas ausentes se excluye de las macros que necesitan esos datos; sí aparece donde la
cobertura es completa. No se rellenan posiciones con rankings ficticios.

`f1_model_ranking` conserva además rangos por idioma-backbone, entidad-backbone e
idioma-entidad-backbone. Los cruces de tareas son descriptivos; el último tiene rango 1 porque
solo existe un agregado por celda, sin variantes de configuración identificadas. F1 descendente,
rango denso y hash de identidad ascendente resuelven empates; top-k contiene exactamente k cuando
hay suficientes modelos. Best–worst se elige sobre la tabla F1 completa, antes de saber qué geometrías
terminarán: no se sustituye el extremo ausente por el siguiente que haya completado.

Leer y filtrar sin pesos:

```python
from weight_analysis.experiments.exports import read_joined
frame = read_joined(RUN_DIR, filters={"comparison_type": "same_language_cross_entity"})
by_layer = read_joined(RUN_DIR, table="weight_geometry_f1_by_layer")
```

## Figuras

Después de finalizar:

```bash
python -m weight_analysis.experiments figures --run-dir "$RUN_DIR"
```

Genera heatmaps de todos los modelos y best–worst por idioma/entidad, heatmaps de celdas por
backbone, perfiles por capa y componente frente al pretraining y scatter distancia–gap de F1.
Las etiquetas de modelos contienen F1/SD cuando están disponibles. Las figuras intrínsecas
usan escalas comunes por métrica entre paneles, orden F1 explícito y referencia «pretraining
por modelo»; los IDs particulares de referencia permanecen en Parquet. No se mezclan entidades
para fabricar un heatmap global. `figures.json` registra filtros, escalas, archivos e incidencias.
No se recargan checkpoints ni se agregan valores ocultamente en las funciones de dibujo.

## Coste y validación

Con 21 celdas por backbone hay 210 pares directos, 21 descripciones y 11 grupos compartidos.
Cada bloque de capas se distribuye como una tarea; embeddings y parámetros sin capa son tareas
separadas. Todos los pares cuestan O(n²P) y pueden estar limitados por lectura de disco aunque
las reducciones estén en GPU. Hashes, Parquet, preparación y finalización usan CPU. La SVD usa
la matriz completa, incluso en modo truncado: `block_elements` limita reducciones vectoriales
y grupos, no el workspace de una SVD. Los resultados detallados y sus vistas pueden ocupar
mucho más que los resúmenes; comprobar cuota en scratch antes de lanzar el grid completo.

No se dispone aquí de pesos reales, GPU ni SLURM de Picasso para medir tiempos o VRAM. Ajustar
`--layers-per-task` y concurrencia tras el primer run de prueba; la reanudación evita perder
unidades terminadas si un shard necesita varios turnos de 8 horas.

`tests/test_picasso_experiments.py` cubre esquema/SD/agregación, rechazo de duplicados, rankings
deterministas, cobertura, manifiestos, ejecución sintética CPU, unión F1, finalización sin lectura
de pesos, figuras, interrupciones, reanudación, mutaciones de archivos, bloqueo F1 y sbatch simulado.
`bash -n` valida los adaptadores; ningún test envía jobs reales. La prueba marcada `gpu` se omite
sin CUDA y permite verificar el worker con GPU real en destino.
