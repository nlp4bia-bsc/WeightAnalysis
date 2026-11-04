# Geometría comparativa fina

Esta extensión es independiente de `ner_lab` y conserva las funciones históricas. No entrena,
no usa activaciones ni F1 para calcular pesos y no implementa scheduling. El catálogo previo y
su reutilización están en [functionality_catalog.md](functionality_catalog.md), la referencia
arquitectónica verificada en [architecture_reference.md](architecture_reference.md).

## Existente y ampliado

| Capacidad | Primitiva reutilizada | Ampliación |
| --- | --- | --- |
| Lectura y metadatos | `WeightReader`, nombres canónicos, `CheckpointSpec`, hashes | Identidad de artefacto independiente de ruta, evidencia de inicialización y procedencia |
| Distancias y drift | `PAIR_METRICS`, `DRIFT_METRICS` y definiciones históricas | Bloques contiguos, agregación estable por coordenadas, selección de embeddings |
| Espectros y alineamiento | `spectral_metrics`, `principal_basis`, `principal_angles`, `topk_overlap` | Perfiles finos y fuentes final/delta, estado exacto/aproximado/indefinido |
| Backbones diferentes | `compare_backbones`, TRACE, curvas espectrales, cuantiles | Ruta intrínseca seleccionable por F1, componentes y profundidad relativa |
| F1 | Correlaciones descriptivas existentes | Validación estricta, ámbito de score, selección determinista, pares congelados |
| Especialización | Primitivas tensoriales | Centroide, proyección, residuo, dispersión y leave-one-out compartidos entre idiomas/entidades |
| Resultados | `AnalysisResult`, Parquet con digest | Tablas normalizadas, particiones, manifiesto, reunión determinista y caché por unidad |
| Figuras | Matplotlib opcional | Heatmaps, perfiles, referencias y scatter sin agregación implícita |
| Ejecución | Resolución explícita de dispositivo | Continuación por unidad, prints, JSONL, trazas y tablas de incidencias/eventos |

## Contratos y selección científica

```python
from weight_analysis import (
    CheckpointSpec, CheckpointEvidence, ReferenceSpec, ParameterSelection,
    input_digest, build_catalog, select_performance_pairs, compare_selected,
    aggregate_geometry, RunReporter, write_dataset,
)

checkpoint = CheckpointSpec(
    "bert", "es", "disease", "/datos/modelo", fold=1, seed=7,
    checkpoint_id="epoch-3",
)
evidence = CheckpointEvidence(
    initialization_id=input_digest("/datos/pretraining"),
    provenance_id="sha256-del-manifiesto-de-entrenamiento",
    tokenizer_id="sha256-vocabulario-y-orden-de-tokens",
    training_run_id="es-disease-fold1-seed7",
    configuration_id="bert-es-disease-configA",
)
reporter = RunReporter(log_path="/resultados/events.jsonl")
catalog = build_catalog([checkpoint], {checkpoint.uid: evidence}, reporter=reporter)
```

`build_catalog` necesita todos los candidatos del experimento; el ejemplo mínimo anterior solo
muestra un registro. La evidencia se obtiene del entrenamiento: la biblioteca no puede demostrar
una genealogía a partir de formas. `initialization_id` debe ser el digest del checkpoint original
real. Un hash de manifiesto identifica evidencia externa, no demuestra que esta sea verdadera.
Las claves de parámetros y campos estructurales de configuración determinan `parameter_space_id`.
Distintos modelos con el mismo tamaño y otra inicialización no se restan directamente.

La tabla F1 requiere `score_id`, `score_scope`, `strict_f1_mean`, `strict_f1_std`, `protocol_id`,
`evaluation_id`. En ámbito `checkpoint`, requiere el `artifact_id` exacto obtenido del catálogo.
Se exige media en [0,1] y desviación estándar finita no negativa; `scale="percent"` convierte
ambas explícitamente en la selección. No se calcula F1 desde predicciones ni se inventan
réplicas: una SD cero solo debe declararse si corresponde al protocolo realmente usado.

Para `score_scope="configuration_aggregate"`, requiere `configuration_id`, listas `folds` y
`seeds`, y sus recuentos `n_folds`, `n_seeds`. Es obligatorio pasar
`score_alignment={artifact_id: score_id}` y el checkpoint debe pertenecer a esa configuración,
fold y semilla. El agregado conserva su ámbito en rankings y extremos; no se convierte en el
rendimiento observado de un checkpoint. Las dimensiones opcionales de un score individual
se contrastan con el catálogo. Debe seleccionarse un protocolo antes de vincular varios scores
al mismo artefacto. Los scores ausentes generan exclusiones explícitas.

```python
selection = select_performance_pairs(
    checkpoints=catalog, performance=performance_df,
    strategy="best_worst", tie_policy="stable_id", k=3,
)
```

Estrategias: `best_worst`, `best_all`, `worst_all`, `top_bottom`, `all_pairs`. La selección por
rendimiento compara dentro de idioma, entidad, fold, semilla, protocolo y evaluación.
Por defecto puede ordenar varios backbones dentro de ese estrato; añadir `backbone` a `strata`
restringe también la arquitectura. Quitar dimensiones exige `allow_mixed_strata=True` y
`experimental_policy="justificación y protocolo del cruce"` explícita. Cada cambio queda en la procedencia.

Orden: F1 descendente, después `artifact_id` ascendente. `f1_rank` es rango denso; `selection_order`
resuelve empates de forma reproducible. `include_all` incluye todo el empate del límite y
rechaza cohortes superior/inferior solapadas. Top-k/bottom-k exige al menos 2k candidatos.
Estratos insuficientes se excluyen con motivo. Los pares son no ordenados y únicos; la tabla
retiene qué extremo rinde más, empates, F1/SD de ambos, gap absoluto y criterios de selección.
Seleccionar extremos sobre estos mismos F1 produce `descriptive_post_selection`.

Para comparar entidades o idiomas distintos, sin ordenar tareas diferentes por F1:

```python
from weight_analysis import select_axis_pairs
pairs = select_axis_pairs(
    checkpoints=catalog, axis="entity", member_ids=selected_artifact_ids,
    performance=performance_df,  # opcional
)
```

Se fija backbone, idioma (o entidad), fold, semilla, parametrización e inicialización.
Debe seleccionarse exactamente un checkpoint por valor del eje. Ninguna función promedia
semillas, folds o entidades para elegir ese representante. Para otro diseño, preparar
explícitamente los miembros antes de invocar la operación.

## API de cálculo y fragmentos

```python
from weight_analysis.configuration import MetricOptions

result = compare_selected(
    selection=selection,
    references={"bert": ReferenceSpec("/datos/pretraining", tokenizer_id="vocab-id")},
    parameters=ParameterSelection(
        layers=(0, 1, 2), components=("q", "k", "v", "o", "ffn_up", "ffn_down"),
        embeddings=("token", "position"),
    ),
    metrics=("l2_distance", "cosine", "pearson", "normalized_l2"),
    drift_metrics=("relative_l2", "delta_energy"),
    spectral_metrics=("stable_rank", "spectral_energy_curve"),
    intrinsic_metrics=("trace_distance_delta", "curve_l1_delta", "quantile_w1_final"),
    sources=("final", "delta"),
    options=MetricOptions(spectral_mode="auto", max_svd_elements=3_000_000),
    device="gpu", block_elements=100_000,
    reporter=RunReporter(log_path="/resultados/events.jsonl"),
    on_error="continue", cache_dir="/resultados/cache", resume=True,
)
aggregated = aggregate_geometry(result, scopes=("layer_component", "layer", "component", "encoder"))
write_dataset(aggregated, "/resultados/estudio")
```

Solo se ejecutan las familias solicitadas. La selección de parámetros usa metadatos antes de
leer pesos. Las componentes son las del catálogo histórico (`q`, `k`, `v`, `o`, `ffn_up`,
`ffn_down`, `ffn_gate`, `norm`, `other`); las cabezas NER se excluyen. `embeddings=()` las excluye;
`token`, `position`, `token_type`, `other` o `"*"` las seleccionan independientemente de `layers`.
`tensors=(clave_canónica, ...)` añade un filtro exacto. Un filtro sin parámetros se registra
como error de unidad. Las métricas espectrales se aplican solo a matrices.

En coordenadas compatibles se usan métricas directas. En parametrizaciones/inicializaciones
diferentes solo se usan fingerprints intrínsecos; nunca se restan tensores entre backbones
incompatibles. Los componentes ausentes dan resultados indefinidos o incidencias visibles.
Los perfiles intrínsecos interpolan las capas en una rejilla relativa de 16 posiciones;
esto no alinea neuronas. Los agregados intrínsecos conservan la definición histórica, incluido
el mínimo explícito de componentes (`min_fingerprint_components`), y se etiquetan con su ámbito.

Congelar `selection` antes de fragmentar. `pair_ids=[...]` ejecuta un subconjunto de pares;
`ParameterSelection` puede fragmentar capas o componentes. Para reunir:

```python
from weight_analysis import merge_comparative, read_dataset
combined = aggregate_geometry(merge_comparative([part_a, part_b]))
restored = read_dataset("/resultados/estudio", filters={"language": "es"})
```

La reunión ordena las coordenadas canónicamente, deduplica resultados idénticos y rechaza
reintentos en conflicto. La agregación se hace **después** de reunir estadísticas suficientes
si se necesita equivalencia con el cálculo completo. No sumar normas agregadas previamente.
El ámbito `encoder` significa todas las coordenadas seleccionadas y disponibles; puede incluir
embeddings si se solicitaron. `n_parameters`, estadísticas, incidencias y procedencia permiten
comprobar cobertura: un resultado parcial no se presenta como el encoder completo.

## Matemáticas y agregación

Para vectores alineados de coordenadas seleccionadas, sean A, B, D=A−B y N su dimensión:

- Energía de diferencia: E=ΣD²; distancia L2=√E; RMSE=√(E/N).
- Distancia L1 media=Σ|D|/N; máximo= max |D|. Las normas finales son √ΣA² y √ΣB².
- Coseno=ΣAB/(√ΣA²√ΣB²); Pearson usa sumas centradas y combinación de Chan para medias,
  varianzas y covarianzas. No se promedian cosenos ni correlaciones de tensores.
- Las métricas relativas y acuerdos de signo mantienen denominadores y umbrales históricos.
  `VectorMoments.value` conserva las definiciones atómicas de `PAIR_METRICS` y `DRIFT_METRICS`.
- Energías, productos, sumas absolutas y recuentos son aditivos. Medias, correlaciones,
  cosenos, RMS y fracciones necesitan estadísticas suficientes. Máximos se combinan por máximo.
- Rangos, entropías, cuantiles, curvas, top-k y subespacios no se agregan promediando tensores.
  Se conservan al nivel calculado. Los perfiles intrínsecos tienen su propia definición explícita.
- Histogramas opcionales usan `histogram_edges` fijos y crecientes; conservan recuentos,
  valores por debajo/encima y coordenadas de bloque. Solo sumar bins con bordes idénticos.

La acumulación es float64, incluso si el checkpoint es float16/bfloat16. El orden de reducción
es canónico para los mismos bloques. Cambiar el tamaño de bloque o el dispositivo puede cambiar
el último bit; la identidad de ejecución registra ambos. El umbral EPS histórico define cuándo
una razón o un coseno queda indefinido; no se reemplazan NaN por cero.

## Entidades e idiomas: una misma operación

```python
from weight_analysis import analyze_shared_adaptation, member_performance_view
shared = analyze_shared_adaptation(
    checkpoints=catalog, axis="entity", references=references,
    member_ids=selected_artifact_ids,
    parameters=ParameterSelection(components=("q", "v")),
    performance=performance_df,  # opcional; no entra en la geometría
    metrics=("mean_energy", "dispersion", "distance_to_mean", "projection_fraction",
             "direction_residual_energy", "loo_distance_to_mean"),
    device="gpu", reporter=RunReporter(log_path="/resultados/shared.jsonl"),
)
shared = aggregate_geometry(shared)
f1_view = member_performance_view(shared, table="geometry_aggregated")
```

`axis="entity"` fija backbone e idioma; `axis="language"` fija backbone y entidad.
Ambos fijan fold, semilla, parametrización e inicialización. `member_ids` define el universo
congelado y debe contener un único checkpoint por valor. Todas sus actualizaciones deben usar
una referencia compartida y, para embeddings, evidencia de vocabulario idéntico.

Para n miembros, Δᵢ=Wᵢ−R y μ=(ΣΔᵢ)/n:

| Métrica | Definición |
| --- | --- |
| `mean_energy`, `mean_norm` | ‖μ‖², ‖μ‖ |
| `update_energy` | ‖Δᵢ‖² |
| `centered_energy`, `distance_to_mean` | ‖Δᵢ−μ‖², ‖Δᵢ−μ‖ |
| `dispersion` | (Σ‖Δᵢ−μ‖²)/n; dispersión poblacional del universo declarado |
| `projection_coefficient` | (Δᵢ·μ)/‖μ‖², con signo |
| `projection_fraction` | (Δᵢ·μ)²/(‖Δᵢ‖²‖μ‖²) |
| `direction_residual_energy` | ‖Δᵢ−[(Δᵢ·μ)/‖μ‖²]μ‖² |
| `alignment_cosine` | (Δᵢ·μ)/(‖Δᵢ‖‖μ‖) |
| `loo_distance_to_mean` | ‖Δᵢ−μ₋ᵢ‖, μ₋ᵢ=(Σⱼ≠ᵢΔⱼ)/(n−1) |
| `loo_projection_fraction`, `loo_alignment_cosine`, `loo_direction_residual_energy` | Las anteriores usando μ₋ᵢ |

Energía y norma se ofrecen como representaciones explícitas: la primera es aditiva y permite
reunir fragmentos; la segunda expresa distancias en las unidades de los pesos. La fracción
proyectada equivale al coseno al cuadrado cuando está definida, mientras el coseno conserva
el signo. No constituyen evidencias independientes ni se recalculan como métricas distintas.

Las distancias al centroide y dispersión se calculan desde pesos finales para conservar
exactamente su invariancia al origen compartido. Para agregar, se suman energías y productos
por miembro **antes** de formar proyecciones, cosenos y raíces. Se verifica la presencia de
todos los miembros y se rechazan tensores repetidos. La identidad media de energía satisface
(Σ‖Δᵢ‖²)/n=‖μ‖²+dispersion. El centroide vectorial está disponible en la primitiva pública
`weight_analysis.metrics.update_geometry`; no se persiste un checkpoint combinado.
La propia inclusión en μ puede aumentar la alineación: LOO la describe explícitamente.
Una dirección cero genera resultados indefinidos. Residuos cerca del redondeo se marcan
`roundoff_limited`. Todas estas cantidades son descriptivas, sin interpretación causal.

## Referencias y cambio de origen

`ReferenceSpec(kind="pretraining")` contrasta el contenido de la referencia con
`CheckpointEvidence.initialization_id`. Para una baseline:

```python
baseline = ReferenceSpec(
    "/datos/baseline", kind="finetuned_baseline",
    initialization_id=original_digest,
    provenance_id="baseline-training-manifest", tokenizer_id="vocab-id",
    fold=1, seed=7, scope="matched",
)
```

Una baseline debe compartir inicialización y procedencia documentada. `scope="matched"`
exige fold/semilla iguales; `scope="shared_declared"` declara deliberadamente una referencia
común a varias réplicas. Se comprueban campos de arquitectura, claves y formas del encoder;
igualdad de formas por sí sola es insuficiente. `allow_unverified_origin=True` permite ausencia
de evidencia de entrenamiento, no inventa una inicialización ni relaja incompatibilidad
geométrica. El manifiesto y el catálogo conservan la evidencia disponible.

| Cantidad | Cambio de origen compartido R→R+C |
| --- | --- |
| L2, RMSE y L1 entre dos modelos | Invariante: (Wᵢ−R)−(Wⱼ−R)=Wᵢ−Wⱼ |
| Distancia al centroide, dispersión y distancia LOO | Invariantes |
| Norma/energía de actualización y dirección media | Cambian |
| Coseno, proyecciones y residuo ortogonal respecto de la dirección media | Pueden cambiar |
| Fingerprints espectrales de actualizaciones | Pueden cambiar |
| Geometría de pesos finales | Independiente de la referencia |

Las cantidades invariantes se registran con `source="final"`, referencia nula e identidad
compartida. Reunir dos análisis con distintas referencias conserva una sola observación de
estas cantidades. Las medidas de actualizaciones usan `source="delta"` y referencia explícita.
No se crea una métrica adicional para una distancia matemáticamente equivalente.

## Parquet normalizado: comparative-1

Cada tabla es un DataFrame, sin escritura obligatoria. Claves SHA256 identifican contenido y
criterios; `definition_id="fine-grained-1"` identifica la versión matemática. Las columnas
opcionales ausentes pueden no existir en tablas vacías. IDs y etiquetas son strings; valores
son float64, capas/recuentos enteros anulables, listas son listas Arrow. El lector conserva
los datos y metadatos originales y comprueba digest y checksum.

| Tabla | Clave | Columnas principales |
| --- | --- | --- |
| `checkpoints` | `artifact_id` | content/parameter-space ID, backbone, idioma, entidad, fold, semilla, snapshot, inicialización, tokenizer y procedencia |
| `performance` | `score_id` | Ámbito, F1 medio/SD, protocolo, evaluación, artefacto o configuración, folds/semillas declarados |
| `rankings` | `ranking_id` | selección, estrato, artefacto, score, rango F1 y orden determinista |
| `selection_pairs` | `selection_pair_id` | selección/comparación, extremos, dominio, F1/SD/scope, gap y criterio descriptivo |
| `comparisons` | `comparison_id` | extremos canónicos, dominio y dimensiones comunes o etiquetas de cruce |
| `references` | `reference_id` | tipo, digest, inicialización, procedencia, fold/semilla y ámbito |
| `groups` | `group_id` | eje, dimensiones fijas, universo y número de miembros |
| `group_members` | `membership_id` | grupo, artefacto, dimensiones y F1 opcional con su ámbito |
| `geometry_detailed` | `observation_id` | sujeto/tipo, métrica, valor, fuente/referencia, capa/componente/tensor/embedding, ámbito, estado, versión y ejecución |
| `geometry_aggregated` | `observation_id` | Los anteriores, N seleccionado y regla explícita de agregación |
| `sufficient_statistics` | `statistic_id` | sujeto, tensor, inicio/fin de bloque, N, energías/productos, medias/M2/covarianza, signos y métricas solicitadas |
| `shared_statistics` | `shared_statistic_id` | grupo, miembro, tensor, N, energías, productos con media y LOO, ejecución y métricas solicitadas |
| `profile_points` | `point_id` | checkpoint, tensor, fuente, métrica, posición normalizada, valor y estado de método |
| `histograms` | `histogram_id` | sujeto/tensor/bloque, edges, counts, below y above |
| `exclusions` | `exclusion_id` | selección, candidato y motivo |
| `issues` | `issue_id` | unidad, etapa/contexto, timestamp, tipo, mensaje y traceback |
| `events` | `event_id` | timestamp, estado, unidad, contexto y tiempo transcurrido |
| `associations` | `association_id` | estrato/ámbito/métrica, correlación descriptiva y recuentos de pares, checkpoints y scores |

`write_dataset` particiona cada tabla por las dimensiones disponibles de
`("language", "entity", "backbone", "comparison_type")`; se puede cambiar `partition_by`.
Los resultados geométricos incluyen esas dimensiones; los índices relacionales globales
se escriben sin una partición que no les corresponda. Un filtro en `read_dataset` selecciona
particiones de tablas que disponen de esa dimensión y conserva índices globales.
No es un join científico ni una agregación. Los archivos son inmutables por contenido y
procedencia, comprimidos y con escritura atómica; `manifest.json` se sustituye al terminar.
No se confían archivos sueltos que no figuren en el manifiesto. La escritura completa puede
fallar por disco/permisos: el objeto en memoria sigue disponible para reintentar.

La caché por unidad depende de hashes de pesos/referencia, coordenadas, fuente, versión,
dispositivo, hardware y opciones numéricas; F1 no entra en su identidad. Un cache corrupto se
registra y recalcula. Solo se guardan unidades terminadas. Verificar hashes requiere leer
archivos en CPU; se hace una vez por entrada/referencia durante una ejecución. Los checkpoints
deben permanecer inmutables durante ella. Reanudar con otros pesos invalida la reutilización.

## F1 y figuras

`comparison_view(result, table="geometry_aggregated")` vincula geometría con extremos F1.
Si hay varias selecciones reunidas, requiere `selection_id`. `associate_f1` da correlaciones
descriptivas con número de pares, checkpoints y scores únicos. `associate_member_f1` relaciona la geometría de cada miembro con F1 dentro del grupo declarado.
No genera valores p ni intervalos
iid para pares que comparten checkpoints. Los datos ausentes o constantes no se convierten en
asociaciones significativas. Para inferencia confirmatoria hace falta un protocolo externo
preespecificado con replicación y dependencia tratadas explícitamente.

Importar las figuras desde `weight_analysis.visualization`; consumen exclusivamente tablas:

| Figura | Función | Selección requerida |
| --- | --- | --- |
| Modelo × modelo, best–worst o cohortes | `plot_pair_heatmap(axis="artifact")` | Un estrato y una observación por par; `order` desde rankings F1 |
| Entidad × entidad | `plot_pair_heatmap(axis="entity")` | Pares de `select_axis_pairs`, backbone/idioma/fold/seed fijos |
| Idioma × idioma | `plot_pair_heatmap(axis="language")` | Pares de `select_axis_pairs`, backbone/entidad/fold/seed fijos |
| Profundidad absoluta o relativa | `plot_geometry_profiles(x="layer" / "relative_depth")` | Una observación por posición y serie |
| Componentes | `plot_geometry_profiles(x="component")` | Agregación de componente explícita |
| Pretraining/baseline | `plot_reference_panels` | Tablas de ambas referencias y `ColorScale` común |
| Miembro frente a F1 con su SD | `plot_member_f1` | Vista de miembros, un grupo y una observación por miembro |
| Distancia frente a gap de F1 | `plot_f1_gap` | Vista F1, una observación por par, sin ajuste inferencial |

Ejemplo: filtrar primero `scope="encoder"`, métrica, fuente, referencia y estrato; pasar
`order=rankings.sort_values("selection_order").artifact_id` para ordenar por F1. No hay
`pivot_table`, medias ocultas ni diagonales inventadas. Repetir un par o una posición requiere
agregación explícita previa. `ColorScale(metric, minimum, maximum)` fija escalas comparables.
Las funciones devuelven Axes/Figure; guardar imágenes es decisión del consumidor.

## Incidencias, GPU y coste

`on_error="continue"` es el valor predeterminado de las rutas nuevas. Un archivo ausente,
referencia incompatible, tensor inválido, error numérico o fallo de escritura de caché se
registra y no cancela las unidades independientes siguientes. Los resultados parciales tienen
`complete=False` e incidencias con traceback. Una selección inválida globalmente (p. ej., una
métrica desconocida o un diseño ambiguo) lanza error antes de ejecutar. `on_error="raise"`
permite depurar deteniéndose en la primera incidencia. KeyboardInterrupt/SystemExit no se
ocultan. Un proceso terminado, fallo fatal de CUDA o fallo de máquina no puede recuperarse en
el mismo proceso; las unidades persistidas permiten reanudar.

`RunReporter(verbose=True)` imprime progreso, inicio, fin, caché, errores y resumen con flush.
`log_path` habilita JSONL append con flush por evento; el logger
`weight_analysis.experiments` permite añadir handlers en la aplicación sin modificar logging
global. Un fallo del archivo de log aparece en consola/logger y en `events`; no pierde el
resultado numérico. Compartir un reporter conserva su historial acumulado, por lo que sus
resúmenes e incidencias cubren ese historial; usar uno por llamada si se necesita aislar runs.
Los eventos contienen timestamps y no se espera que su digest sea igual entre reejecuciones.

GPU CUDA es el dispositivo por defecto. No hay fallback silencioso: si falta CUDA, la
prevalidación queda registrada y la llamada nueva devuelve un resultado con incidencias;
usando `on_error="raise"` lanza el error. CPU se elige con `device="cpu"`. Reducciones de pesos,
centroides, proyecciones, cuantiles, SVD y perfiles compatibles se ejecutan en el dispositivo
pedido; hashes, lectura de archivos, pandas, Parquet, logging y figuras usan CPU.

| Trabajo | Tiempo orientativo en operaciones | Memoria de trabajo |
| --- | --- | --- |
| Catálogo/verificación de n checkpoints de P parámetros | O(nP), limitado por lectura/hash | Metadatos y buffers de archivo |
| Geometría de p pares | O(pP) | O(B) tensores float64, B=`block_elements` |
| Grupos con n miembros y LOO | O(nP), sin n análisis adicionales de pesos | O(nB), varios buffers |
| SVD exacta de matriz r×c | O(rc·min(r,c)) | Matriz completa y workspace del backend |
| SVD truncada de rango k | Iterativa, aprox. O(rc·k·iteraciones) | Matriz completa y bases de rango k |
| Cuantiles/fingerprints | Dependiente de tamaño y algoritmo de cuantiles | Matriz completa en esta ruta |
| Todos frente a todos | p=n(n−1)/2 por estrato | Resultados/estadísticas O(p·número de bloques) |

El bloque predeterminado de 100k elementos ocupa 0.8 MB por vector float64; un grupo de n
miembros necesita varios múltiplos de 0.8n MB más matrices intermedias. No es una estimación
exacta del pico CUDA. Los espectros/subespacios usan matrices completas y pueden agotar memoria;
no hay benchmark de velocidad ni pico medido en este entorno sin GPU. Solicitar solo familias
necesarias, restringir parámetros y guardar unidades reduce coste. Distancias invariantes y
estadísticas vectoriales se reutilizan; no se crea un segundo resultado solo por cambiar R.

## Validación y orden de uso recomendado

Pruebas: `tests/test_comparative_extension.py`. Oráculos NumPy independientes contrastan
centroides, productos, proyecciones, residuos, dispersión y LOO. Se verifican métricas atómicas
históricas, Pearson con grandes offsets, invariancia al origen, empates deterministas,
reubicación de pesos, ámbitos de F1, bloqueado y no bloqueado, fragmentos de pares/capas/métricas,
Parquet particionado, caché corrupta, incompatibilidad, errores de log, prints y continuación.
Las pruebas GPU reales cubren comparación directa/intrínseca y ambos ejes, y se omiten cuando
`torch.cuda.is_available()` es False. No se simula una ejecución GPU aprobada.

1. Registrar candidatos/evidencia y validar F1.
2. Congelar selección y criterios antes de distribuir fragmentos.
3. Calcular geometría vectorial y guardar unidades; añadir solo los espectros necesarios.
4. Calcular entidades/idiomas con un universo explícito por grupo.
5. Reunir estadísticas, agregar explícitamente y comprobar incidencias/cobertura.
6. Relacionar F1 y dibujar vistas filtradas; guardar el dataset y su procedencia.

Ejemplo ejecutable sin red: [comparative_geometry.py](../examples/comparative_geometry.py).
