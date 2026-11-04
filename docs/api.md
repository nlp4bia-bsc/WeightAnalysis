# API pública y catálogo de operaciones

## Contratos y capas

`weight_analysis` es el namespace instalado; `weightspace` conserva interfaces históricas.
La biblioteca nueva no importa el paquete histórico ni `ner_lab`.

| Paquete | Responsabilidad |
| --- | --- |
| `core` | Identidad explícita de checkpoints, lectura local, configuración del modelo, parámetros y alineación |
| `metrics` | Primitivas matemáticas independientes de pipelines y persistencia |
| `analyses` | Drift/fingerprints y descomposición factorial |
| `comparisons` | Comparación de pesos alineados, perfiles entre backbones y centralidad descriptiva |
| `statistics` | Asociaciones F1 descriptivas; no inferencia causal ni independencia ficticia de pares |
| `results` | `AnalysisResult`, serialización validada y agregación de unidades |
| `configuration` | `MetricOptions`, parámetros numéricos tipados sin rutas de experimentos |
| `execution` | Etapas selectivas de composición, sin procesos ni scheduling |
| `visualization` | Gráficos opcionales desde resultados ya calculados |

Los exports principales son `CheckpointSpec`, `analyze`, `AnalysisResult`, `write_result`,
`read_result` y `merge_results`. Las primitivas se importan desde `weight_analysis.metrics`;
`MetricOptions`, desde `weight_analysis.configuration`. Los módulos de menor nivel conservan
funciones históricas para regresión; no todas implementan la selección fina de `analyze`.

`CheckpointSpec(backbone, language, entity, path, fold=1, seed=None, checkpoint_id=None)`
declara las dimensiones del experimento. No infiere fold, semilla o idioma del nombre de carpeta;
no impone el vocabulario MultiClinAI de la búsqueda histórica. `uid` identifica la declaración
completa, incluida la ruta resuelta. Cambiar la ubicación cambia el identificador, aunque los
pesos sean idénticos. El hash de contenidos se registra por separado al calcular una unidad.

El lector admite `model.safetensors`, su índice y shards, `pytorch_model.bin` o su índice.
La normalización de claves usa el `model_type` para el prefijo del encoder. Las cabezas NER y los
buffers enteros no se restan como parámetros del encoder. Coincidir en forma no demuestra que
dos modelos compartan el mismo origen: el usuario debe declarar el original correcto.

## Selección y catálogo

La entrada principal es una operación, una secuencia de `CheckpointSpec` y argumentos nombrados.
`originals` mapea backbone a directorio de pesos originales. No hay rutas HPC implícitas.

| `analysis` | Entradas científicas | Salida y alcance |
| --- | --- | --- |
| `pretraining_drift` | Checkpoints y originales | Estadísticas seleccionadas de W menos W original, por tensor |
| `spectral_fingerprints` | Checkpoints; originales si la fuente los necesita | Métricas seleccionadas por matriz y fuente `base`, `final` o `delta` |
| `cross_language` | Checkpoints de un mismo backbone/origen, distinta lengua y misma entidad | Comparaciones seleccionadas de pesos alineados |
| `cross_entity` | Mismo backbone/origen e idioma, distinta entidad | Comparaciones seleccionadas de pesos alineados |
| `language_entity` | Mismo backbone/origen, cambian lengua y entidad | Comparaciones seleccionadas de pesos alineados |
| `cross_backbone` | Checkpoints de backbones distintos; perfiles reutilizables opcionales | Distancias intrínsecas entre arquitecturas; calcula solo los intermediarios necesarios si no se proporcionan |
| `language_entity_decomposition` | Grupo de celdas language/entity con original compartido | Sumas de cuadrados y fracciones descriptivas de un diseño completo |
| `model_centrality` | Tabla explícita de distancias de pares | Distancia media/mediana y vecinos; no centralidad por caminos ni equivalencia funcional |

Filtros de experimento: `backbones`, `languages`, `entities`, `folds`, `seeds` y `checkpoint_ids`.
Filtros de parámetros: `layers`, `components` y `tensors`. `layers` usa los índices encontrados
en las claves del encoder; `components` usa nombres como `q`, `k`, `v`, `o`, `ffn_up`,
`ffn_down`, `ffn_gate`, `norm` y `embeddings`. `tensors` selecciona claves normalizadas.
`pairs` restringe comparaciones mediante tuplas de dos `CheckpointSpec.uid`; `checkpoint_ids`
filtra las etiquetas opcionales declaradas, no esos UID. Los pares deben respetar el diseño del
análisis y compartir fold y semilla. Los filtros no inventan experimentos ausentes.

Los metadatos/formatos se validan antes de materializar tensores; la cobertura de alineación se
comprueba sobre las coordenadas del encoder. Pedir una selección no elimina esa comprobación.
Una selección vacía no es evidencia de validación científica.

`sources` selecciona pesos originales (`base`), finales (`final`) o actualizaciones (`delta`).
No mezclar esos significados al agregar perfiles. `spectral_results`, `drift_results` y
`pair_results` aceptan un DataFrame o `AnalysisResult` y permiten compartir tablas intermedias
de forma explícita. Sin perfiles suministrados, `cross_backbone` calcula los necesarios para
las métricas pedidas, sin ejecutar análisis ajenos. Calcular centralidad
no vuelve a leer pesos ni crea las comparaciones automáticamente.

## Métricas seleccionables

Las constantes `DRIFT_METRICS`, `PAIR_METRICS` y `SPECTRAL_METRICS` enumeran nombres soportados.
Las funciones atómicas validan nombres desconocidos en lugar de ignorarlos.

- `relative_frobenius(final, base)` calcula norma Frobenius de la actualización dividida por
  norma del original. Es la misma definición que el nombre histórico `relative_l2`.
- `drift_metrics(..., metrics=[...])` ofrece normas, energía, RMS, MAE, cambio medio/máximo
  y coseno final-original. No necesita descomposición espectral.
- `pair_metrics(a, b, metrics=[...])` ofrece coseno/ángulo, Pearson, distancias L1/L2 normalizadas,
  RMS, normas y acuerdo/conflicto de signo en coordenadas compartidas.
- `spectral_metrics(matrix, metrics=[...])` ofrece norma Frobenius/espectral/nuclear, rangos
  estable/efectivo, entropías, fracciones de energía, curva de energía y cuantiles normalizados.
  Norma Frobenius y cuantiles no requieren SVD; varias estadísticas espectrales reutilizan
  una descomposición. El estado distingue resultado exacto, aproximado, desactivado o indefinido.
- `principal_basis`, `principal_angles`, `topk_overlap`, `quantile_w1`,
  `paper_trace_similarity` y las funciones de alineación publicadas son importables por separado.

Ejemplo sin checkpoints ni escritura:

```python
import torch
from weight_analysis.metrics import relative_frobenius, spectral_metrics

base = torch.eye(3, dtype=torch.float64)
final = 2 * base
assert relative_frobenius(final, base) == 1.0
profile = spectral_metrics(final, metrics=["matrix_fro"], mode="exact")
```

`MetricOptions` controla dispositivo, modo/rango de SVD, umbral exacto, rango de subespacio,
top-k, umbral de signo, cobertura mínima y tamaño de fragmentos factoriales. El default es CPU.
Solicitar CUDA no debe producir una sustitución silenciosa por CPU. La versión actual requiere
valores positivos para los rangos; la selección de métricas evita el cálculo no requerido.

## Persistencia y unidades independientes

`AnalysisResult(analysis, data, provenance)` contiene un DataFrame y metadatos JSON compatibles.
La dataclass congela sus campos, no convierte pandas en una estructura inmutable: no modificar
el DataFrame si se pretende reutilizar su identidad o compararlo con un resultado persistido.

```python
from weight_analysis import merge_results, read_result, write_result

write_result(result, "salidas/drift.parquet")
restored = read_result("salidas/drift.parquet")
combined = merge_results(
    [result_part_a, result_part_b], observation_keys=["checkpoint", "tensor", "level"]
)
```

La escritura Parquet usa un temporal, verifica su lectura y reemplaza atómicamente el destino.
El formato guarda versión de esquema, análisis, procedencia y digest de contenidos. La lectura
rechaza archivos sin estos metadatos, esquemas desconocidos y digests inconsistentes.

`merge_results` une unidades compatibles sin depender de su orden. Reintentos idénticos se
deduplican por identidad; un reintento con datos o procedencia distintos se rechaza. Unidades
distintas conservan observaciones distintas aunque coincidan sus valores. La unión no recalcula
rankings ni estadísticas de grupo; esas operaciones deben invocarse explícitamente después.
Los selectores de parámetros pertenecen a la identidad de la unidad, no a su definición numérica:
pueden combinarse fragmentos disjuntos por checkpoint o por capas con las mismas métricas/opciones.
`observation_keys` rechaza observaciones solapadas entre unidades distintas; añadir `source` a las
claves para espectros. Sin estas claves, la unión conserva las filas de unidades distintas y no
demuestra que los fragmentos sean disjuntos.

Un `AnalysisResult` intermedio conserva su cadena de procedencia. Al contrastarlo con checkpoints
declarados, sus hashes deben seguir coincidiendo; un perfil obsoleto se rechaza.
La identidad del original registrado también debe coincidir con el original declarado en la
nueva llamada. Una copia en otra ruta se admite si sus archivos y configuración son idénticos.
Un DataFrame suministrado directamente tiene su contenido hasheado y se marca como procedencia
no verificable.
La centralidad sobre una tabla histórica no necesita volver a abrir los checkpoints originales.

`cache_dir` activa reanudación para unidades de cálculo; con `resume=True` puede recuperarse
un resultado válido. Sin `cache_dir`, `analyze` devuelve resultados en memoria y no los escribe.
La identidad incluye selección, opciones, dimensiones, archivos de pesos/configuración y versiones
numéricas. Verificar esos hashes exige leer los bytes de los archivos, incluso si se filtran
parámetros o se va a reutilizar un cálculo. Evita materialización/SVD redundante, no toda lectura.

La biblioteca no lanza procesos, jobs ni colas: el usuario distribuye llamadas independientes,
persiste sus resultados y los agrega. Véase el ejemplo Python autocontenido.

```python
comparison = analyze(
    analysis="cross_language", checkpoints=[spanish, english],
    originals={"encoder": original_path},
    pairs=[(spanish.uid, english.uid)], sources=["delta"], metrics=["cosine"],
    layers=[0], components=["q"], cache_dir="salidas/cache", resume=True,
)
```

En este ejemplo `spanish` y `english` son declaraciones del mismo backbone, entidad, fold y
semilla, con idiomas distintos. Una petición posterior con los mismos contenidos y opciones
puede reutilizar la unidad validada. Cambiar los pesos, el `config.json` o la definición numérica
identifica otra unidad.

## Interpretación y compatibilidad

Un denominador nulo produce NaN como valor indefinido, no similitud cero. El rango efectivo
de Roy–Vetterli usa probabilidades proporcionales a valores singulares; la variante de energía
usa sus cuadrados y es una métrica diferente. SVD truncada no determina entropía o norma nuclear
exactas. `quantile_w1` aproxima una integral sobre cuantiles, no es un oráculo de Wasserstein
exacto sobre todas las muestras.

Las funciones históricas `basis`/`subspaces` preservan su definición para reproducibilidad:
un rango solicitado mayor que el rango numérico puede incluir direcciones arbitrarias del
núcleo. La API estricta `principal_basis` permite tratar ese caso explícitamente. No mezclar
ambas versiones sin registrar la elección. Las proyecciones deben compartir coordenadas;
los fingerprints entre arquitecturas son comparaciones intrínsecas, no funcionales.

La descomposición factorial requiere celdas completas y tratamiento explícito de réplicas.
Folds y semillas no pueden sobrescribirse ni convertirse en grados de libertad inferenciales
desde un único checkpoint. Las asociaciones con F1 agregado son descriptivas; los pares de
modelos comparten nodos y no son observaciones independientes.

Las fachadas históricas siguen disponibles para reproducir esquemas y cálculos previos; la API
selectiva identifica sus contratos nuevos. Ninguna prueba sintética certifica resultados reales
del paper ausente. Consulte la auditoría para fórmulas, decisiones y límites de validación.
