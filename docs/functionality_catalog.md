# Catálogo de funcionalidades y contratos científicos

El objeto auditado es WeightAnalysis, baseline `285cd8bec45f0cfb30807519ca2e539a3acdc4df`.
El paper de investigación objetivo y los checkpoints reales no se han suministrado.
Las hipótesis indicadas aquí son preguntas que el código puede explorar; no se
atribuyen al paper del usuario. Las referencias a artículos en el código original
no demuestran por sí solas una reproducción completa de esos artículos.

## Capas y superficies

| Capa | Responsabilidad | Superficie |
| --- | --- | --- |
| `core` | Identidad, formatos de pesos, claves y cobertura | `Checkpoint`, `CheckpointSpec`, `WeightReader`, `audit_alignment`, `identify` |
| `metrics` | Operaciones sobre tensores/perfiles sin checkpoints | `drift`, `VectorAccumulator`, `spectral`, `basis`, `subspaces`, `topk_overlap`, `quantile_w1`, funciones `paper_*` |
| `analyses` | Describir un checkpoint y descomposición factorial | `describe_checkpoint`, `factorial`, `validate_factorial_design` |
| `comparisons` | Pares alineados, perfiles heterogéneos y centralidad | `compare_related`, `compare_backbones`, `model_centrality`, `model_distance_summary`, `ranks`, `rank_related_encoder_pairs` |
| `statistics` | Asociaciones F1 descriptivas | `summarize_f1`, `correlation_row` |
| `results` | Esquemas, procedencia y persistencia explícita | Separados del cálculo por la API nueva |
| `visualization` | Figuras desde tablas ya calculadas | `plot_metric` (extra matplotlib opcional) |

Los imports históricos `weightspace.analyses` y `weightspace.performance` permanecen
como fachadas. Los módulos científicos nuevos no dependen del namespace histórico.
Los runners `weightspace.cli` y `weightspace.parallel` permanecen para reproducir el
flujo anterior; el segundo ejecuta fragmentos localmente y no genera infraestructura.
No se han añadido scripts ni sistemas de scheduling.

## Análisis existentes

### Drift respecto al pretraining

`describe_checkpoint(cp, base_path, settings, ..., compute_drift=True,
compute_spectra=True)` alinea primero el checkpoint afinado con su original por
claves y formas. Excluye las cabezas de tarea. Calcula
`ΔW = W_final − W_original` en doble precisión. Las filas contienen identidad,
paths, tensor, componente, capa, cobertura por elementos y métricas.

La norma relativa es `||ΔW||F / ||W_original||F`; RMS divide por la raíz del número
de parámetros, y MAE por ese número. Los agregados de capa, componente y encoder
suman energías y luego extraen la raíz: no promedian normas tensoriales. Los
productos escalares se suman antes de normalizar el coseno agregado. Las cabezas
atencionales se cortan solo para layouts conservadores conocidos de familias BERT.

Los flags permiten solicitar drift o espectros de forma separada en el runner
histórico. La API nueva permite seleccionar métricas sin ejecutar este runner
completo. La identificación previa de formas usa metadatos y no materializa pesos.

**Pregunta exploratoria:** cuánto cambia cada región del encoder respecto a su
original. **Límite:** magnitud de cambio no implica mejora funcional ni F1.

### Fingerprints espectrales

El mismo lector histórico produce filas separadas para `source=base`, `final` y
`delta` sobre proyecciones Transformer reconocidas. SVD exacta permite reportar
normas espectral/nuclear, rango estable, entropías y rangos efectivos, además de
CDF energética y cuantiles de pesos normalizados por RMS. SVD truncada se etiqueta
`topk_approx` y no se presenta como rango efectivo, norma nuclear ni CDF exactos.
Los resultados base pueden reutilizarse mediante un cache explícito proporcionado
por el llamante, sin estado global. Los caches históricos se limitan a una única
combinación de original y configuración: sus claves no codifican todos los parámetros
del cálculo y no deben reutilizarse entre configuraciones distintas.

**Pregunta:** cómo varía el perfil intrínseco por profundidad. **Límite:**
comparar perfiles intrínsecos no alinea neuronas ni demuestra equivalencia funcional.

### Pares cross-language, cross-entity y language-entity

`compare_related(a, b, base_path, settings, subspace_cache=None)` compara tanto
`ΔW_a` con `ΔW_b` como pesos finales con pesos finales, etiquetados por
`comparison_kind`. El experimento debe compartir original, arquitectura y
coordenadas. Devuelve normas, coseno, Pearson, ángulo normalizado, distancias L1/L2,
acuerdo/conflicto de signo; además ofrece overlap top-k y alineamiento de subespacios
solo donde corresponda. La función no decide qué pares son cross-language: eso lo
selecciona explícitamente el llamante o el runner.

El overlap descarta actualizaciones exactamente nulas y resuelve empates por índice
para que la selección sea reproducible. Los subespacios históricos usan vectores
singulares izquierdos. Las métricas `paper_colm26_*` usan las direcciones/capas
menos alineadas según la implementación histórica; la media de todos los cosenos
principales es otra métrica y conserva otro nombre.

`rank_related_encoder_pairs` añade rangos solo a filas encoder, `delta`, familia
`vector`; no mezcla tensores, cabezas, pesos finales o subespacios con ese ranking.

**Pregunta:** si actualizaciones de idiomas o entidades comparten geometría.
**Límites:** procedencia común del original debe confirmarse externamente; coincidencia
de nombres y formas no la demuestra. Los pares que comparten modelo no son
observaciones estadísticamente independientes.

### Cross-backbone

`compare_backbones(a, b, spectral_df, drift_df, min_components=3)` consume resultados
intermedios explícitos: no vuelve a cargar checkpoints. Produce perfiles Frobenius
por componente/profundidad y su correlación TRACE histórica, distancias entre CDF
energéticas, Wasserstein aproximada sobre cuantiles y distancia de perfiles
`log1p(relative_l2)`. Reporta cobertura de componentes y distingue familias. No
sustrae pesos de distintas arquitecturas. `ranks` ordena cada métrica por separado.

**Pregunta:** parecido estructural intrínseco de arquitecturas. **Límites:** no hay
similitud funcional; la correlación es indefinida para perfiles constantes y se
conserva como NaN. La implementación histórica asigna `fold=1`: la API nueva debe
estratificar y declarar las dimensiones del estudio de forma explícita. No deben
atribuirse sus resultados a otros folds por ese campo fijo.

La ruta nueva de cross-backbone prepara únicamente los extremos de los pares
solicitados. Cada fuente (`base`, `final`, `delta`) calcula sus dependencias concretas:
por ejemplo, solicitar TRACE delta y cuantiles finales no calcula cuantiles delta.
Los intermedios deben identificar checkpoint, tensor, fuente y componente, sin
observaciones duplicadas; los perfiles numéricos tienen dimensiones y orden válidos.
No se convierte silenciosamente una tabla incompleta o mal formada en evidencia de
comparabilidad.

### Descomposición language/entity

`factorial(checkpoints, base_path, settings, *, layers=None, components=None,
tensors=None)` calcula la descomposición descriptiva de una cuadrícula balanceada:

- `μ = mean_language,entity(ΔW)`.
- `L_l = mean_entity(ΔW_l,e) − μ`.
- `E_e = mean_language(ΔW_l,e) − μ`.
- `I_l,e = ΔW_l,e − μ − L_l − E_e`.
- `SS_language = n_entities Σ_l ||L_l||²`,
  `SS_entity = n_languages Σ_e ||E_e||²`,
  `SS_interaction = Σ_l,e ||I_l,e||²`.
- `SS_total = Σ_l,e ||ΔW_l,e − μ||²` y se valida
  `SS_language + SS_entity + SS_interaction = SS_total`.

Las fracciones dividen por SS total; total cero produce NaN. Se mantiene una sola
copia por tensor/celda y se procesan coordenadas en chunks. La cobertura de la
cuadrícula se verifica antes de aplicar filtros; los tensores excluidos no se
materializan. Los tests verifican efectos ortogonales conocidos y equivalencia de
orden/chunks.

`validate_factorial_design` exige un backbone, fold y seed comunes, al menos dos
idiomas y entidades y una observación por cada celda. El runner histórico conserva
su comportamiento previo de omitir cuadrículas incompletas y etiquetar fold 1 para
reproducibilidad. La nueva validación evita mezclar réplicas o backbones.

**Límite estadístico:** no hay réplica por celda. La interacción absorbe ruido de
entrenamiento y no se interpreta como un efecto causal ni se realizan F-tests.

### Centralidad de modelos

`model_distance_summary(pair_df)` conserva el cálculo histórico.
`model_centrality(pair_df, *, metrics=None)` añade un contrato estricto para los
mismos estadísticos descriptivos: cada arista no dirigida aparece una sola vez,
no hay autoaristas y fold/seed, si están presentes, definen estratos independientes.
Calcula media/mediana de las distancias a vecinos, vecino más cercano/lejos y rango
ascendente de distancia media solo para comparaciones completas.

La completitud exige un vecino válido por cada otro modelo observado. NaN no cuenta
como vecino. Se rechazan duplicados y distancias negativas/infinitas. La selección
`metrics` evita procesar columnas no solicitadas; el resultado es independiente del
orden y orientación de aristas. No modifica el DataFrame de entrada.

**Pregunta:** qué modelo está más próximo al conjunto según una distancia concreta.
**Límite:** no es centralidad por caminos mínimos, linaje demostrado ni calidad NER.

### Visualización explícita

`plot_metric(result, *, metric, x="layer", group_by=("checkpoint", "source"),
ax=None)` acepta un DataFrame o `AnalysisResult` y retorna ejes matplotlib. Solo
consume resultados; no carga pesos, agrega observaciones, guarda ni muestra figuras.
Cuando existe `level`, selecciona filas tensoriales; ordena x y exige una observación
por posición/grupo. Rechaza mezclar identidades científicas distintas dentro de una
línea y mediciones/posiciones no finitas. Para múltiples tensores por capa, el
llamante elige grupos adicionales o hace una agregación científica explícita.

### Geometría y F1

`summarize_f1(drift, cross_backbone)` y `correlation_row` calculan Spearman y Pearson
con casos completos, requieren variación y al menos tres observaciones. No producen
p-values para díadas no independientes. El flujo histórico relaciona pesos de fold
1 con media/desviación F1 de cuatro folds: se declara explícitamente y no se trata
como cuatro réplicas de pesos. La comprobación de claves de rendimiento y las
uniones son responsabilidad de la capa de resultados, no de las métricas.

Los informes históricos están en `src/weightspace/reports`; se invocan como
`python -m weightspace.reports` con rutas explícitas o `WS_DATA`/`WS_F1`. No se ejecutan como efecto secundario de la librería. No se han
aportado esas tablas ni se han validado resultados estadísticos del estudio real.

## Funcionalidad no implementada por falta de evidencia/datos

Estabilidad entre folds necesita checkpoints por fold/seed; similitud funcional e
interpolación necesitan entradas, evaluación y arquitecturas compatibles. Ninguna
se inventa a partir de geometría de pesos. Priorización P0–P3 y plan futuro en
`architecture_reference.md`; definiciones atómicas y riesgos numéricos en
`mathematical_audit.md`.
