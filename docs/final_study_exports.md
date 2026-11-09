# Exportaciones finales por estudio

Este paso CPU lee únicamente los Parquet publicados del run. No ejecuta `prepare`,
no abre checkpoints y no modifica el run original. El exportador vive fuera del
namespace científico congelado, de modo que añadirlo no cambia su firma de cálculo.
Elegir un directorio de salida NUEVO, separado del run.

Desde la raíz actualizada de WeightAnalysis en Picasso, manteniendo `RUN_DIR`:

```bash
export WEIGHT_ANALYSIS_PROJECT_DIR="$PWD"
export EXPORT_DIR="${RUN_DIR}-final-tables"
sbatch --chdir="$PWD" \
  --output="$RUN_DIR/logs/export-%j.out" \
  --error="$RUN_DIR/logs/export-%j.err" \
  scripts/picasso/compact/export.sbatch "$RUN_DIR" "$EXPORT_DIR"
```

El job necesita CPU, 48 GB y hasta ocho horas; no reserva GPU. Usa PyArrow/pandas
del módulo pytorch/2.10.0, sin necesitar DuckDB. Procesa bloques de filas y escribe
Parquet ZSTD. Comprueba hashes de entrada, cardinalidad de los enlaces F1 y número
de filas exportadas. Nunca une por nombres aproximados.

Se generan `01_pretraining_drift`, `02_spectral_fingerprints`, `03_cross_language`,
`04_cross_entity`, `05_cross_language_entity`, `07_cross_backbone`,
`08_language_entity_decomposition`, `11_model_centrality`,
`12_entity_adaptation_by_language`, `13_language_adaptation_by_entity`,
`14_mean_adaptation_by_backbone`, `15_coverage`, `16_exclusions`,
`17_sufficient_statistics`, `18_intrinsic_profiles`, y las tablas F1 de scores,
rankings y pares. Todos son `.parquet`.

`09_cross_backbone_by_entity_index.parquet` indexa los filtros por entidad/idioma,
métrica, fuente y alcance de `07`; no inventa otra métrica ni duplica sus pares.

Cada fila conserva `family`, `metric`, `source`, `scope`, capa, componente, tensor,
referencia, `status`, `exactness` y los IDs originales. `measurement` explica qué
se mide. Las columnas `f1_*_a/b` conservan identidad, F1 medio, SD y ámbito de cada
extremo. Fold/semilla físicos son `fold_a/b`, `seed_a/b`; el F1 agregado no se
atribuye a esa fold. Los centroides no reciben un F1 ficticio. `payload` contiene
los arrays espectrales y estadísticas suficientes; los alias se resuelven para
que el archivo espectral sea autocontenido.

Centralidad reutiliza `model_centrality`: distancia media a los otros nodos dentro
de idioma/entidad/fold/semilla y por separado para cada métrica, fuente, alcance y
componente. TRACE se convierte explícitamente en `1-trace_similarity`. Las aristas
ausentes o indefinidas impiden ordenar al nodo como completo. La centralidad es
una descripción de distancias, no de calidad ni equivalencia funcional.

`catalog.json` registra medidas, filas, hashes, procedencia, estado del run y
criterios. Los nombres se parecen a los históricos, pero NO garantizan equivalencia
de esquemas, universos ni definiciones: las columnas y el catálogo son la referencia.

**Límite conocido:** el run actual incluyó cabezas `cls.*` y pooler en parte de sus
denominadores esperados. Estos exports conservan `partial_coverage` y no afirman
haber corregido esos agregados. El drift por capa validado puede analizarse ya;
los agregados globales requieren la revisión correspondiente. No se han eliminado
filas parciales ni métricas indefinidas.

Exportar todo puede seguir ocupando varios GB. Este paso organiza y documenta los
resultados; no los convierte automáticamente en un archivo pequeño.
