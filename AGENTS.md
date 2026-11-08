# Trabajo en Weight Analysis

- Mantener `weight_analysis` independiente de `ner_lab`; la referencia arquitectónica y su SHA
  están en `docs/architecture_reference.md`. No integrar ni copiar código de ner_lab.
- La petición posterior del usuario autoriza adaptadores SLURM en `scripts/picasso/`.
  Mantener scheduling fuera del núcleo científico; no ejecutar sbatch durante desarrollo.
- Preservar definiciones y compatibilidad histórica. Obtener baseline y regresiones antes de
  cambiar fórmulas; justificar cambios en la auditoría y conservar identificaciones históricas.
- Separar contratos/lectura, métricas, análisis, comparación, estadística, resultados y visualización.
  Los filtros se aplican antes de materializar tensores. Una métrica no debe disparar otras costosas.
- Usar argumentos explícitos, tipos modernos y configuración de una sola responsabilidad.
  El cálculo devuelve objetos; I/O, caché y logging global deben ser decisiones explícitas.
- Para verificar, ejecutar pytest desde el repositorio con el paquete instalado. Las pruebas
  obligatorias usan CPU y fixtures sintéticas; distinguir fallos, errores, omisiones y pruebas
  no ejecutadas. No ocultar fallos previos de scripts ausentes ni inventar sustitutos.
- No afirmar confirmación del paper sin paper, hipótesis y datos. Pares dependientes no son
  réplicas independientes; no atribuir una media de F1 entre folds a un único checkpoint.
- Todo el código de producción vive en src/ (weight_analysis, weight_analysis_scheduler y weightspace).
  Los ejemplos ejecutables también se distribuyen como módulos; no restaurar launchers Python de raíz.
- Mantener README y AGENTS breves; detalles científicos y contratos públicos pertenecen a `docs/`.
- MultiClinNER nuevo usa `experiments.compact`: diario por tarea y Parquet agrupados por estudio;
  no crear archivos por tensor ni escribir sobre runs históricos.
