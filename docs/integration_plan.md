# Contrato para una integración futura

Este documento no realiza la integración. La revisión de convenciones y sus fuentes está en
[architecture_reference.md](architecture_reference.md), incluida la priorización científica.

## Frontera que debe permanecer estable

Un adaptador futuro debería recibir manifiestos del entrenamiento y producir `CheckpointSpec`:

| Información externa | Contrato independiente |
| --- | --- |
| Identidad del encoder preentrenado | `backbone` y ruta de `originals[backbone]`, con procedencia verificable |
| Ruta física de pesos entrenados | `CheckpointSpec.path` |
| Idioma, entidad, fold, semilla y etiqueta de snapshot | Campos explícitos del checkpoint; no deducidos de la carpeta |
| Subconjunto de interés | Filtros de `analyze` |
| Producto científico | `AnalysisResult.data` y `AnalysisResult.provenance` |
| Persistencia o intercambio entre procesos | `write_result`, `read_result` y `merge_results` |

Los adaptadores viven en el borde de la aplicación; las métricas y lectores no importan
`lab.ner.training`, no conocen los directorios de Picasso y no dependen de cómo se envió un job.
Una tabla F1 debe declarar qué checkpoint/fold/semilla representa antes de unirse a geometría.

## Decisiones reservadas para esa tarea

El proyecto oficial usa `src/lab` como namespace. La intención `ner_lab/analysis/` necesita
acordar su ubicación importable; `lab.analysis` sería una opción, no una carpeta creada ahora.
También deben decidirse distribución opcional frente a migración controlada del namespace,
compatibilidad de esquemas y pin de torch: el commit de referencia fija 2.10.0.
La suite sintética independiente se ha validado con esa versión en CPU, pero la combinación
completa de dependencias y la integración de ner_lab requerirán su propia validación.

Una vez autorizada, la integración debe demostrar:

1. Equivalencia de la misma unidad desde Python, el adaptador y la eventual entrada de tarea.
2. Ausencia de imports inversos hacia tareas de entrenamiento y de efectos secundarios al importar.
3. Identidad de pesos y configuraciones mediante hashes; distinción de original y entrenado.
4. Selección, fold/semilla, lectura/escritura, reanudación y agregación sin pérdida de procedencia.
5. Reproducción de resultados históricos o registro explícito de cada cambio científico.

No se añaden scheduler, colas, scripts de envío ni ejecución automática de otros análisis.
