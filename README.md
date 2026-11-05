# Weight Analysis

Biblioteca Python independiente para geometría de checkpoints: drift, fingerprints espectrales,
comparaciones por idioma/entidad/backbone y descomposición factorial. Usa GPU CUDA por defecto,
admite CPU explícita (`device="cpu"`) y no depende
de `ner_lab`, de Picasso ni de un scheduler. Requiere Python 3.11 o posterior; las dependencias y
extras de desarrollo, estadística y visualización están declarados en `pyproject.toml`.

## Uso

Con la distribución instalada, cada análisis se invoca por separado. Las rutas son explícitas;
no se descargan modelos ni se ejecuta inferencia:

```python
from weight_analysis import CheckpointSpec, analyze

checkpoint = CheckpointSpec(
    backbone="encoder", language="es", entity="disease",
    path="/datos/entrenado", fold=1, seed=42,
)
result = analyze(
    analysis="pretraining_drift",
    checkpoints=[checkpoint],
    originals={"encoder": "/datos/original"},
    layers=[0], components=["q"], metrics=["relative_frobenius"],
)
print(result.data)
```

`device="GPU"`, `"gpu"`, `"cuda"` y `"cuda:0"` seleccionan CUDA. Requiere una GPU NVIDIA
y PyTorch con CUDA; no hay sustitución silenciosa por CPU. [Detalles GPU](docs/gpu.md).

El cálculo devuelve `AnalysisResult`; escribirlo requiere una llamada explícita a `write_result`.
Solo `cache_dir` solicita persistencia automática. El ejemplo autocontenido
[`independent_analysis.py`](src/weight_analysis/examples/independent_analysis.py) genera pesos sintéticos,
calcula unidades independientes y las agrega, sin checkpoints grandes ni red.

## Guías

- [API, catálogo y ejecución independiente](docs/api.md).
- [Organización de src y equivalencias de los scripts retirados](docs/repository_layout.md).
- [Parquet NER real, arrays GPU de 8 horas y ejecución en Picasso](docs/picasso_experiments.md).
- [Geometría comparativa fina: F1, entidades/idiomas, referencias, Parquet y continuidad ante errores](docs/comparative_geometry.md).
- [Auditoría matemática y limitaciones](docs/mathematical_audit.md).
- [Referencia arquitectónica verificada de ner_lab y prioridades P0–P3](docs/architecture_reference.md).
- [Plan de integración futura](docs/integration_plan.md).
- [Instalación, pruebas ejecutadas y limitaciones conocidas](docs/validation.md).

Las rutas históricas `weightspace` se mantienen para reproducibilidad. La API nueva es
`weight_analysis`; los adaptadores SLURM opcionales están en `scripts/picasso/`. Las pruebas pytest usan fixtures CPU y oráculos
independientes. El paper objetivo y los pesos reales no están disponibles en este entorno: las verificaciones
matemáticas no equivalen a confirmar hipótesis empíricas ni reproducir resultados del estudio.
