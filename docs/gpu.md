# Ejecución en GPU

`analyze` y `MetricOptions` solicitan GPU por defecto. `device="GPU"`, `"gpu"`, `"cuda"`
usan la GPU CUDA actual; `"gpu:1"` o `"cuda:1"` seleccionan un índice concreto. El argumento
`device` de `analyze` tiene prioridad sobre `MetricOptions.device`. Una GPU ausente, un índice
inválido o PyTorch sin CUDA producen un error antes de leer los checkpoints. No hay fallback
silencioso por indisponibilidad u OOM. Para un equipo sin CUDA usa `device="cpu"`.

```python
from weight_analysis import analyze
from weight_analysis.configuration import MetricOptions

result = analyze(
    analysis="spectral_fingerprints", checkpoints=checkpoints, originals=originals,
    device="gpu", metrics=["stable_rank", "normalized_weight_quantiles"],
    options=MetricOptions(spectral_mode="auto", max_svd_elements=3_000_000, topk=32),
)
```

## Trabajo trasladado

| Operación | Ejecución |
| --- | --- |
| Alineación, configuración, hashes y lectura de pesos | CPU y archivos locales |
| Resta final-original, normas, RMS, coseno, Pearson y signos | Torch en el dispositivo elegido |
| Overlap/Jaccard y selección top-k estable | Torch/CUDA para tensores CUDA |
| SVD exacta y estadísticas del espectro, entropías y curvas | Torch/CUDA, sin convertir matrices a NumPy |
| SVD truncada CUDA | Range finder con generador local, ocho direcciones extra y cuatro iteraciones |
| Bases y ángulos principales | GPU; las bases permanecen en GPU para compararlas |
| Cuantiles normalizados | Torch/CUDA; ordenación e interpolación para entradas por encima del límite de torch.quantile |
| Descomposición language/entity | Chunks en GPU, acumulación de sumas de cuadrados en GPU |
| Interpolación de perfiles, TRACE y distancias entre perfiles | Torch en el dispositivo elegido |
| Extracción de tablas, groupby, centralidad, correlaciones F1, plots y Parquet | pandas/NumPy/SciPy/Arrow en CPU |

Las métricas vectoriales agrupan su salida escalar en una transferencia final, evitando una
sincronización por cada norma o distancia solicitada.

Los tensores se transfieren por parámetro antes de restar y permanecen en GPU durante sus
métricas. La factorial conserva los pesos crudos en CPU y transfiere chunks, evitando exigir
que todos los modelos quepan en VRAM. Se devuelven escalares/listas para formar las tablas;
esa frontera y las comprobaciones de validez requieren sincronizaciones. No se promete
que cada operación ni toda la aplicación permanezcan en CUDA.

## Precisión y aproximaciones

Se conserva float64 para la aritmética científica. Las definiciones exactas no cambian;
CPU y GPU pueden diferir por redondeo y orden de reducciones. No se habilitan TF32, precisión
mixta ni cambios de configuración CUDA globales. Las primitivas sin `device` conservan el
dispositivo de entrada; las funciones históricas mantienen su camino CPU anterior.

La ruta top-k CPU conserva ARPACK. La CUDA usa una aproximación aleatorizada con semilla local
fija, sin alterar el RNG global. Se etiqueta `topk_approx`: no se deducen entropías, rango efectivo,
norma nuclear ni CDF exactas de un espectro parcial. No se promete igualdad numérica entre
solvers aproximados. Las bases CUDA grandes usan la aproximación al superar
`max_svd_elements` con rango solicitado menor que el máximo; el resultado de comparación informa
`subspace_status="topk_approx"`. Para descomposición completa eleva el umbral, asumiendo su coste.

La procedencia registra dispositivo efectivo y versión CUDA, además de las versiones numéricas.
`selective-v2-device-resident` distingue estas rutas de los resultados anteriores y evita
reutilizar sus caches como si fueran el mismo cálculo.

## Memoria, velocidad y comprobación

Reduce `factorial_chunk_elements` si la factorial no cabe en VRAM. Para matrices grandes usa
`spectral_mode="auto"`/`"topk"`, un rango pequeño y selecciona solo las métricas necesarias.
Un cuantíl exacto o la ordenación top-k pueden necesitar memoria proporcional al parámetro.
CUDA no implica mayor velocidad para modelos pequeños, tablas o GPUs con float64 lento.

`tests/test_gpu_execution.py` comprueba alias, errores, ausencia de fallback, algoritmos Torch
contra oráculos NumPy y el RNG local. Sus pruebas CUDA ejecutan las ocho familias de análisis
y comparan CPU/GPU cuando hay hardware. Se omiten explícitamente sin CUDA. El entorno de este
desarrollo tiene PyTorch 2.15.0+cpu y ninguna GPU: no se ha ejecutado la rama CUDA real ni medido
una aceleración. Es necesaria esa validación en el destino con CUDA antes de afirmar rendimiento.
