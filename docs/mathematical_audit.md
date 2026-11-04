# Auditoría de métricas y contratos matemáticos

## Alcance y evidencia

Baseline anterior a la modularización: commit `285cd8bec45f0cfb30807519ca2e539a3acdc4df`,
51 pruebas CPU aprobadas. Las funciones históricas se han trasladado a módulos cohesivos
de `weight_analysis.metrics`; `weightspace.metrics` conserva sus nombres e implementación
numérica. Los cambios de comportamiento que se describen debajo pertenecen a funciones
nuevas con nombres explícitos. No se atribuye a una mera reorganización una corrección
de resultados científicos previos.

No se ha proporcionado el manuscrito objetivo ni checkpoints o resultados F1 reales.
Esta auditoría contrasta definiciones del código y referencias matemáticas independientes;
no certifica hipótesis ni conclusiones de un paper no disponible. Las referencias a TRACE
y a artículos presentes en el código identifican la procedencia declarada del algoritmo,
sin afirmar aquí que se hayan ejecutado sus repositorios originales.

## Catálogo y definiciones

Sea `f` un tensor final, `b` su pretraining alineado, `d=f-b`, `n` el número de
parámetros y `x,y` dos vectores en las mismas coordenadas. Todas las normas de tensores
se aplican a su vectorización; para matrices coinciden con la norma de Frobenius.

| API o familia | Definición y unidades | Condiciones y límites |
| --- | --- | --- |
| `drift`, `drift_metrics` | `delta_l2=||d||₂`; `relative_l2=||d||₂/||b||₂`; `delta_rms=||d||₂/√n`; `delta_energy=||d||₂²` | `relative_frobenius` es el alias explícito de `relative_l2` en la API selectiva. Los deltas conservan el signo final menos base. |
| Derivas de primer orden | `delta_l1=Σ|dᵢ|`; `delta_mae=delta_l1/n`; `max_abs_delta=max|dᵢ|`; `delta_mean=Σdᵢ/n` | No mezclar promedio por tensor y norma global: sus ponderaciones difieren. |
| Normas y dirección | `base_l2=||b||₂`; `final_l2=||f||₂`; `cosine_final_base=〈f,b〉/(||f||₂||b||₂)` | Cero o norma ≤`1e-14` produce NaN donde existe división por una norma nula. |
| `VectorAccumulator`, `pair_metrics` | Coseno; ángulo/π; Pearson; `||x-y||₂`; RMSE; L1 medio; normas individuales | La acumulación de momentos permite comparar varios tensores sin concatenar. Pearson histórico puede perder precisión por cancelación. |
| Distancia normalizada | `normalized_l2=||x-y||₂/√(||x||₂²+||y||₂²)` | Es una normalización simétrica; no es distancia relativa al pretraining ni está limitada a 1. |
| Concordancia de signo | Proporción de signos iguales/opuestos sobre coordenadas donde ambas magnitudes superan `sign_floor` | No contar coordenadas cero como acuerdos. Sin soporte compartido: NaN. |
| `topk_overlap` | `k=ceil(n·fraction)`, selección estable por magnitud descendente e índice ascendente; intersección, Jaccard y coeficiente de solapamiento | Solo coordenadas no nulas; el soporte efectivo puede ser menor que k. Empates de cardinalidad exacta: no reproduce el threshold trimming de TIES. |
| Control nulo top-k | `E[intersección]=k_a·k_b/n`, exceso = observado menos esperado | Supone soportes uniformes independientes de tamaños fijos; no constituye un p-valor ni modela correlaciones entre parámetros. |
| Jaccard ponderado | `Σmin(|x|,|y|)/Σmax(|x|,|y|)` | Diagnóstico de magnitudes; el signo se informa separadamente. Dos vectores cero: NaN. |
| `spectral`, `spectral_metrics` | Valores singulares `σ`; norma espectral `max σ`; nuclear `Σσ`; Frobenius `√Σσ²` | Solo norma de Frobenius y cuantiles de entradas no requieren SVD. |
| Rango estable | `||W||_F²/σ_max²` | Con `σ_max` aproximado debe interpretarse como aproximado. Matriz cero: NaN. |
| Rango efectivo de Roy–Vetterli | `exp(-Σpᵢ log pᵢ)`, `pᵢ=σᵢ/Σσ` | Las probabilidades cero no contribuyen. Es distinto del rango efectivo de energía. |
| Rango efectivo de energía | Misma entropía exponenciada con `pᵢ=σᵢ²/Σσ²` | Es una medida derivada identificada por otro nombre, no el eRank original. |
| Curva y concentración | CDF de energía singular ordenada descendente, interpolada sobre rango relativo; fracciones de rango necesarias para 90/95/99% de energía | Entropías, nuclear, CDF y fracciones de rango exactas no se infieren de top-k: se devuelven NaN/None. `top1/5/10_energy` con menos valores calculados son sumas parciales. |
| Cuantiles normalizados | Cuantiles de `W / RMS(W)` | Invariantes a escalado positivo; conservan signos. No son autovalores ni prueban equivalencia funcional. |
| `quantile_w1` | Cuadratura trapezoidal de `∫₀¹|Qₐ(u)-Q_b(u)|du` sobre una malla uniforme | Aproximación de W1 por cuantiles; no coincide necesariamente con W1 empírico exacto sobre las muestras originales. |
| `basis`, `principal_basis` | Vectores singulares izquierdos principales | `basis` conserva el rango solicitado histórico; `principal_basis` limita al rango numérico positivo y devuelve base vacía para la matriz cero. |
| `subspaces`, `principal_angles` | Singular values de `UᵀV` son cosenos principales; cuerda normalizada = `√mean(1-cos²θ)` | Requieren un espacio de coordenadas común. `principal_angles` valida ortonormalidad y permite rangos distintos; `subspaces` histórico presupone bases válidas y exige rangos iguales. |
| `paper_trace_similarity` | Interpolación de perfiles de normas Frobenius a profundidad común, estandarización y Pearson por componente, promedio de componentes válidos | Al menos dos capas y variación no nula; omisiones se informan por número de componentes. Es distinta de distancia entre CDF espectrales. |
| Alineamiento jerárquico histórico | Por capa: media de los J cosenos principales menores; por componente: media de K capas menos alineadas; finalmente media entre componentes | La agregación de capas es explícita. No confundir con promedio global de cosenos ni con una distancia funcional. |

## Hallazgos confirmados y decisiones de compatibilidad

1. **Validación incompleta de la función histórica `drift`.** Rechazaba un final entero,
   pero no un pretraining entero, y podía devolver NaN/Inf para entradas no finitas.
   `drift_metrics` valida ambos tensores reales, flotantes, finitos, no vacíos y con
   forma idéntica. `drift` se conserva para reproducir contratos anteriores.
2. **Cancelación de Pearson en `VectorAccumulator`.** La fórmula histórica resta
   momentos grandes: `Σx²-(Σx)²/n`. Con desplazamiento `1e12` y variación pequeña puede
   devolver NaN para correlación verdadera 1. `pair_metrics(..., metrics=['pearson'])`
   usa vectores centrados y tiene una prueba contra SciPy. La vía histórica sigue
   disponible, identificable por su función; no se reescriben resultados previos.
3. **Bases de núcleo arbitrarias.** `basis(W, rank=k)` usa k vectores incluso si el
   rango numérico es menor. Esas direcciones pueden cambiar sin información científica.
   `principal_basis` aplica tolerancia relativa (por defecto `max(shape)·eps64`) y
   retiene solo singular values mayores que ese umbral relativo. Quien necesite
   reproducir análisis anteriores utiliza `basis` de forma explícita.
4. **Ortonormalidad presupuesta.** `subspaces` y la función paper-inspired histórica
   confían en el llamador; matrices arbitrarias pueden producir scores inválidos
   después del clipping. `principal_angles` rechaza bases no ortonormales y espacios
   incompatibles. No transforma las coordenadas de dos arquitecturas para forzar una
   comparación.
5. **Parámetros espectrales históricos laxos.** Un modo desconocido podía terminar
   usando top-k; algunos tamaños de malla inválidos fallaban tarde. `spectral_metrics`
   valida modo, dispositivo, k, umbrales y mallas antes de calcular.
6. **Trabajo innecesario.** `spectral` calcula cuantiles incluso cuando solo interesa
   un rango. La API selectiva omite los cuantiles no solicitados y comparte una única
   SVD; `matrix_fro`/cuantiles solos no descomponen la matriz. Las derivas y las
   comparaciones vectoriales no llaman SVD. Esto se verifica haciendo fallar las
   operaciones prohibidas mediante monkeypatch.

La API estricta calcula en float64 sobre CPU y no conserva el grafo de autograd.
Los umbrales absolutos históricos se preservan donde hay compatibilidad; no prometen
invariancia a escalados que crucen `1e-14`. NaN representa una cantidad indefinida,
no cero ni evidencia de ausencia de efecto.

## Validación y cobertura

`tests/test_mathematical_contracts.py` utiliza NumPy y SciPy como referencias
independientes para todos los nombres selectivos de deriva, vector y espectro.
Comprueba invariancia de escala/rotación, casos cero, entradas inválidas, determinismo
top-k, una sola SVD por selección conjunta, ausencia de SVD en peticiones baratas,
rango numérico, bases ortonormales y el caso adverso de Pearson con gran desplazamiento.

`tests/test_metrics.py` y `tests/test_paper_equivalence.py` preservan los contratos
históricos de métricas y fórmulas declaradas. Los oráculos externos siguen optativos:
una comparación con fórmulas independientes no se presenta como ejecución de código
upstream. La equivalencia con checkpoints y tablas reales permanece pendiente hasta
disponer de esos datos y del paper objetivo.

## Interpretación de análisis existentes

La geometría de deltas requiere el mismo pretraining y coordenadas alineadas, no
solo formas tensoriales iguales. Los fingerprints intrínsecos permiten descripciones
entre backbones, pero no prueban parentesco ni similitud funcional. La descomposición
language/entity con una observación por celda es descriptiva: interacción también
absorbe ruido de entrenamiento. Un único fold de pesos no estima estabilidad entre
folds. F1 agregado sobre cuatro folds no es el F1 específico del checkpoint de fold 1;
las asociaciones entre díadas dependientes no admiten p-valores iid sin un diseño
estadístico adicional. No se añade automáticamente ningún análisis de inferencia,
interpolación o similitud funcional.
