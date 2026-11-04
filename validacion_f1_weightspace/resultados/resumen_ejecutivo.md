# Validación F1 y WeightSpace

Resultados exploratorios. Los pares comparten backbones y no son réplicas independientes.

## Cobertura

- Modelos con F1 y drift: 459
- Pares con F1: 4788
- Permutaciones por grupo: 1999

## Correlaciones F1–drift

- delta_energy: rho mediana -0.1158; 17/21 negativas.
- delta_rms: rho mediana -0.1925; 20/21 negativas.
- relative_l2: rho mediana -0.4091; 21/21 negativas.

## Distancia estructural y diferencia F1

- quantiles_base: rho mediana 0.0832; 18/21 positivas.
- quantiles_delta: rho mediana 0.3180; 21/21 positivas.
- spectral_curve_base: rho mediana 0.2881; 21/21 positivas.
- spectral_curve_delta: rho mediana 0.3677; 21/21 positivas.
- trace_base: rho mediana 0.1905; 17/21 positivas.
- trace_delta: rho mediana 0.3080; 19/21 positivas.

## Top–Top vs Top–Bottom

- curve_l1_base: contraste mediano -0.00018; 7/21 positivos.
- curve_l1_delta: contraste mediano 0.01035; 20/21 positivos.
- quantile_w1_base: contraste mediano -0.05384; 9/21 positivos.
- quantile_w1_delta: contraste mediano 0.03078; 20/21 positivos.
- trace_distance_base: contraste mediano 0.11444; 12/21 positivos.
- trace_distance_delta: contraste mediano 0.40853; 20/21 positivos.

## Permutaciones con control base

- quantiles: 13 p nominales < 0.05; 7 q BH < 0.05.
- spectral_curve: 15 p nominales < 0.05; 15 q BH < 0.05.
- trace: 10 p nominales < 0.05; 4 q BH < 0.05.

## Advertencia

Estos p y q son exploratorios y locales. Se requiere una prueba global que preserve la dependencia entre idiomas, entidades y backbones antes de declarar confirmación.
