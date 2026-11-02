# TP3 – Optimización de Modelos de Machine Learning

**Materia:** 72.75 Aprendizaje Automático (2026 Q2)

**Problema:** predecir si una reserva de infraestructura GPU va a ser cancelada antes de ejecutarse (`canceled_job`), con el dataset `gpu_resource_reservations.csv`.

Este README es el informe del trabajo. Sigue el orden de la consigna ([`TP3. Optimización de Modelos de ML..pdf`](TP3.%20Optimizaci%C3%B3n%20de%20Modelos%20de%20ML..pdf)) y funciona también como guía paso a paso: cada sección dice qué script se corrió, qué decisiones se tomaron y por qué, y qué resultados salieron. Cada paso se commiteó por separado, así que el historial de git sigue el mismo orden.

---

## Índice

- [Paso 0 – Estructura del proyecto y cómo reproducir](#paso-0--estructura-del-proyecto-y-cómo-reproducir)
- [Paso 1 – Preprocesamiento y análisis exploratorio](#paso-1--preprocesamiento-y-análisis-exploratorio)
- [Paso 2 – Model-centered ML: baseline, validación cruzada y métricas](#paso-2--model-centered-ml-baseline-validación-cruzada-y-métricas)

---

## Paso 0 – Estructura del proyecto y cómo reproducir

```
.
├── gpu_resource_reservations.csv   # dataset original
├── requirements.txt
├── tp3/                            # código reutilizable
│   ├── config.py                   # rutas, semilla, grupos de columnas, columnas con leakage
│   ├── data.py                     # carga, limpieza determinística, split train/test agrupado
│   ├── preprocessing.py            # ColumnTransformer (imputación, one-hot, escalado)
│   ├── evaluation.py               # k-fold CV, métricas, análisis por cohortes
│   └── plotting.py                 # estilo común de figuras
├── steps/                          # un script por paso del TP (se corren en orden)
├── reports/figures/                # figuras generadas (se muestran en este README)
├── reports/tables/                 # tablas de resultados (.csv y .md)
├── models/                         # modelo final serializado
└── predictor/                      # función predictora para entregar
```

**Instalación**

```bash
pip install -r requirements.txt
```

**Ejecución:** cada paso es un script independiente que lee el CSV, regenera sus figuras y tablas y las guarda en `reports/`:

```bash
python steps/01_eda.py
```

(los comandos de los pasos siguientes se indican en cada sección).

**Reproducibilidad:** semilla fija (`RANDOM_STATE = 42`) en splits y modelos. Todos los scripts toman los datos de `tp3.data.get_data()`, así el conjunto de test es siempre el mismo y no se usa en ningún paso salvo la evaluación final (Paso 5).

---

## Paso 1 – Preprocesamiento y análisis exploratorio

**Script:** `python steps/01_eda.py` → tablas `reports/tables/01_*` y figuras `reports/figures/01_*`.

### 1.1 Primer vistazo al dataset

| chequeo | valor |
|:--|:--|
| filas × columnas | 47 842 × 34 (las 29 de la consigna + 5 extra: `execution_status`, `execution_status_date`, `total_processes`, `total_gpu_hours`, `high_value_client`) |
| tasa de cancelación | **41.7 %**: desbalance moderado, no extremo |
| período | ejecuciones de julio 2015 a agosto 2017 |
| filas idénticas en todas las features | **15 936 (33 %)**. El 40.6 % de las filas pertenece a algún grupo de duplicados |
| cancelación en filas duplicadas vs únicas | **61.2 % vs 28.4 %** |
| grupos de duplicados con etiquetas contradictorias | 79 de 3 499 |

Valores faltantes:

| columna | faltantes | interpretación |
|:--|--:|:--|
| `organization_id` | 45 540 (95.2 %) | la reserva **no** está asociada a una organización |
| `orchestration_agent_id` | 5 035 (10.5 %) | la reserva no pasó por un agente de orquestación |
| `client_region` | 20 (0.04 %) | región desconocida |
| `num_auxiliary_processes` | 1 | faltante puntual |

### 1.2 Data leakage: columnas que se eliminan

Antes de modelar revisamos qué columnas no van a estar disponibles al momento de predecir o que codifican el target:

| columna | evidencia | decisión |
|:--|:--|:--|
| `execution_status` | es una re-codificación exacta del target: `canceled`/`no_show` ⇒ 1 y `completed` ⇒ 0 en el 100 % de los casos (tabla `01_leakage_execution_status`) | **eliminar** |
| `execution_status_date` | en las reservas no canceladas es la fecha de finalización (0 a 4 días *después* del inicio); en las canceladas es la fecha de cancelación (mediana 53 días *antes* del inicio). Separa las clases casi perfectamente y no se conoce al reservar | **eliminar** |

El resto de las columnas son las de la consigna y se usan como features. Hay dos casos que conviene mencionar:

* `assigned_gpu_type` y `queue_wait_time` describen la asignación final de recursos. En un sistema real podrían conocerse recién cerca de la ejecución, lo que sería *leakage temporal*. Cuando la GPU asignada difiere de la reservada, la tasa de cancelación cae a **6 %** (probablemente porque el cambio se registra al ejecutar). Como la consigna las define como variables del problema y la función predictora va a recibir esas columnas, **se mantienen**, pero hay que tenerlo en cuenta al interpretar los resultados.
* `total_processes` y `total_gpu_hours` son exactamente la suma de sus componentes (verificado) y `high_value_client` está asociado al costo por hora (4.55 vs 7.56 USD/h). Son columnas derivadas sin información nueva, pero no tienen leakage. Se mantienen.

### 1.3 Split train/test (antes de cualquier ajuste)

* **Hold-out 80/20** estratificado por el target: 38 274 filas de train y 9 568 de test, ambos con 41.7 % de cancelación.
* **Agrupado por duplicados (`StratifiedGroupKFold`):** cada combinación única de features recibe un `group id` y todas las copias de un grupo caen del mismo lado del split. Sin esto, un tercio del test serían filas que ya están en train, el modelo "las recordaría" y la métrica estimada estaría inflada. Verificamos que hay **0 combinaciones de features compartidas** entre train y test.
* Todo el EDA que sigue y todos los pasos de modelado usan **sólo train**. El test se usa una única vez, en el Paso 5.

### 1.4 Limpieza y tratamiento de faltantes

La limpieza se separa en dos tipos, según si aprende algo de los datos o no:

1. **Determinística, fila a fila (`tp3.data.clean`):** no aprende nada de los datos, así que puede aplicarse antes del split y también en inferencia (la usa la función predictora).
   * elimina `execution_status` y `execution_status_date`;
   * categóricas: `NaN` → `"missing"`;
   * IDs de agente y organización: se convierten a **categoría** (son identificadores, no magnitudes: el agente 240 no es "más" que el 9). `NaN` → `"none"`, que significa "sin agente" o "sin organización". Eso en sí es informativo: sin agente se cancela 33 % y con organización 24 %, contra 43 % en el resto.
2. **Aprendida, dentro del `Pipeline` de sklearn (`tp3.preprocessing`):** se ajusta sólo con el fold de entrenamiento en cada iteración de CV, así no hay leakage entre folds.
   * numéricas: `SimpleImputer(median)` (cubre el único faltante de `num_auxiliary_processes`) + `StandardScaler`;
   * categóricas: `OneHotEncoder` con `min_frequency`. Las categorías raras (< 20 casos; < 50 en `client_region`, que tiene 149 países; < 100 en los IDs) se agrupan en una categoría `infrequent`. Así se evitan columnas casi vacías y se resuelven las categorías no vistas en test (`handle_unknown="infrequent_if_exist"`).

**Anomalías detectadas** (tabla `01_anomalias`):

| anomalía | filas | % cancelación | decisión |
|:--|--:|--:|:--|
| `total_processes == 0` (workload sin procesos) | 108 | 10 % | se mantienen. Son inconsistentes, pero tienen un patrón propio (casi no cancelan) y pueden aparecer en datos nuevos. En el Paso 4 se evalúa sacarlas del entrenamiento |
| `total_gpu_hours == 0` | 193 | 6 % | ídem |
| `estimated_cost_per_hour == 0` | 727 | 11 % | se mantienen: reservas sin costo (cortesía o contratos especiales) |
| `estimated_cost_per_hour > 20` | 2 | – | outlier extremo (270 USD/h contra una mediana de 5). No afecta a los árboles y con `StandardScaler` es un único punto, así que se mantiene |
| `client_type` / `request_channel` = `"unknown"` | 1 | – | queda absorbida por la categoría `infrequent` del one-hot |

**Duplicados.** No se eliminan, porque representan reservas reales cargadas en lote (reservas grupales) y su tasa de cancelación es muy distinta (61 % vs 28 %). Eliminarlos cambiaría la distribución que vamos a encontrar en producción. Lo que sí se hace es controlar su efecto en la evaluación con el split agrupado (1.3). En el Paso 4 se prueba entrenar sin duplicados como estrategia data-centered.

### 1.5 Análisis exploratorio

![target](reports/figures/01_target.png)

**Cohortes y subgrupos.** Tasa de cancelación por valor de cada variable categórica. La línea punteada es la tasa global.

![cohortes](reports/figures/01_cancel_rate_cohortes.png)

* **`reservation_commitment` es la variable más fuerte:** `locked_allocation` se cancela el **~100 %** de las veces (6 316 reservas de train) y `no_commitment` el 30 %. Se esperaría que un compromiso firme baje la cancelación, pero acá pasa lo contrario. La tabla cruzada con `client_type` sugiere una explicación: el `locked_allocation` se concentra en `research_consortium` (3 553 reservas, 99.9 % canceladas) y `managed_account` (2 523, 100 %), que son bloques reservados en lote y cancelados en masa. Esto explica también buena parte de los duplicados.
* **`client_type`:** `research_consortium` cancela 68 %, `managed_account` 43 % y `cloud_marketplace` 38 %. Los clientes directos (`independent_client` 16 %, `enterprise_client` 23 %) cancelan mucho menos.
* **`service_tier`** y **`reserved_gpu_type`** discriminan poco por sí solos: 37–43 % y 24–44 % respectivamente.
* **`client_region`:** PRT (39 % de train) cancela **65 %** y el resto de Europa entre 17 y 30 %.
* **`request_channel`:** `broker_platform` cancela 45 % y el canal directo/API 17–21 %.
* **GPU asignada ≠ reservada:** 6 % de cancelación (ver la advertencia sobre leakage temporal en 1.2).

Cruce `client_type` × `reservation_commitment` (tasa de cancelación / n en train):

| client_type | locked_allocation | no_commitment |
|:--|--:|--:|
| cloud_marketplace | 0.85 (13) | 0.38 (18 546) |
| managed_account | 1.00 (2 523) | 0.19 (5 910) |
| research_consortium | 1.00 (3 553) | 0.30 (2 973) |
| enterprise_client | 0.95 (215) | 0.11 (1 229) |
| independent_client | 0.92 (12) | 0.15 (2 911) |

Una vez que se controla por `reservation_commitment`, el ranking de cancelación entre tipos de cliente cambia: **dentro de `no_commitment`, `cloud_marketplace` es el que más cancela (38 %)**. Esto anticipa dónde van a estar los errores difíciles: reservas sin compromiso hechas por marketplace.

**Variables numéricas** (tasa de cancelación por decil o por valor):

![numericas](reports/figures/01_cancel_rate_numericas.png)

* `hours_before_execution` (anticipación) es la numérica más informativa. La cancelación crece de 12 % (≤ 4 h) a 72 % (> 276 h): cuanto más lejos está la ejecución, más probable es que el cliente cancele.
* `special_resource_requests`: 55 % de cancelación sin pedidos especiales contra ~20 % con al menos uno. Pedir recursos especiales indica compromiso con el workload.
* `previous_job_cancellations = 1` → 96 % de cancelación. `successful_previous_jobs ≥ 1` → < 10 %. El historial del cliente es muy predictivo, pero sólo el 6 % de las filas tiene historial.
* `queue_wait_time > 0` → 60–90 % de cancelación, aunque afecta sólo al 4 % de las reservas.
* `resource_modifications ≥ 1` → 15–25 % de cancelación.
* `estimated_cost_per_hour` y `execution_week` tienen poca relación marginal con el target. Pueden aportar en interacción con otras variables.

![distribuciones](reports/figures/01_distribuciones_por_clase.png)

**Correlaciones** (Spearman):

![correlaciones](reports/figures/01_correlaciones.png)

Las variables más asociadas al target son `special_resource_requests` (−0.32), `hours_before_execution` (+0.30), `previous_job_cancellations` (+0.27) y `resource_modifications` (−0.19). Hay redundancias esperables: `execution_month` y `execution_week` (ρ = 1.00), `total_*` con sus componentes y `returning_client` con `successful_previous_jobs` (0.79). Para modelos lineales con regularización y para árboles esto no es un problema grave. En el Paso 4 se evalúa la selección de variables.

**Temporal:**

![temporal](reports/figures/01_temporal.png)

La tasa de cancelación varía entre 30 % y 50 % según el mes, sin una estacionalidad muy marcada. Julio de 2015 (el comienzo del período) es atípico (69 %). Como el test se toma al azar del mismo período, `execution_year` se puede usar como feature. Si el objetivo fuera predecir años futuros habría que hacer un split temporal.

### 1.6 Conclusiones del EDA que guían los pasos siguientes

1. Se eliminan 2 columnas con leakage y el split es **agrupado por duplicados** para que la evaluación sea honesta.
2. El problema tiene señal fuerte en pocas variables: compromiso, tipo de cliente, región, anticipación, pedidos especiales e historial. Varias de esas relaciones son **no lineales y con interacciones** (por ejemplo `client_type × commitment`), así que se espera que los modelos basados en árboles superen a los lineales.
3. Las cohortes `client_type`, `service_tier` y `reserved_gpu_type` tienen tasas de cancelación muy distintas. Ahí se va a centrar el análisis de errores (Paso 3).
4. El desbalance es moderado (42/58), así que para el baseline no hace falta re-muestrear. Sí conviene elegir métricas que no dependan sólo de la accuracy (Paso 2.3).

---

## Paso 2 – Model-centered ML: baseline, validación cruzada y métricas

**Script:** `python steps/02_baseline.py` (~8 min con 4 cores) → `reports/tables/02_*`, `reports/figures/02_*`, `models/baseline.json`.

### 2.2 Esquema de validación cruzada

Este punto va primero porque todo lo demás depende de él.

* **5-fold CV estratificado y agrupado (`StratifiedGroupKFold`, `shuffle=True`, semilla 42), sólo sobre el conjunto de train.** Estratificar mantiene el 41.7 % de cancelaciones en cada fold. Agrupar garantiza que las copias idénticas de una misma reserva no queden repartidas entre entrenamiento y validación (ver 1.3).
* Todo el preprocesamiento (imputación, escalado, one-hot) vive **dentro del `Pipeline`**, así que se re-ajusta en cada fold con los datos de entrenamiento de ese fold. Ninguna estadística de validación se filtra al entrenamiento.
* Se usan **los mismos 5 folds para todos los modelos y experimentos** del TP (`tp3.evaluation.get_cv`). Así las comparaciones son pareadas y las diferencias se deben al modelo y no al azar del split.
* Se reporta **media ± desvío estándar** entre folds, tanto en validación como en train (para diagnosticar overfitting).
* Además de las métricas, la CV propia (`tp3.evaluation.cross_validate`) devuelve las **predicciones out-of-fold (OOF)** de cada fila de train. Esas predicciones son las que se usan para el análisis de errores del Paso 3, sin tocar el test.

### 2.3 Métricas elegidas y justificación

| métrica | por qué |
|:--|:--|
| **ROC-AUC** | Mide qué tan bien el modelo **ordena** las reservas por riesgo de cancelación, sin depender de un umbral. Esto importa en este problema: la operación puede elegir el umbral según la capacidad disponible (por ejemplo, sobre-reservar sólo el top-k de riesgo). Además es insensible al desbalance moderado (42/58) y, al ser continua, es más estable para comparar hiperparámetros que una métrica umbralizada. |
| **F1 de la clase "cancelada"** | Mide la calidad de las **decisiones** al umbral de operación (0.5). Combina *precision* (de las reservas que marcamos como canceladas, cuántas lo son; un FP lleva a liberar o sobre-vender capacidad de un cliente que sí viene) y *recall* (cuántas cancelaciones detectamos; un FN deja GPU reservada ociosa). La accuracy sola no alcanza: un clasificador trivial que nunca predice cancelación ya obtiene 58 % de accuracy y F1 = 0. |

**Cómo se usan:**

* **Comparar modelos:** se reportan las dos (más accuracy, precision y recall como contexto).
* **Seleccionar hiperparámetros:** `GridSearchCV(scoring={"roc_auc", "f1"}, refit="roc_auc")`. ROC-AUC es el criterio principal porque es independiente del umbral y menos ruidosa. F1 se mira como verificación.
* **Decidir el baseline final:** se elige el modelo con mejor ROC-AUC y se exige que también tenga el mejor F1 (el script lo verifica con un `assert`).

### 2.1 Clasificadores y calibración de hiperparámetros

Se calibraron tres clasificadores vistos en clase, todos dentro del mismo `Pipeline` de preprocesamiento (el árbol sin escalado, porque no lo necesita):

| modelo | grilla | mejor configuración |
|:--|:--|:--|
| Regresión Logística (L2) | `C ∈ {0.001, …, 100}` | `C = 0.1` |
| k-NN | `k ∈ {5, 15, 31, 51, 101}` × `weights ∈ {uniform, distance}` | `k = 31, weights = distance` |
| Árbol de Decisión | `max_depth ∈ {3, 5, 8, 12, 16, 20, None}` × `min_samples_leaf ∈ {1, 5, 20, 50}` | `max_depth = 8, min_samples_leaf = 1` |

![curvas de validación](reports/figures/02_curvas_validacion_baseline.png)

Qué muestran las curvas de validación:

* **Regresión logística:** prácticamente plana a partir de `C = 0.01`, con train y validación casi iguales (0.907 vs 0.898). Es un modelo de **alto sesgo**: la frontera lineal no captura las interacciones vistas en el EDA.
* **k-NN:** con `weights=distance` el score de train es 1.0 por construcción (cada punto es su propio vecino). En validación el óptimo está en k ≈ 31. Sufre con las ~100 dimensiones del one-hot (maldición de la dimensionalidad).
* **Árbol:** muestra el *trade-off* sesgo-varianza clásico. Hasta `max_depth = 8` mejoran train y validación. A partir de ahí train sigue subiendo hasta 1.0, mientras validación cae de 0.910 a 0.80 sin límite de profundidad (overfitting).

### Resultado: comparación con 5-fold CV

| modelo | ROC-AUC | F1 | accuracy | precision | recall | train ROC-AUC |
|:--|:--|:--|:--|:--|:--|--:|
| Dummy (clase mayoritaria) | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5829 | 0.0000 | 0.0000 | 0.500 |
| Regresión Logística (C=0.1) | 0.8982 ± 0.0055 | 0.7627 ± 0.0098 | 0.8148 | 0.8196 | 0.7134 | 0.907 |
| k-NN (k=31, distance) | 0.8784 ± 0.0110 | 0.7544 ± 0.0138 | 0.8068 | 0.8026 | 0.7117 | 1.000 |
| **Árbol de Decisión (depth=8)** | **0.9100 ± 0.0035** | **0.7901 ± 0.0061** | **0.8357** | **0.8459** | **0.7411** | 0.925 |

![comparación baseline](reports/figures/02_comparacion_baseline.png)

**Baseline elegido: Árbol de Decisión (`max_depth=8`).** Es el mejor en las dos métricas, con la menor varianza entre folds y una brecha train/validación chica (0.925 vs 0.910). Además es interpretable, lo que ayuda en el análisis de errores. Este es el punto de referencia que cada estrategia del Paso 4 tiene que superar: **ROC-AUC 0.910 / F1 0.790**.

Un patrón que aparece en los tres modelos: **la precision (≈ 0.82–0.85) es mayor que el recall (≈ 0.71–0.74)**. Con umbral 0.5, el modelo deja pasar más de una de cada cuatro cancelaciones (FN). Esto se analiza en el Paso 3.
