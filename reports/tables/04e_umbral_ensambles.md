| modelo                                   | F1 umbral 0.5   | F1 umbral ajustado   |   umbral medio |
|:-----------------------------------------|:----------------|:---------------------|---------------:|
| LightGBM (tuneado)                       | 0.8242 ± 0.0141 | 0.8321 ± 0.0075      |         0.4100 |
| XGBoost (tuneado)                        | 0.8212 ± 0.0146 | 0.8235 ± 0.0114      |         0.3800 |
| Random Forest (tuneado)                  | 0.8222 ± 0.0127 | 0.8290 ± 0.0091      |         0.4200 |
| Regresión Logística + FE                 | 0.7791 ± 0.0060 | 0.7928 ± 0.0124      |         0.3800 |
| Bagging de semillas LightGBM (×5)        | 0.8243 ± 0.0131 | 0.8268 ± 0.0126      |         0.3700 |
| Soft voting (LGBM + XGB + RF)            | 0.8262 ± 0.0116 | 0.8321 ± 0.0096      |         0.4100 |
| Soft voting (LGBM + XGB + RF + LR)       | 0.8239 ± 0.0120 | 0.8253 ± 0.0104      |         0.4100 |
| Stacking (LGBM + XGB + RF + LR → LogReg) | 0.8260 ± 0.0109 | 0.8312 ± 0.0091      |         0.4200 |