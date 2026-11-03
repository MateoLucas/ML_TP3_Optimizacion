| modelo                                                              |   umbral |   roc_auc |     f1 |   accuracy |   precision |   recall |
|:--------------------------------------------------------------------|---------:|----------:|-------:|-----------:|------------:|---------:|
| Baseline (Árbol d=8)                                                |   0.5000 |    0.9160 | 0.7954 |     0.8400 |      0.8522 |   0.7457 |
| Final: LightGBM tuneado + FE + selección de variables + umbral 0.41 |   0.5000 |    0.9460 | 0.8315 |     0.8625 |      0.8504 |   0.8133 |
| Final: LightGBM tuneado + FE + selección de variables + umbral 0.41 |   0.4100 |    0.9460 | 0.8357 |     0.8596 |      0.8166 |   0.8557 |