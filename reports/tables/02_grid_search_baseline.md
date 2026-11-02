| modelo              | params                                      |   val_roc_auc |   val_roc_auc_std |   val_f1 |   train_roc_auc |
|:--------------------|:--------------------------------------------|--------------:|------------------:|---------:|----------------:|
| Regresión Logística | {"C": 0.1}                                  |        0.8982 |            0.0049 |   0.7627 |          0.9071 |
| Regresión Logística | {"C": 0.01}                                 |        0.8971 |            0.0060 |   0.7585 |          0.9018 |
| Regresión Logística | {"C": 1}                                    |        0.8963 |            0.0052 |   0.7648 |          0.9082 |
| Regresión Logística | {"C": 10}                                   |        0.8955 |            0.0056 |   0.7651 |          0.9083 |
| Regresión Logística | {"C": 100}                                  |        0.8954 |            0.0056 |   0.7644 |          0.9083 |
| Regresión Logística | {"C": 0.001}                                |        0.8838 |            0.0080 |   0.7382 |          0.8867 |
| k-NN                | {"n_neighbors": 31, "weights": "distance"}  |        0.8784 |            0.0099 |   0.7544 |          1.0000 |
| k-NN                | {"n_neighbors": 51, "weights": "distance"}  |        0.8769 |            0.0134 |   0.7469 |          1.0000 |
| k-NN                | {"n_neighbors": 101, "weights": "distance"} |        0.8750 |            0.0153 |   0.7425 |          1.0000 |
| k-NN                | {"n_neighbors": 15, "weights": "distance"}  |        0.8744 |            0.0081 |   0.7502 |          0.9999 |
| k-NN                | {"n_neighbors": 31, "weights": "uniform"}   |        0.8739 |            0.0109 |   0.7458 |          0.9174 |
| k-NN                | {"n_neighbors": 51, "weights": "uniform"}   |        0.8712 |            0.0146 |   0.7417 |          0.9077 |
| k-NN                | {"n_neighbors": 15, "weights": "uniform"}   |        0.8706 |            0.0085 |   0.7469 |          0.9322 |
| k-NN                | {"n_neighbors": 101, "weights": "uniform"}  |        0.8679 |            0.0164 |   0.7327 |          0.8961 |
| k-NN                | {"n_neighbors": 5, "weights": "distance"}   |        0.8529 |            0.0133 |   0.7414 |          0.9996 |
| k-NN                | {"n_neighbors": 5, "weights": "uniform"}    |        0.8482 |            0.0139 |   0.7408 |          0.9601 |
| Árbol de Decisión   | {"max_depth": 8, "min_samples_leaf": 1}     |        0.9100 |            0.0032 |   0.7899 |          0.9254 |
| Árbol de Decisión   | {"max_depth": 8, "min_samples_leaf": 20}    |        0.9095 |            0.0041 |   0.7886 |          0.9239 |
| Árbol de Decisión   | {"max_depth": 8, "min_samples_leaf": 5}     |        0.9092 |            0.0052 |   0.7829 |          0.9254 |
| Árbol de Decisión   | {"max_depth": 12, "min_samples_leaf": 20}   |        0.9089 |            0.0064 |   0.7852 |          0.9439 |
| Árbol de Decisión   | {"max_depth": 16, "min_samples_leaf": 50}   |        0.9087 |            0.0075 |   0.7830 |          0.9429 |
| Árbol de Decisión   | {"max_depth": null, "min_samples_leaf": 50} |        0.9085 |            0.0075 |   0.7836 |          0.9443 |
| Árbol de Decisión   | {"max_depth": 12, "min_samples_leaf": 50}   |        0.9085 |            0.0070 |   0.7824 |          0.9379 |
| Árbol de Decisión   | {"max_depth": 8, "min_samples_leaf": 50}    |        0.9085 |            0.0045 |   0.7856 |          0.9221 |
| Árbol de Decisión   | {"max_depth": 20, "min_samples_leaf": 50}   |        0.9084 |            0.0076 |   0.7832 |          0.9442 |
| Árbol de Decisión   | {"max_depth": 16, "min_samples_leaf": 20}   |        0.9047 |            0.0061 |   0.7804 |          0.9535 |
| Árbol de Decisión   | {"max_depth": 20, "min_samples_leaf": 20}   |        0.9028 |            0.0067 |   0.7798 |          0.9588 |
| Árbol de Decisión   | {"max_depth": 12, "min_samples_leaf": 5}    |        0.9020 |            0.0086 |   0.7815 |          0.9511 |
| Árbol de Decisión   | {"max_depth": null, "min_samples_leaf": 20} |        0.9014 |            0.0068 |   0.7780 |          0.9605 |
| Árbol de Decisión   | {"max_depth": 12, "min_samples_leaf": 1}    |        0.8989 |            0.0086 |   0.7904 |          0.9541 |
| Árbol de Decisión   | {"max_depth": 5, "min_samples_leaf": 20}    |        0.8957 |            0.0052 |   0.7469 |          0.8988 |
| Árbol de Decisión   | {"max_depth": 5, "min_samples_leaf": 50}    |        0.8956 |            0.0053 |   0.7469 |          0.8987 |
| Árbol de Decisión   | {"max_depth": 5, "min_samples_leaf": 1}     |        0.8956 |            0.0051 |   0.7477 |          0.8993 |
| Árbol de Decisión   | {"max_depth": 5, "min_samples_leaf": 5}     |        0.8943 |            0.0081 |   0.7394 |          0.8993 |
| Árbol de Decisión   | {"max_depth": 16, "min_samples_leaf": 5}    |        0.8876 |            0.0101 |   0.7737 |          0.9676 |
| Árbol de Decisión   | {"max_depth": 20, "min_samples_leaf": 5}    |        0.8769 |            0.0094 |   0.7732 |          0.9789 |
| Árbol de Decisión   | {"max_depth": 16, "min_samples_leaf": 1}    |        0.8707 |            0.0081 |   0.7802 |          0.9768 |
| Árbol de Decisión   | {"max_depth": 3, "min_samples_leaf": 5}     |        0.8629 |            0.0047 |   0.7101 |          0.8630 |
| Árbol de Decisión   | {"max_depth": 3, "min_samples_leaf": 1}     |        0.8628 |            0.0047 |   0.7101 |          0.8630 |
| Árbol de Decisión   | {"max_depth": 3, "min_samples_leaf": 20}    |        0.8628 |            0.0048 |   0.7100 |          0.8629 |
| Árbol de Decisión   | {"max_depth": 3, "min_samples_leaf": 50}    |        0.8628 |            0.0048 |   0.7100 |          0.8628 |
| Árbol de Decisión   | {"max_depth": null, "min_samples_leaf": 5}  |        0.8619 |            0.0090 |   0.7608 |          0.9870 |
| Árbol de Decisión   | {"max_depth": 20, "min_samples_leaf": 1}    |        0.8441 |            0.0105 |   0.7774 |          0.9911 |
| Árbol de Decisión   | {"max_depth": null, "min_samples_leaf": 1}  |        0.7986 |            0.0099 |   0.7637 |          1.0000 |