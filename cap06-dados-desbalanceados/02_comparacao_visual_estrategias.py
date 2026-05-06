# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 06 Dados Desbalanceados

Arquivo: 02_comparacao_visual_estrategias.py
"""

# =============================================================================
# 6.3 Comparação Visual das Estratégias
# =============================================================================

# Comparar F1 da classe minoritária
resultados_desb = {
    "Baseline": f1_score(y_teste, y_pred_base),
    "Class Weight": f1_score(y_teste, y_pred_balanced),
}

try:
    resultados_desb["SMOTE"] = f1_score(y_teste, y_pred_smote)
    resultados_desb["SMOTE+Tomek"] = f1_score(y_teste, y_pred_st)
except NameError:
    pass

# Adicionar Random Forest com class_weight
rf_balanced = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
rf_balanced.fit(X_treino_s, y_treino)
y_pred_rf = rf_balanced.predict(X_teste_s)
resultados_desb["RF Balanced"] = f1_score(y_teste, y_pred_rf)

# Visualização
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1 Score comparativo
nomes = list(resultados_desb.keys())
scores = list(resultados_desb.values())
cores = plt.cm.viridis(np.linspace(0.3, 0.9, len(nomes)))

barras = axes[0].barh(nomes, scores, color=cores)
for barra, score in zip(barras, scores):
    axes[0].text(barra.get_width() + 0.01, barra.get_y() + barra.get_height()/2,
                 f"{score:.3f}", va="center", fontweight="bold")
axes[0].set_xlabel("F1-Score (classe fraude)")
axes[0].set_title("Comparação de Estratégias")
axes[0].set_xlim(0, 1)

# Precision-Recall para melhor modelo
y_proba_rf = rf_balanced.predict_proba(X_teste_s)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_teste, y_proba_rf)

axes[1].plot(rec, prec, "steelblue", linewidth=2)
axes[1].fill_between(rec, prec, alpha=0.1, color="steelblue")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precisão")
axes[1].set_title(f"Precisão-Recall: RF Balanced (AUC-ROC={roc_auc_score(y_teste, y_proba_rf):.3f})")

plt.suptitle("Tratamento de Dados Desbalanceados - Detecção de Fraude", fontweight="bold")
plt.tight_layout()
plt.show()
