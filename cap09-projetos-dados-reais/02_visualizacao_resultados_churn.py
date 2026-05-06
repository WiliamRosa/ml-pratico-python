# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 09 Projetos Dados Reais

Arquivo: 02_visualizacao_resultados_churn.py
"""

# =============================================================================
# 9.3 Visualização de Resultados
# =============================================================================

melhor_pipe = resultados_churn[melhor]["pipe"]
y_pred_final = melhor_pipe.predict(X_teste)
y_proba_final = melhor_pipe.predict_proba(X_teste)[:, 1]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Matriz de confusão
cm = confusion_matrix(y_teste, y_pred_final)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["Retido", "Churn"], yticklabels=["Retido", "Churn"])
axes[0, 0].set_title(f"Matriz de Confusao - {melhor}")
axes[0, 0].set_xlabel("Previsto")
axes[0, 0].set_ylabel("Real")

# 2. Distribuição de probabilidade por classe
df_prob = pd.DataFrame({"probabilidade_churn": y_proba_final, "real": y_teste})
sns.histplot(data=df_prob, x="probabilidade_churn", hue="real", bins=30,
             ax=axes[0, 1], palette=["steelblue", "coral"])
axes[0, 1].axvline(0.5, color="black", linestyle="--", alpha=0.5)
axes[0, 1].set_title("Distribuicao de Probabilidade por Classe")

# 3. Feature Importance (se tree-based)
try:
    clf = melhor_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        prep = melhor_pipe.named_steps["prep"]
        cat_features = prep.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(features_cat).tolist()
        all_features = features_num + cat_features
        
        imp = pd.DataFrame({
            "feature": all_features[:len(clf.feature_importances_)],
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=True).tail(10)
        
        axes[1, 0].barh(imp["feature"], imp["importance"], color="seagreen")
        axes[1, 0].set_title("Top 10 Features Mais Importantes")
except Exception:
    axes[1, 0].text(0.5, 0.5, "Feature importance\nnao disponivel", ha="center", va="center")

# 4. Comparação de modelos
nomes_modelos = list(resultados_churn.keys())
f1_scores = [resultados_churn[m]["f1"] for m in nomes_modelos]
auc_scores = [resultados_churn[m]["auc"] for m in nomes_modelos]

x_pos = np.arange(len(nomes_modelos))
width = 0.35
axes[1, 1].bar(x_pos - width/2, f1_scores, width, label="F1-Score", color="steelblue")
axes[1, 1].bar(x_pos + width/2, auc_scores, width, label="AUC-ROC", color="coral")
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels([n.replace(" ", "\n") for n in nomes_modelos], fontsize=9)
axes[1, 1].set_title("Comparacao de Modelos")
axes[1, 1].legend()
axes[1, 1].set_ylim(0, 1)

plt.suptitle("Projeto: Predicao de Churn - Resultados Finais", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
