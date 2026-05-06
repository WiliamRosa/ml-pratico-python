# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 05 Pipelines Ml

Arquivo: 02_curvas_roc_precisao_recall.py
"""

# =============================================================================
# 5.3 Curvas ROC e Precisão-Recall
# =============================================================================
pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest (tuned)": grid.best_estimator_,
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ])
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cores = ["steelblue", "coral", "seagreen", "purple"]

for (nome, pipe), cor in zip(pipelines.items(), cores):
    if nome != "Random Forest (tuned)":
        pipe.fit(X_treino, y_treino)
    y_proba = pipe.predict_proba(X_teste)[:, 1]
    auc = roc_auc_score(y_teste, y_proba)
    
    fpr, tpr, _ = roc_curve(y_teste, y_proba)
    axes[0].plot(fpr, tpr, color=cor, label=f"{nome} (AUC={auc:.3f})")
    
    prec, rec, _ = precision_recall_curve(y_teste, y_proba)
    f1 = f1_score(y_teste, pipe.predict(X_teste))
    axes[1].plot(rec, prec, color=cor, label=f"{nome} (F1={f1:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("Falso Positivo"); axes[0].set_ylabel("Verdadeiro Positivo")
axes[0].set_title("Curva ROC"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precisão")
axes[1].set_title("Curva Precisão-Recall"); axes[1].legend(fontsize=8)
plt.suptitle("Comparação de Modelos - Câncer de Mama", fontweight="bold")
plt.tight_layout()
plt.show()
