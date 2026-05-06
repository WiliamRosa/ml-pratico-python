# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 04 Primeiro Modelo

Arquivo: 03_validacao_cruzada_overfitting.py
"""

# =============================================================================
# 4.4-4.5 Validação Cruzada e Diagnóstico de Overfitting
# =============================================================================
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.tree import DecisionTreeClassifier

# --- Validação Cruzada Estratificada ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== Validação Cruzada (5-Fold Estratificado) ===")
for nome, res in resultados.items():
    scores = cross_val_score(res["modelo"], X_treino_scaled, y_treino, cv=cv, scoring="accuracy")
    print(f"  {nome:<25}: {scores.mean():.4f} ± {scores.std():.4f}  (folds: {scores.round(3)})")

# --- Demonstrar Overfitting vs Underfitting ---
print("\n=== Diagnóstico: Overfitting vs Underfitting ===")

modelos_diagnostico = {
    "Decision Tree (depth=1) [Underfitting]": DecisionTreeClassifier(max_depth=1, random_state=42),
    "Decision Tree (depth=3) [Bom ajuste]": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Decision Tree (depth=None) [Overfitting]": DecisionTreeClassifier(max_depth=None, random_state=42),
}

for nome, modelo in modelos_diagnostico.items():
    modelo.fit(X_treino_scaled, y_treino)
    acc_treino = modelo.score(X_treino_scaled, y_treino)
    acc_teste = modelo.score(X_teste_scaled, y_teste)
    gap = acc_treino - acc_teste
    print(f"  {nome}")
    print(f"    Treino: {acc_treino:.4f}  |  Teste: {acc_teste:.4f}  |  Gap: {gap:.4f}")

# --- Curva de Aprendizado ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for i, (nome, modelo) in enumerate(modelos_diagnostico.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        modelo, X_treino_scaled, y_treino, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy"
    )
    
    axes[i].plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino", color="steelblue")
    axes[i].plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validação", color="coral")
    axes[i].fill_between(train_sizes,
                         train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1),
                         alpha=0.1, color="steelblue")
    axes[i].fill_between(train_sizes,
                         val_scores.mean(axis=1) - val_scores.std(axis=1),
                         val_scores.mean(axis=1) + val_scores.std(axis=1),
                         alpha=0.1, color="coral")
    axes[i].set_title(nome.split("[")[1].replace("]", ""))
    axes[i].set_xlabel("Amostras de Treino")
    axes[i].set_ylabel("Acurácia")
    axes[i].legend(loc="lower right")
    axes[i].set_ylim(0.4, 1.05)

plt.suptitle("Curvas de Aprendizado - Diagnóstico de Overfitting", fontweight="bold")
plt.tight_layout()
plt.show()
