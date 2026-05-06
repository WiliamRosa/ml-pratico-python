# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 04 Primeiro Modelo

Arquivo: 04_regressao_california_housing.py
"""

# =============================================================================
# 4.6 Regressão - Prevendo Preço de Imóveis (California Housing)
# =============================================================================
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Carregar dataset ---
housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing["preco"] = housing.target  # em centenas de milhares de dólares

print(f"Dataset: {df_housing.shape}")
print(f"\nTarget (preço):")
print(df_housing["preco"].describe().round(3))

# --- Preparar dados ---
X_h = df_housing.drop("preco", axis=1)
y_h = df_housing["preco"]

X_h_treino, X_h_teste, y_h_treino, y_h_teste = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

scaler_h = StandardScaler()
X_h_treino_s = scaler_h.fit_transform(X_h_treino)
X_h_teste_s = scaler_h.transform(X_h_teste)

# --- Treinar e comparar modelos de regressão ---
modelos_reg = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1)": Ridge(alpha=1.0),
    "Lasso (alpha=0.01)": Lasso(alpha=0.01),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
}

print("\n=== Resultados de Regressão ===")
print(f"{'Modelo':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 55)

for nome, modelo in modelos_reg.items():
    modelo.fit(X_h_treino_s, y_h_treino)
    y_pred = modelo.predict(X_h_teste_s)
    
    mae = mean_absolute_error(y_h_teste, y_pred)
    rmse = np.sqrt(mean_squared_error(y_h_teste, y_pred))
    r2 = r2_score(y_h_teste, y_pred)
    
    print(f"{nome:<25} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

# --- Visualizar predições do melhor modelo ---
melhor_reg = modelos_reg["Gradient Boosting"]
y_pred_gb = melhor_reg.predict(X_h_teste_s)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Real vs Previsto
axes[0].scatter(y_h_teste, y_pred_gb, alpha=0.3, s=10, color="steelblue")
axes[0].plot([0, 5.5], [0, 5.5], "r--", linewidth=2, label="Perfeito")
axes[0].set_xlabel("Preço Real")
axes[0].set_ylabel("Preço Previsto")
axes[0].set_title("Gradient Boosting: Real vs Previsto")
axes[0].legend()

# Distribuição dos resíduos
residuos = y_h_teste - y_pred_gb
axes[1].hist(residuos, bins=50, color="coral", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_xlabel("Resíduo (Real - Previsto)")
axes[1].set_ylabel("Frequência")
axes[1].set_title(f"Distribuição dos Resíduos (média={residuos.mean():.3f})")

plt.tight_layout()
plt.show()
