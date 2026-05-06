# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 03 Analise Exploratoria

Arquivo: 03_deteccao_outliers.py
"""

# =============================================================================
# 3.4 Detecção e Tratamento de Outliers
# =============================================================================

# --- Método IQR ---
def detectar_outliers_iqr(serie: pd.Series, fator: float = 1.5) -> pd.Series:
    """Retorna máscara booleana de outliers pelo método IQR."""
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - fator * IQR
    limite_superior = Q3 + fator * IQR
    return (serie < limite_inferior) | (serie > limite_superior)

# Analisar outliers por feature numérica
features_numericas = ["idade", "renda_mensal_imputada", "tempo_emprego_imputado",
                      "valor_emprestimo", "score_credito_imputado"]

print("=== Detecção de Outliers (Método IQR) ===")
for feat in features_numericas:
    outliers = detectar_outliers_iqr(df_limpo[feat])
    n_outliers = outliers.sum()
    pct = n_outliers / len(df_limpo) * 100
    print(f"  {feat:<30}: {n_outliers:>3} outliers ({pct:.1f}%)")

# --- Visualizar outliers ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Renda: claramente com outliers injetados
sns.boxplot(data=df_limpo, y="renda_mensal_imputada", ax=axes[0], color="lightcoral")
axes[0].set_title("Renda Mensal (com outliers)")

# Score: distribuição mais comportada
sns.boxplot(data=df_limpo, y="score_credito_imputado", ax=axes[1], color="steelblue")
axes[1].set_title("Score de Crédito")

# Valor empréstimo
sns.boxplot(data=df_limpo, y="valor_emprestimo", ax=axes[2], color="seagreen")
axes[2].set_title("Valor do Empréstimo")

plt.tight_layout()
plt.show()

# --- Tratar outliers de renda (capping / winsorization) ---
Q1_renda = df_limpo["renda_mensal_imputada"].quantile(0.01)
Q99_renda = df_limpo["renda_mensal_imputada"].quantile(0.99)

df_limpo["renda_tratada"] = df_limpo["renda_mensal_imputada"].clip(Q1_renda, Q99_renda)

print(f"\nRenda antes do capping: min={df_limpo['renda_mensal_imputada'].min():.0f}, max={df_limpo['renda_mensal_imputada'].max():.0f}")
print(f"Renda após capping (1%-99%): min={df_limpo['renda_tratada'].min():.0f}, max={df_limpo['renda_tratada'].max():.0f}")
