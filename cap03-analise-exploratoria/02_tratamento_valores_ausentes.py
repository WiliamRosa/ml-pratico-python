# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 03 Analise Exploratoria

Arquivo: 02_tratamento_valores_ausentes.py
"""

# =============================================================================
# 3.3 Tratamento de Dados Ausentes
# =============================================================================

# --- Remover duplicatas primeiro ---
df_limpo = df_credito.drop_duplicates().reset_index(drop=True)
print(f"Após remover duplicatas: {df_limpo.shape[0]} registros (removidos {len(df_credito) - len(df_limpo)})")

# --- Visualizar padrão de nulos ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Gráfico de barras de nulos
nulos = df_limpo.isnull().sum()
nulos[nulos > 0].sort_values().plot.barh(ax=axes[0], color="coral")
axes[0].set_title("Contagem de Valores Ausentes")
axes[0].set_xlabel("Número de nulos")

# Verificar se nulos estão correlacionados com o target
for col in ["renda_mensal", "tempo_emprego_anos", "score_credito"]:
    nulo = df_limpo[df_limpo[col].isnull()]["inadimplente"].mean()
    presente = df_limpo[df_limpo[col].notna()]["inadimplente"].mean()
    axes[1].barh([f"{col}\n(nulo)", f"{col}\n(presente)"],
                 [nulo, presente],
                 color=["coral", "steelblue"])

axes[1].set_title("Taxa de Inadimplência: Nulo vs Presente")
axes[1].set_xlabel("Taxa de inadimplência")
plt.tight_layout()
plt.show()

# --- Estratégias de imputação ---
from sklearn.impute import SimpleImputer, KNNImputer

# 1. Mediana para renda (distribuição assimétrica)
mediana_renda = df_limpo["renda_mensal"].median()
df_limpo["renda_mensal_imputada"] = df_limpo["renda_mensal"].fillna(mediana_renda)

# 2. Média para score_credito (distribuição mais simétrica)
media_score = df_limpo["score_credito"].mean()
df_limpo["score_credito_imputado"] = df_limpo["score_credito"].fillna(round(media_score))

# 3. Mediana para tempo_emprego
df_limpo["tempo_emprego_imputado"] = df_limpo["tempo_emprego_anos"].fillna(
    df_limpo["tempo_emprego_anos"].median()
)

# 4. Criar indicadores de nulo (a ausência pode ser informativa)
for col in ["renda_mensal", "score_credito", "tempo_emprego_anos"]:
    df_limpo[f"{col}_ausente"] = df_limpo[col].isnull().astype(int)

print("\nApós imputação:")
print(f"  Nulos restantes: {df_limpo.isnull().sum().sum()}")
print(f"  Indicadores criados: {[c for c in df_limpo.columns if '_ausente' in c]}")
