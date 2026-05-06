# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 02 Numpy Pandas Visualizacao

Arquivo: 03_pandas_transformacoes.py
"""

# =============================================================================
# Transformações e Feature Engineering com Pandas
# =============================================================================

# --- Criar novas features ---
df["imc"] = (df["peso"] / (df["altura"] ** 2)).round(1)

# Categorizar IMC
def categorizar_imc(imc):
    if pd.isna(imc):
        return "desconhecido"
    elif imc < 18.5:
        return "abaixo_peso"
    elif imc < 25:
        return "normal"
    elif imc < 30:
        return "sobrepeso"
    else:
        return "obeso"

df["categoria_imc"] = df["imc"].apply(categorizar_imc)

# Faixa etária
df["faixa_etaria"] = pd.cut(
    df["idade"],
    bins=[0, 30, 45, 60, 100],
    labels=["jovem", "adulto", "meia_idade", "idoso"]
)

# --- Agregações: análise por grupo ---
print("=== Risco Cardíaco por Faixa Etária ===")
analise_faixa = df.groupby("faixa_etaria").agg(
    n_pacientes=("idade", "count"),
    idade_media=("idade", "mean"),
    pressao_media=("pressao_sistolica", "mean"),
    taxa_risco=("risco_cardiaco", "mean")
).round(2)
print(analise_faixa)

# --- Filtragem combinada ---
pacientes_risco = df[
    (df["idade"] > 60) &
    (df["fumante"] == "sim") &
    (df["pressao_sistolica"] > 140)
]
print(f"\nPacientes alto risco (>60 anos, fumantes, hipertensos): {len(pacientes_risco)}")

# --- Tabela cruzada ---
print("\n=== Tabela Cruzada: Fumante x Risco ===")
tab_cruzada = pd.crosstab(
    df["fumante"],
    df["risco_cardiaco"],
    margins=True,
    normalize="index"
).round(3)
print(tab_cruzada)

# --- Preparar dados para ML ---
print("\n=== Preparação para ML ===")
# Separar features numéricas e categóricas
features_num = df.select_dtypes(include=["number"]).columns.drop("risco_cardiaco").tolist()
features_cat = ["fumante", "atividade_fisica", "categoria_imc"]

print(f"Features numéricas ({len(features_num)}): {features_num}")
print(f"Features categóricas ({len(features_cat)}): {features_cat}")
print(f"Target: risco_cardiaco (distribuição: {df['risco_cardiaco'].value_counts().to_dict()})")
