# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 02 Numpy Pandas Visualizacao

Arquivo: 04_visualizacao_matplotlib_seaborn.py
"""

# =============================================================================
# 2.3 Visualização de Dados - Matplotlib e Seaborn
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Análise Exploratória - Dataset de Risco Cardíaco", fontsize=14, fontweight="bold")

# 1. Distribuição da idade por risco
sns.histplot(data=df, x="idade", hue="risco_cardiaco", kde=True, ax=axes[0, 0], bins=25)
axes[0, 0].set_title("Distribuição de Idade por Risco")
axes[0, 0].set_xlabel("Idade")

# 2. Boxplot de pressão por atividade física
sns.boxplot(data=df, x="atividade_fisica", y="pressao_sistolica", hue="risco_cardiaco", ax=axes[0, 1])
axes[0, 1].set_title("Pressão Sistólica por Atividade Física")
axes[0, 1].set_xlabel("Atividade Física")

# 3. Scatter: IMC vs Pressão
sns.scatterplot(data=df, x="imc", y="pressao_sistolica", hue="risco_cardiaco",
                alpha=0.5, ax=axes[0, 2])
axes[0, 2].set_title("IMC vs Pressão Sistólica")

# 4. Contagem por categoria de IMC
sns.countplot(data=df, x="categoria_imc", hue="risco_cardiaco", ax=axes[1, 0],
              order=["abaixo_peso", "normal", "sobrepeso", "obeso"])
axes[1, 0].set_title("Distribuição por Categoria de IMC")
axes[1, 0].tick_params(axis="x", rotation=15)

# 5. Heatmap de correlação
# Selecionar apenas colunas numéricas para a matriz de correlação
features_num = df.select_dtypes(include=np.number).columns.tolist()
corr = df[features_num].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=axes[1, 1], cbar_kws={"shrink": 0.8})
axes[1, 1].set_title("Matriz de Correlação")

# 6. Violin plot: glicose por fumante
sns.violinplot(data=df, x="fumante", y="glicose", hue="risco_cardiaco",
               split=True, ax=axes[1, 2])
axes[1, 2].set_title("Glicose por Hábito de Fumar")

plt.tight_layout()
plt.show()
