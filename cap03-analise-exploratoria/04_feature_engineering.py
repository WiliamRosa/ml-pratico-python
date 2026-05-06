# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 03 Analise Exploratoria

Arquivo: 04_feature_engineering.py
"""

# =============================================================================
# 3.5 Feature Engineering
# =============================================================================

# --- Razões e proporções ---
df_limpo["parcela_mensal"] = df_limpo["valor_emprestimo"] / df_limpo["num_parcelas"]
df_limpo["comprometimento_renda"] = (df_limpo["parcela_mensal"] / df_limpo["renda_tratada"]).clip(0, 1)
df_limpo["emprestimo_por_renda"] = df_limpo["valor_emprestimo"] / df_limpo["renda_tratada"]

# --- Transformação logarítmica (para renda - distribuição assimétrica) ---
df_limpo["log_renda"] = np.log1p(df_limpo["renda_tratada"])
df_limpo["log_valor_emprestimo"] = np.log1p(df_limpo["valor_emprestimo"])

# --- Encoding ordinal (escolaridade tem ordem natural) ---
mapa_escolaridade = {"fundamental": 1, "medio": 2, "superior": 3, "pos_graduacao": 4}
df_limpo["escolaridade_num"] = df_limpo["escolaridade"].map(mapa_escolaridade)

# --- Binning: faixas de score ---
df_limpo["faixa_score"] = pd.cut(
    df_limpo["score_credito_imputado"],
    bins=[0, 400, 550, 700, 900],
    labels=["muito_baixo", "baixo", "medio", "alto"]
)

# --- Validar novas features ---
print("=== Novas Features Criadas ===")
novas_features = ["parcela_mensal", "comprometimento_renda", "emprestimo_por_renda",
                  "log_renda", "escolaridade_num", "faixa_score"]

for feat in novas_features:
    if df_limpo[feat].dtype in ["float64", "int64", "float32"]:
        print(f"  {feat:<25}: mean={df_limpo[feat].mean():.2f}, std={df_limpo[feat].std():.2f}")
    else:
        print(f"  {feat:<25}: {df_limpo[feat].value_counts().to_dict()}")

# --- Correlação das novas features com o target ---
print("\n=== Correlacao com Target (inadimplente) ===")
features_para_corr = ["idade", "renda_tratada", "score_credito_imputado",
                      "comprometimento_renda", "emprestimo_por_renda",
                      "escolaridade_num", "tempo_emprego_imputado"]

corr_target = df_limpo[features_para_corr + ["inadimplente"]].corr()["inadimplente"].drop("inadimplente")
corr_target = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)

for feat, corr in corr_target.items():
    barra = "#" * int(abs(corr) * 50)
    sinal = "+" if corr > 0 else "-"
    print(f"  {feat:<30}: {sinal}{abs(corr):.3f} {barra}")
