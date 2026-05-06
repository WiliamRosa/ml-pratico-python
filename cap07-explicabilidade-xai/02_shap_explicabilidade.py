# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 07 Explicabilidade Xai

Arquivo: 02_shap_explicabilidade.py
"""

# =============================================================================
# 7.4 SHAP - Explicabilidade Local e Global
# =============================================================================
try:
    import shap
    
    # Criar explainer para tree-based models
    explainer = shap.TreeExplainer(gbr)
    shap_values = explainer.shap_values(X_teste)
    
    # --- Summary Plot (global) ---
    print("=== SHAP Summary Plot ===")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_teste, show=False)
    plt.title("SHAP Summary Plot - Impacto Global das Features")
    plt.tight_layout()
    plt.show()
    
    # --- Explicação local: um paciente específico ---
    idx = 0  # primeiro paciente do teste
    print(f"\n=== Explicação Local - Paciente {idx} ===")
    print(f"Valor real: {y_teste.iloc[idx]:.1f}")
    print(f"Valor previsto: {y_pred[idx]:.1f}")
    print(f"Valor base (média): {explainer.expected_value:.1f}")
    
    # Force plot (em texto)
    top_shap = pd.DataFrame({
        "feature": X.columns,
        "valor": X_teste.iloc[idx].values,
        "shap_value": shap_values[idx]
    }).sort_values("shap_value", key=abs, ascending=False)
    
    print("\nContribuições SHAP:")
    for _, row in top_shap.head(5).iterrows():
        direcao = "(+)" if row["shap_value"] > 0 else "(-)"
        print(f"  {row['feature']:<8}: valor={row['valor']:.3f}, SHAP={row['shap_value']:+.2f} {direcao}")

except ImportError:
    print("[NOTA] Biblioteca SHAP não instalada.")
    print("Para instalar: %pip install shap")
    print("\nSHAP (SHapley Additive exPlanations) utiliza valores Shapley")
    print("da teoria dos jogos para atribuir a contribuição de cada feature")
    print("para cada predição individual, fornecendo explicabilidade local e global.")
