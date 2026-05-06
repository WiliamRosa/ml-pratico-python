# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 01 Fundamentos Python

Arquivo: 02_controle_de_fluxo.py
"""

# =============================================================================
# 1.3 Controle de Fluxo - Condicionais e Laços
# =============================================================================

# --- Condicionais: seleção de estratégia baseada em dados ---
n_amostras = 1500
n_classes = 3
balanceamento = [0.70, 0.20, 0.10]  # proporção por classe

# Decidir estratégia de treinamento baseada no tamanho do dataset
if n_amostras < 500:
    estrategia = "cross-validation com k=10 (dataset pequeno)"
elif n_amostras < 5000:
    estrategia = "cross-validation com k=5 (dataset médio)"
else:
    estrategia = "holdout 80/20 (dataset grande)"

print(f"Dataset: {n_amostras} amostras, {n_classes} classes")
print(f"Estratégia recomendada: {estrategia}")

# Verificar desbalanceamento
menor_classe = min(balanceamento)
if menor_classe < 0.15:
    print(f"\n[ALERTA] Classe minoritária com {menor_classe:.0%}, considerar técnicas de balanceamento")

# --- Laços: iteração sobre modelos candidatos ---
modelos_candidatos = {
    "Logistic Regression": {"tipo": "linear", "interpretavel": True},
    "Random Forest": {"tipo": "ensemble", "interpretavel": False},
    "SVM": {"tipo": "kernel", "interpretavel": False},
    "KNN": {"tipo": "inst\u00e2ncia", "interpretavel": True}
}

print("\nModelos candidatos:")
for nome, props in modelos_candidatos.items():
    status = "Sim" if props["interpretavel"] else "Nao"
    print(f"  {nome:.<25} tipo: {props['tipo']:<10} interpretavel: {status}")

# --- Enumerate e zip: padrões comuns ---
epocas = [1, 5, 10, 20, 50]
losses = [2.31, 1.45, 0.87, 0.42, 0.18]

print("\nEvolução do treinamento:")
for i, (epoca, loss) in enumerate(zip(epocas, losses)):
    barra = "#" * int((1 - loss/2.5) * 20)
    print(f"  Epoca {epoca:>3}: loss = {loss:.2f} {barra}")

# --- While: condição de parada (similar a early stopping) ---
loss_atual = 2.0
paciencia = 3
sem_melhora = 0
epoca = 0

print("\nSimulação de early stopping:")
import random
random.seed(42)

while sem_melhora < paciencia and epoca < 20:
    epoca += 1
    novo_loss = loss_atual * random.uniform(0.85, 1.05)
    if novo_loss < loss_atual:
        loss_atual = novo_loss
        sem_melhora = 0
        print(f"  Epoca {epoca:>2}: loss = {loss_atual:.4f} (melhora)")
    else:
        sem_melhora += 1
        print(f"  Epoca {epoca:>2}: loss = {novo_loss:.4f}    (sem melhora {sem_melhora}/{paciencia})")

print(f"  Treinamento encerrado na epoca {epoca} com loss = {loss_atual:.4f}")
