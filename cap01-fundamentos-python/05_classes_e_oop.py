# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 01 Fundamentos Python

Arquivo: 05_classes_e_oop.py
"""

# =============================================================================
# 1.6 Classes, Padrões OOP em Machine Learning
# =============================================================================

# --- Classe básica: encapsular um experimento ---
class ExperimentoML:
    """Gerencia um experimento de Machine Learning."""
    
    def __init__(self, nome: str, tipo: str = "classificacao"):
        self.nome = nome
        self.tipo = tipo
        self.metricas = {}
        self.hiperparametros = {}
        self._executado = False
    
    def configurar(self, **params):
        """Define hiperparâmetros do experimento."""
        self.hiperparametros.update(params)
        return self  # Permite encadeamento (method chaining)
    
    def executar(self, X_treino, y_treino):
        """Simula execução do treinamento."""
        n_amostras = len(X_treino)
        print(f"Treinando '{self.nome}' com {n_amostras} amostras...")
        print(f"  Hiperpar\u00e2metros: {self.hiperparametros}")
        
        # Simular métricas
        import random
        random.seed(hash(self.nome) % 100)
        self.metricas = {
            "acuracia": round(random.uniform(0.75, 0.95), 4),
            "f1_score": round(random.uniform(0.70, 0.92), 4)
        }
        self._executado = True
        print(f"  Resultado: {self.metricas}")
        return self
    
    def resumo(self) -> str:
        """Retorna resumo do experimento."""
        status = "[OK] Executado" if self._executado else "[Pendente]"
        return f"[{status}] {self.nome} ({self.tipo}), Metricas: {self.metricas}"
    
    def __repr__(self):
        return f"ExperimentoML(nome='{self.nome}', tipo='{self.tipo}')"


# --- Herança: especializar para classificação ---
class ExperimentoClassificacao(ExperimentoML):
    """Experimento especializado para classificação."""
    
    def __init__(self, nome: str, n_classes: int = 2):
        super().__init__(nome, tipo="classificacao")
        self.n_classes = n_classes
    
    def executar(self, X_treino, y_treino):
        """Sobrescreve execução com métricas específicas."""
        super().executar(X_treino, y_treino)
        # Adicionar métricas específicas de classificação
        import random
        self.metricas["auc_roc"] = round(random.uniform(0.80, 0.97), 4)
        print(f"  AUC-ROC: {self.metricas['auc_roc']}")
        return self


# Uso prático
print("=== Executando Experimentos ===")

# Dados simulados
X_treino = [[1, 2], [3, 4], [5, 6]] * 100
y_treino = [0, 1, 0] * 100

# Experimento com encadeamento
exp1 = (
    ExperimentoClassificacao("GradientBoosting_v1", n_classes=3)
    .configurar(n_estimators=150, max_depth=5, learning_rate=0.1)
    .executar(X_treino, y_treino)
)

print(f"\n{exp1.resumo()}")
print(f"Objeto: {exp1}")
