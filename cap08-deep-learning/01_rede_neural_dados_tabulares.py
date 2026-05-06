# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 08 Deep Learning

Arquivo: 01_rede_neural_dados_tabulares.py
"""

# =============================================================================
# 8.3 Primeira Rede Neural - Classificação com Dados Tabulares
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    print(f"TensorFlow versão: {tf.__version__}")
    
    # --- Dataset: Vinhos ---
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_treino_s = scaler.fit_transform(X_treino)
    X_teste_s = scaler.transform(X_teste)
    
    # --- Construir modelo ---
    modelo = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_treino_s.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax")  # 3 classes
    ])
    
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n=== Arquitetura do Modelo ===")
    modelo.summary()
    
    # --- Treinar com early stopping ---
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    
    historico = modelo.fit(
        X_treino_s, y_treino,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # --- Avaliar ---
    loss, acc = modelo.evaluate(X_teste_s, y_teste, verbose=0)
    y_pred = modelo.predict(X_teste_s, verbose=0).argmax(axis=1)
    
    print(f"\n=== Resultado ===")
    print(f"Loss: {loss:.4f}, Acurácia: {acc:.4f}")
    print(classification_report(y_teste, y_pred, target_names=wine.target_names))
    
    # --- Curvas de treinamento ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(historico.history["loss"], label="Treino")
    axes[0].plot(historico.history["val_loss"], label="Validação")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Evolução do Loss")
    axes[0].legend()
    
    axes[1].plot(historico.history["accuracy"], label="Treino")
    axes[1].plot(historico.history["val_accuracy"], label="Validação")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_title("Evolução da Acurácia")
    axes[1].legend()
    
    plt.suptitle("Treinamento da Rede Neural - Classificação de Vinhos", fontweight="bold")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("[NOTA] TensorFlow não instalado.")
    print("Para instalar: %pip install tensorflow")
    print("\nEm ambientes Databricks, TensorFlow está disponível no ML Runtime.")
