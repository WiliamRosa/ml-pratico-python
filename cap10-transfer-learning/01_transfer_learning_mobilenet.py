# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 10 Transfer Learning

Arquivo: 01_transfer_learning_mobilenet.py
"""

# =============================================================================
# 10.3 Transfer Learning - Feature Extraction e Fine-Tuning
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    
    # --- Demonstrar conceito com CIFAR-10 ---
    (X_treino, y_treino), (X_teste, y_teste) = keras.datasets.cifar10.load_data()
    
    classes = ["avião", "automóvel", "pássaro", "gato", "cervo",
               "cachorro", "sapo", "cavalo", "navio", "caminhão"]
    
    # Usar subconjunto para demonstração rápida
    n_treino, n_teste = 5000, 1000
    X_treino = X_treino[:n_treino].astype("float32") / 255.0
    X_teste = X_teste[:n_teste].astype("float32") / 255.0
    y_treino = y_treino[:n_treino].flatten()
    y_teste = y_teste[:n_teste].flatten()
    
    # Redimensionar para MobileNetV2 (mínimo 32x32, ideal 96+)
    X_treino_resized = tf.image.resize(X_treino, (96, 96))
    X_teste_resized = tf.image.resize(X_teste, (96, 96))
    
    print(f"Treino: {X_treino_resized.shape}")
    print(f"Teste:  {X_teste_resized.shape}")
    
    # --- 1. CNN Simples (baseline) ---
    cnn_simples = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(96, 96, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    cnn_simples.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    hist_simples = cnn_simples.fit(
        X_treino_resized, y_treino, epochs=10, batch_size=64,
        validation_split=0.2, verbose=0
    )
    acc_simples = cnn_simples.evaluate(X_teste_resized, y_teste, verbose=0)[1]
    
    # --- 2. Transfer Learning: MobileNetV2 (Feature Extraction) ---
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3)
    )
    base_model.trainable = False  # Congelar camadas pré-treinadas
    
    modelo_transfer = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    
    modelo_transfer.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    print(f"\nCNN Simples: {cnn_simples.count_params():,} parâmetros")
    print(f"Transfer (total): {modelo_transfer.count_params():,} parâmetros")
    print(f"Transfer (treináveis): {sum(p.numpy().size for p in modelo_transfer.trainable_weights):,}")
    
    hist_transfer = modelo_transfer.fit(
        X_treino_resized, y_treino, epochs=10, batch_size=64,
        validation_split=0.2, verbose=0
    )
    acc_transfer = modelo_transfer.evaluate(X_teste_resized, y_teste, verbose=0)[1]
    
    # --- 3. Fine-Tuning: Descongelar últimas camadas ---
    base_model.trainable = True
    # Congelar todas exceto as últimas 20 camadas
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    modelo_transfer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # LR baixo!
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    hist_finetune = modelo_transfer.fit(
        X_treino_resized, y_treino, epochs=5, batch_size=64,
        validation_split=0.2, verbose=0
    )
    acc_finetune = modelo_transfer.evaluate(X_teste_resized, y_teste, verbose=0)[1]
    
    # --- Comparar resultados ---
    print(f"\n{'='*50}")
    print(f"RESULTADOS")
    print(f"{'='*50}")
    print(f"CNN Simples (do zero):    {acc_simples:.4f}")
    print(f"Transfer (Feature Ext.):  {acc_transfer:.4f}")
    print(f"Transfer (Fine-Tuned):    {acc_finetune:.4f}")
    print(f"Melhoria Transfer vs CNN: +{(acc_transfer - acc_simples):.4f}")
    print(f"Melhoria Fine-tune total: +{(acc_finetune - acc_simples):.4f}")
    
    # --- Visualizar ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Acurácia comparativa
    nomes = ["CNN\nSimples", "Transfer\n(Feature Ext.)", "Transfer\n(Fine-Tuned)"]
    accs = [acc_simples, acc_transfer, acc_finetune]
    cores = ["lightcoral", "steelblue", "seagreen"]
    barras = axes[0].bar(nomes, accs, color=cores)
    for b, a in zip(barras, accs):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{a:.2%}", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Comparação de Acurácia")
    
    # Predições do melhor modelo
    preds = modelo_transfer.predict(X_teste_resized[:16], verbose=0).argmax(axis=1)
    for i in range(16):
        ax = fig.add_subplot(2, 8, 9 + i) if i < 8 else None
    
    # Curvas de treinamento
    axes[1].plot(hist_simples.history["val_accuracy"], "--", label="CNN Simples (val)", color="lightcoral")
    axes[1].plot(hist_transfer.history["val_accuracy"], label="Transfer (val)", color="steelblue")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_title("Curvas de Validação")
    axes[1].legend()
    
    plt.suptitle("Transfer Learning - CIFAR-10 com MobileNetV2", fontweight="bold")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("[NOTA] TensorFlow não disponível. Ver notas do Capítulo 8.")
except Exception as e:
    print(f"[NOTA] Erro: {e}")
    print("Transfer Learning requer conexão para baixar pesos pré-treinados.")
