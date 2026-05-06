# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 08 Deep Learning

Arquivo: 02_cnn_mnist.py
"""

# =============================================================================
# 8.4 CNN - Classificação de Dígitos Manuscritos (MNIST)
# =============================================================================
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Carregar MNIST ---
    (X_treino, y_treino), (X_teste, y_teste) = keras.datasets.mnist.load_data()
    
    # Normalizar pixels para [0, 1] e adicionar dimensão de canal
    X_treino = X_treino.astype("float32") / 255.0
    X_teste = X_teste.astype("float32") / 255.0
    X_treino = X_treino[..., np.newaxis]  # (60000, 28, 28, 1)
    X_teste = X_teste[..., np.newaxis]
    
    print(f"Treino: {X_treino.shape}")
    print(f"Teste:  {X_teste.shape}")
    print(f"Classes: 0-9 (dígitos)")
    
    # --- Construir CNN ---
    cnn = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    
    cnn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"\nParâmetros: {cnn.count_params():,}")
    
    # --- Treinar ---
    hist = cnn.fit(
        X_treino, y_treino,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    
    # --- Avaliar ---
    loss, acc = cnn.evaluate(X_teste, y_teste, verbose=0)
    print(f"\nAcurácia no teste: {acc:.4f}")
    
    # --- Visualizar predições ---
    y_pred = cnn.predict(X_teste[:20], verbose=0).argmax(axis=1)
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(20):
        ax = axes[i // 10, i % 10]
        ax.imshow(X_teste[i].squeeze(), cmap="gray")
        cor = "green" if y_pred[i] == y_teste[i] else "red"
        ax.set_title(f"{y_pred[i]}", fontsize=10, color=cor)
        ax.axis("off")
    
    plt.suptitle(f"Predições da CNN (verde=correto, vermelho=erro) - Acc: {acc:.2%}", fontweight="bold")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("[NOTA] TensorFlow não disponível. Ver nota anterior.")
except Exception as e:
    print(f"[NOTA] Erro ao baixar MNIST: {e}")
    print("Em ambientes restritos, use datasets locais ou Keras datasets.")
