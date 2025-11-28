import os
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from data.data_loader import DataLoader
from models.model_cnn import build_cnn_model
from config.config import PATHS, CLASSES


def train_model():

    print("Cargando dataset...")
    loader = DataLoader(img_size=(128, 128))
    train_loader, test_loader = loader.get_dataloaders()

    print(f"Clases detectadas ({len(CLASSES)}): {CLASSES}")
    print("Batch size (train):", next(iter(train_loader))[0].shape[0])
    print("Batch size (test):", next(iter(test_loader))[0].shape[0])

    print("Construyendo modelo CNN...")
    model = build_cnn_model()
    model.summary()


    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(models_dir, f"cnn_signs_{timestamp}.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,    
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    print("Entrenando modelo...")
    history = model.fit(
        train_loader,
        validation_data=test_loader,
        epochs=60,
        callbacks=callbacks
    )

    print(f"Entrenamiento finalizado. Mejor modelo guardado en: {checkpoint_path}")



    plot_training(history)

    final_path = os.path.join(models_dir, "cnn_signs_final.keras")
    model.save(final_path)
    print(f"Modelo final guardado en: {final_path}")


def plot_training(history):


    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Entrenamiento")
    plt.plot(epochs_range, val_acc, label="Validación")
    plt.title("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Entrenamiento")
    plt.plot(epochs_range, val_loss, label="Validación")
    plt.title("Loss")
    plt.legend()

    # Guardar el gráfico
    os.makedirs("results", exist_ok=True)
    graph_path = os.path.join("results", "training_plot_20_epoch.png")
    plt.savefig(graph_path)

    print(f"Gráfica de entrenamiento guardada en: {graph_path}")
    plt.show()


if __name__ == "__main__":
    train_model()
