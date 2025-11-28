import cv2
import numpy as np
import tensorflow as tf
import os

from config.config import PATHS, CLASSES

IMG_SIZE = (128, 128)


def load_and_preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model_path, img_path):
    print("Cargando modelo...")
    model = tf.keras.models.load_model(model_path)

    print("Procesando imagen...")
    img = load_and_preprocess_image(img_path)

    preds = model.predict(img)
    pred_class_index = np.argmax(preds)
    pred_class_name = CLASSES[pred_class_index]

    print("===================================")
    print(f"Imagen: {img_path}")
    print(f"Predicción: {pred_class_name} ({pred_class_index})")
    print(f"Probabilidades: {preds}")
    print("===================================")

    return pred_class_name


if __name__ == "__main__":
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "train",
            "models",
            "cnn_signs_20251128_160826.keras"
        )

        model_path = os.path.abspath(model_path)
        print("Usando modelo:", model_path)

        img = (r"C:\Users\USUARIO\Desktop\U\2025-2\ColombianSignLanguageTranslator\ColombianSignLanguageTranslator"
               r"\python\data\test_images\b_001.jpg")
        print("Imagen usada:", img)
        predict_image(model_path, img)
