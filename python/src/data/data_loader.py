import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config.config import PATHS, CLASSES


def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.resize(image, (128, 128))
    return image, label

class DataLoader:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.dataset_path = PATHS["RAW_DATA"]
        self.classes = CLASSES

    def load_image(self, path):
        img = cv2.imread(path)

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype("float32") / 255.0
        return img

    def load_dataset(self):
        X = []
        y = []

        print("[INFO] Cargando dataset...")

        for label in self.classes:
            class_path = os.path.join(self.dataset_path, label)
            print(f"[INFO] Leyendo imágenes de la clase '{label}'...")

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = self.load_image(img_path)

                if img is not None:
                    X.append(img)
                    y.append(self.classes.index(label))

        X = np.array(X)
        y = np.array(y)

        print(f"[INFO] Dataset cargado: {X.shape[0]} imágenes.")
        return X, y

    def load_dataset_split(self, test_size=0.2):
        X, y = self.load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y
        )

        print("[INFO] División completada:")
        print(f" - Train: {X_train.shape[0]} imágenes")
        print(f" - Test:  {X_test.shape[0]} imágenes")

        return X_train, X_test, y_train, y_test

    def get_dataloaders(self, batch_size=32, test_size=0.2):
        """
        Carga el dataset, lo divide y devuelve tf.data.Dataset para entrenamiento y validación.
        """
        X_train, X_test, y_train, y_test = self.load_dataset_split(test_size=test_size)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_ds = (
            train_ds
            .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(len(X_train))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_ds = (
            test_ds
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds, test_ds
