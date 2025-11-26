import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from config.config import PATHS, CLASSES


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
        """Carga dataset + lo divide en train/test"""
        X, y = self.load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y
        )

        print("[INFO] División completada:")
        print(f" - Train: {X_train.shape[0]} imágenes")
        print(f" - Test:  {X_test.shape[0]} imágenes")

        return X_train, X_test, y_train, y_test
