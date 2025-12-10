import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → /python

PATHS = {
    "BASE": BASE_DIR,
    "RAW_DATA": os.path.join(BASE_DIR, "data", "raw"),  # <-- FIX AQUÍ
    "PROCESSED_DATA": os.path.join(BASE_DIR, "data", "processed"),
    "MODELS": os.path.join(BASE_DIR, "model"),
}

IMAGE = {
    "WIDTH": 128,
    "HEIGHT": 128,
    "CHANNELS": 3,
}

CLASSES = sorted(os.listdir(PATHS["RAW_DATA"]))

TRAINING = {
    "BATCH_SIZE": 32,
    "EPOCHS": 25,
    "LEARNING_RATE": 0.001,
    "VALIDATION_SPLIT": 0.2
}
