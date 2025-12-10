import tensorflow as tf
import tf2onnx
import os

def export_onnx():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "cnn_signs_20251210_004834.h5")
    output_path = os.path.join(base_dir, "cnn_signs_final.onnx")

    print("Cargando modelo Keras...")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    print("Modelo cargado.")
    print("Convirtiendo a ONNX...")

    spec = (tf.TensorSpec((None, 128, 128, 3), tf.float32, name="input"),)

    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

    print("Exportación correcta en: ", output_path)


if __name__ == "__main__":
    export_onnx()
