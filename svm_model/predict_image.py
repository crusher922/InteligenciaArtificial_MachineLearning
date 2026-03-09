import os
import sys
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_svm.pkl")

IMG_SIZE = (32, 32)
CATEGORIAS = ["perros", "gatos"]


def procesar_imagen(ruta):
    img = Image.open(ruta).convert("L")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_vector = img_array.flatten()

    return img_vector.reshape(1, -1), img


def main():
    if len(sys.argv) < 2:
        print("Uso: python svm_model/predict_image.py svm_model/test/prueba1.jpg")
        return

    ruta_imagen = sys.argv[1]

    if not os.path.exists(ruta_imagen):
        print(f"No existe la imagen: {ruta_imagen}")
        return

    if not os.path.exists(MODEL_PATH):
        print("No existe modelo_svm.pkl. Primero ejecuta train_svm.py")
        return

    modelo = joblib.load(MODEL_PATH)

    vector, img = procesar_imagen(ruta_imagen)
    pred = int(modelo.predict(vector)[0])

    clase = CATEGORIAS[pred]

    print("\n=== PREDICCIÓN ===")
    print("Imagen:", ruta_imagen)
    print("Predicción:", clase)

    plt.imshow(img, cmap="gray")
    plt.title(f"Predicción: {clase}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()