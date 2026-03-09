import os
import numpy as np
from PIL import Image
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_svm.pkl")

CATEGORIAS = ["perros", "gatos"]
IMG_SIZE = (32, 32)


def cargar_imagenes():
    X = []
    y = []

    for etiqueta, categoria in enumerate(CATEGORIAS):
        ruta_categoria = os.path.join(DATASET_DIR, categoria)

        if not os.path.exists(ruta_categoria):
            print(f"No existe la carpeta: {ruta_categoria}")
            continue

        for archivo in os.listdir(ruta_categoria):
            if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            ruta_imagen = os.path.join(ruta_categoria, archivo)

            try:
                img = Image.open(ruta_imagen).convert("L")
                img = img.resize(IMG_SIZE)

                img_array = np.array(img, dtype=np.float32) / 255.0
                img_vector = img_array.flatten()

                X.append(img_vector)
                y.append(etiqueta)

            except Exception as e:
                print(f"Error procesando {ruta_imagen}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    print("Cargando dataset...")
    X, y = cargar_imagenes()

    if len(X) == 0:
        print("No se cargaron imágenes. Revisa dataset/perros y dataset/gatos")
        return

    print(f"Cantidad de imágenes: {len(X)}")
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")

    valores_unicos, conteos = np.unique(y, return_counts=True)
    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    for valor, cantidad in zip(valores_unicos, conteos):
        print(f"{CATEGORIAS[valor]}: {cantidad}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nEntrenando SVM en CPU...")
    modelo = SVC(kernel="rbf", class_weight="balanced", C=10, gamma="scale")
    modelo.fit(X_train, y_train)

    train_pred = modelo.predict(X_train)
    pred = modelo.predict(X_test)

    print("\n=== MÉTRICAS ===")
    print("Accuracy entrenamiento:", round(accuracy_score(y_train, train_pred), 4))
    print("Accuracy prueba:", round(accuracy_score(y_test, pred), 4))

    print("\n=== RESULTADOS DEL MODELO SVM ===")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, pred, target_names=CATEGORIAS))

    joblib.dump(modelo, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()