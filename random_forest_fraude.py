import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def generar_datos(n=1000, random_state=42):
    np.random.seed(random_state)

    monto = np.random.uniform(1, 5000, n)
    hora = np.random.randint(0, 24, n)
    internacional = np.random.randint(0, 2, n)
    intentos = np.random.randint(1, 10, n)
    comercio_riesgoso = np.random.randint(0, 2, n)
    pais_distinto = np.random.randint(0, 2, n)

    fraude = (
        (monto > 2000).astype(int) +
        (internacional == 1).astype(int) +
        (intentos > 3).astype(int) +
        (comercio_riesgoso == 1).astype(int) +
        (pais_distinto == 1).astype(int) +
        ((hora >= 0) & (hora <= 5)).astype(int)
    ) >= 3

    df = pd.DataFrame({
        'monto': monto,
        'hora': hora,
        'internacional': internacional,
        'intentos': intentos,
        'comercio_riesgoso': comercio_riesgoso,
        'pais_distinto': pais_distinto,
        'fraude': fraude.astype(int)
    })

    return df


def main():
    df = generar_datos()

    X = df[['monto', 'hora', 'internacional', 'intentos', 'comercio_riesgoso', 'pais_distinto']]
    y = df['fraude']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    print("=== RANDOM FOREST: FRAUDE EN TARJETAS ===")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, pred))

    # Importancia numérica de variables
    importancias = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": modelo.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    print("\n=== IMPORTANCIA NUMÉRICA DE VARIABLES ===")
    print(importancias.to_string(index=False))

    # Comparación real vs predicho
    comparacion = pd.DataFrame({
        "Fraude_Real": y_test.values[:10],
        "Fraude_Predicho": pred[:10]
    })

    print("\n=== COMPARACIÓN NUMÉRICA: REAL VS PREDICHO ===")
    print(comparacion.to_string(index=False))

    # Estadísticas numéricas
    print("\n=== ESTADÍSTICAS ===")
    print(f"Total transacciones evaluadas: {len(y_test)}")
    print(f"Fraudes reales: {int(sum(y_test))}")
    print(f"Fraudes detectados por el modelo: {int(sum(pred))}")
    print(f"Transacciones normales reales: {int(len(y_test) - sum(y_test))}")
    print(f"Transacciones normales predichas: {int(len(pred) - sum(pred))}")

    # Gráfico de importancia
    plt.figure(figsize=(8, 6))
    plt.barh(importancias["Variable"], importancias["Importancia"])
    plt.title("Importancia de variables - Random Forest")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()