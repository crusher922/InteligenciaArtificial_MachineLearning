import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def generar_datos(n=400, random_state=42):
    np.random.seed(random_state)

    ingreso = np.random.randint(300, 4000, n)
    edad = np.random.randint(18, 70, n)
    historial = np.random.randint(0, 2, n)  # 0 = malo, 1 = bueno
    deuda = np.random.randint(0, 3000, n)
    anios_empleo = np.random.randint(0, 30, n)

    aprobacion = (
        (ingreso > 1200).astype(int) +
        (historial == 1).astype(int) +
        (deuda < 1000).astype(int) +
        (anios_empleo > 2).astype(int)
    ) >= 3

    df = pd.DataFrame({
        'ingreso_mensual': ingreso,
        'edad': edad,
        'historial_bueno': historial,
        'deuda_actual': deuda,
        'anios_empleo': anios_empleo,
        'aprobado': aprobacion.astype(int)
    })

    return df


def main():
    df = generar_datos()

    X = df[['ingreso_mensual', 'edad', 'historial_bueno', 'deuda_actual', 'anios_empleo']]
    y = df['aprobado']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    print("=== ÁRBOL DE DECISIÓN: APROBACIÓN DE CRÉDITO ===")
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
        "Valor_Real": y_test.values[:10],
        "Prediccion_Modelo": pred[:10]
    })

    print("\n=== COMPARACIÓN NUMÉRICA: REAL VS PREDICHO ===")
    print(comparacion.to_string(index=False))

    # Estadísticas numéricas
    print("\n=== ESTADÍSTICAS ===")
    print(f"Total solicitudes evaluadas: {len(y_test)}")
    print(f"Créditos aprobados reales: {int(sum(y_test))}")
    print(f"Créditos aprobados predichos: {int(sum(pred))}")
    print(f"Créditos rechazados reales: {int(len(y_test) - sum(y_test))}")
    print(f"Créditos rechazados predichos: {int(len(pred) - sum(pred))}")

    # Gráfico del árbol
    plt.figure(figsize=(16, 8))
    plot_tree(
        modelo,
        feature_names=X.columns,
        class_names=['No aprobado', 'Aprobado'],
        filled=True,
        rounded=True
    )
    plt.title("Árbol de decisión - Aprobación de crédito")
    plt.tight_layout()
    plt.show()

    # Gráfico de importancia
    plt.figure(figsize=(8, 6))
    plt.barh(importancias["Variable"], importancias["Importancia"])
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.title("Importancia de variables - Árbol de decisión")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()