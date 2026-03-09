import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


def generar_datos(n=800, random_state=42):
    np.random.seed(random_state)

    ingreso = np.random.randint(400, 5000, n)
    deuda = np.random.randint(0, 10000, n)
    score_crediticio = np.random.randint(300, 850, n)
    mora_previa = np.random.randint(0, 2, n)
    porcentaje_endeudamiento = np.random.uniform(0.1, 0.9, n)
    anios_trabajo = np.random.randint(0, 30, n)

    riesgo_alto = (
        (ingreso < 1200).astype(int) +
        (deuda > 5000).astype(int) +
        (score_crediticio < 550).astype(int) +
        (mora_previa == 1).astype(int) +
        (porcentaje_endeudamiento > 0.5).astype(int) +
        (anios_trabajo < 2).astype(int)
    ) >= 3

    df = pd.DataFrame({
        'ingreso': ingreso,
        'deuda': deuda,
        'score_crediticio': score_crediticio,
        'mora_previa': mora_previa,
        'porcentaje_endeudamiento': porcentaje_endeudamiento,
        'anios_trabajo': anios_trabajo,
        'riesgo_alto': riesgo_alto.astype(int)
    })

    return df


def main():
    df = generar_datos()

    X = df[['ingreso', 'deuda', 'score_crediticio', 'mora_previa',
            'porcentaje_endeudamiento', 'anios_trabajo']]
    y = df['riesgo_alto']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric='logloss'
    )

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    print("=== XGBOOST: RIESGO CREDITICIO ===")
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
        "Riesgo_Real": y_test.values[:10],
        "Riesgo_Predicho": pred[:10]
    })

    print("\n=== COMPARACIÓN NUMÉRICA: REAL VS PREDICHO ===")
    print(comparacion.to_string(index=False))

    # Estadísticas numéricas
    print("\n=== ESTADÍSTICAS ===")
    print(f"Clientes evaluados: {len(y_test)}")
    print(f"Clientes de alto riesgo reales: {int(sum(y_test))}")
    print(f"Clientes de alto riesgo predichos: {int(sum(pred))}")
    print(f"Clientes de bajo riesgo reales: {int(len(y_test) - sum(y_test))}")
    print(f"Clientes de bajo riesgo predichos: {int(len(pred) - sum(pred))}")

    # Grafico comparativo de la prediccion contra los datos reales
    labels = ["Alto Riesgo", "Bajo Riesgo"]

    valores_reales = [sum(y_test), len(y_test) - sum(y_test)]
    valores_pred = [sum(pred), len(pred) - sum(pred)]

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 6))
    plt.bar(x - 0.2, valores_reales, width=0.4, label="Real")
    plt.bar(x + 0.2, valores_pred, width=0.4, label="Predicción")
    plt.xticks(x, labels)
    plt.title("Comparación riesgo crediticio real vs predicho")
    plt.ylabel("Cantidad de clientes")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfico de importancia
    plt.figure(figsize=(8, 6))
    plt.barh(importancias["Variable"], importancias["Importancia"])
    plt.title("Importancia de variables - XGBoost")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()