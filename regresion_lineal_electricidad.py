import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generar_datos(n=300, random_state=42):
    np.random.seed(random_state)

    habitantes = np.random.randint(1, 7, n)
    tamano_vivienda = np.random.randint(40, 250, n)
    electrodomesticos = np.random.randint(3, 20, n)
    horas_uso = np.random.uniform(2, 12, n)

    ruido = np.random.normal(0, 15, n)

    consumo = (
        habitantes * 18 +
        tamano_vivienda * 0.9 +
        electrodomesticos * 12 +
        horas_uso * 20 +
        ruido
    )

    df = pd.DataFrame({
        'habitantes': habitantes,
        'tamano_vivienda': tamano_vivienda,
        'electrodomesticos': electrodomesticos,
        'horas_uso': horas_uso,
        'consumo_kwh': consumo
    })

    return df


def main():
    df = generar_datos()

    X = df[['habitantes', 'tamano_vivienda', 'electrodomesticos', 'horas_uso']]
    y = df['consumo_kwh']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    print("=== REGRESIÓN LINEAL: CONSUMO DE ELECTRICIDAD ===")
    print("MAE:", round(mean_absolute_error(y_test, pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
    print("R2:", round(r2_score(y_test, pred), 4))

    # Intercepto
    print("\nIntercepto del modelo:")
    print(round(modelo.intercept_, 4))

    # Tabla numérica de coeficientes
    coeficientes = pd.DataFrame({
        "Variable": X.columns,
        "Coeficiente": modelo.coef_
    }).sort_values(by="Coeficiente", ascending=False)

    print("\n=== IMPORTANCIA NUMÉRICA DE VARIABLES ===")
    print(coeficientes.to_string(index=False))

    # Comparación numérica real vs predicción
    comparacion = pd.DataFrame({
        "Consumo_Real_kWh": y_test.values[:10],
        "Consumo_Predicho_kWh": pred[:10],
        "Error_Absoluto": np.abs(y_test.values[:10] - pred[:10])
    })

    print("\n=== COMPARACIÓN NUMÉRICA: REAL VS PREDICHO ===")
    print(comparacion.round(2).to_string(index=False))

    # Rango numérico general
    print("\n=== RANGOS NUMÉRICOS ===")
    print(f"Consumo real mínimo: {y_test.min():.2f} kWh")
    print(f"Consumo real máximo: {y_test.max():.2f} kWh")
    print(f"Consumo predicho mínimo: {pred.min():.2f} kWh")
    print(f"Consumo predicho máximo: {pred.max():.2f} kWh")

    # Predicción de un nuevo ejemplo manual
    ejemplo_nuevo = pd.DataFrame([{
        'habitantes': 4,
        'tamano_vivienda': 120,
        'electrodomesticos': 10,
        'horas_uso': 6.5
    }])

    pred_nueva = modelo.predict(ejemplo_nuevo)[0]

    print("\n=== EJEMPLO DE PREDICCIÓN NUEVA ===")
    print("Datos de entrada:")
    print(ejemplo_nuevo.to_string(index=False))
    print(f"Consumo estimado: {pred_nueva:.2f} kWh")

    # Gráfico de dispersión real vs predicho
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')
    plt.xlabel("Consumo real")
    plt.ylabel("Consumo predicho")
    plt.title("Regresión lineal - Consumo de electricidad")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico de importancia de variables
    plt.figure(figsize=(8, 6))
    plt.barh(coeficientes["Variable"], coeficientes["Coeficiente"])
    plt.xlabel("Coeficiente")
    plt.ylabel("Variable")
    plt.title("Importancia de variables - Regresión Lineal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()