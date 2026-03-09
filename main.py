import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


def generar_dataset(n=500, random_state=42):
    np.random.seed(random_state)

    ciudades = ['Quito', 'Guayaquil', 'Cuenca', 'Ambato']
    tipos_campana = ['Digital', 'TV', 'Radio', 'Mixta']

    df = pd.DataFrame({
        'gasto_marketing': np.random.randint(1000, 20000, n),
        'visitas_web': np.random.randint(500, 50000, n),
        'descuento_promedio': np.random.uniform(0, 30, n),
        'satisfaccion_cliente': np.random.uniform(1, 10, n),
        'numero_vendedores': np.random.randint(1, 25, n),
        'ciudad': np.random.choice(ciudades, n),
        'tipo_campana': np.random.choice(tipos_campana, n)
    })

    efecto_ciudad = df['ciudad'].map({
        'Quito': 7000,
        'Guayaquil': 8500,
        'Cuenca': 5000,
        'Ambato': 3500
    })

    efecto_campana = df['tipo_campana'].map({
        'Digital': 9000,
        'TV': 11000,
        'Radio': 6000,
        'Mixta': 14000
    })

    ruido = np.random.normal(0, 5000, n)

    df['ventas_mensuales'] = (
        1.8 * df['gasto_marketing'] +
        0.6 * df['visitas_web'] +
        900 * df['satisfaccion_cliente'] +
        1200 * df['numero_vendedores'] -
        350 * df['descuento_promedio'] +
        efecto_ciudad +
        efecto_campana +
        ruido
    )

    return df


def construir_preprocesador():
    numeric_features = [
        'gasto_marketing',
        'visitas_web',
        'descuento_promedio',
        'satisfaccion_cliente',
        'numero_vendedores'
    ]

    categorical_features = ['ciudad', 'tipo_campana']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor


def evaluar_modelo(nombre, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        'Modelo': nombre,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Predicciones': pred
    }


def guardar_grafica_real_vs_pred(y_test, pred, nombre_modelo, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test[:30])), y_test[:30], marker='o', label='Valor real')
    plt.plot(range(len(pred[:30])), pred[:30], marker='s', label='Predicción')
    plt.title(f'Real vs Predicción - {nombre_modelo}')
    plt.xlabel('Muestras')
    plt.ylabel('Ventas mensuales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{nombre_modelo}_real_vs_pred.png'))
    plt.close()


def guardar_grafica_dispersion(y_test, pred, nombre_modelo, output_dir):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred, alpha=0.7)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        linestyle='--'
    )
    plt.title(f'Dispersión Real vs Predicho - {nombre_modelo}')
    plt.xlabel('Valor real')
    plt.ylabel('Valor predicho')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{nombre_modelo}_dispersion.png'))
    plt.close()


def guardar_grafica_residuos(y_test, pred, nombre_modelo, output_dir):
    residuos = y_test - pred

    plt.figure(figsize=(8, 6))
    plt.scatter(pred, residuos, alpha=0.7)
    plt.axhline(y=0, linestyle='--')
    plt.title(f'Residuos - {nombre_modelo}')
    plt.xlabel('Valor predicho')
    plt.ylabel('Error (real - predicción)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{nombre_modelo}_residuos.png'))
    plt.close()


def guardar_grafica_comparativa_metricas(resultados_df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.bar(resultados_df['Modelo'], resultados_df['R2'])
    plt.title('Comparación de R2 por modelo')
    plt.xlabel('Modelo')
    plt.ylabel('R2')
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacion_r2.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(resultados_df['Modelo'], resultados_df['RMSE'])
    plt.title('Comparación de RMSE por modelo')
    plt.xlabel('Modelo')
    plt.ylabel('RMSE')
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacion_rmse.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(resultados_df['Modelo'], resultados_df['MAE'])
    plt.title('Comparación de MAE por modelo')
    plt.xlabel('Modelo')
    plt.ylabel('MAE')
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacion_mae.png'))
    plt.close()


def main():
    output_dir = 'graficas_modelos'
    os.makedirs(output_dir, exist_ok=True)

    print("Generando dataset...")
    df = generar_dataset()

    print("\nPrimeras filas del dataset:")
    print(df.head())

    print("\nInformación general del dataset:")
    print(df.info())

    print("\nDescripción estadística:")
    print(df.describe())

    X = df.drop('ventas_mensuales', axis=1)
    y = df['ventas_mensuales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = construir_preprocesador()

    modelos = {
        'Regresion_Lineal': LinearRegression(),
        'Arbol_Decision': DecisionTreeRegressor(max_depth=8, random_state=42),
        'SVM': SVR(kernel='rbf', C=100, epsilon=0.1),
        'Random_Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective='reg:squarederror'
        )
    }

    resultados = []
    predicciones_por_modelo = {}

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', modelo)
        ])

        resultado = evaluar_modelo(
            nombre, pipeline, X_train, X_test, y_train, y_test
        )

        resultados.append({
            'Modelo': resultado['Modelo'],
            'MAE': resultado['MAE'],
            'RMSE': resultado['RMSE'],
            'R2': resultado['R2']
        })

        predicciones_por_modelo[nombre] = resultado['Predicciones']

        guardar_grafica_real_vs_pred(
            y_test.values,
            resultado['Predicciones'],
            nombre,
            output_dir
        )

        guardar_grafica_dispersion(
            y_test.values,
            resultado['Predicciones'],
            nombre,
            output_dir
        )

        guardar_grafica_residuos(
            y_test.values,
            resultado['Predicciones'],
            nombre,
            output_dir
        )

    resultados_df = pd.DataFrame(resultados).sort_values(by='R2', ascending=False)

    print("\n=== RESULTADOS COMPARATIVOS ===")
    print(resultados_df.to_string(index=False))

    guardar_grafica_comparativa_metricas(resultados_df, output_dir)

    mejor_modelo = resultados_df.iloc[0]['Modelo']
    print(f"\nMejor modelo según R2: {mejor_modelo}")

    comparacion = pd.DataFrame({
        'Valor_Real': y_test.values[:10],
        'Prediccion': predicciones_por_modelo[mejor_modelo][:10]
    })

    print("\n=== COMPARACIÓN REAL VS PREDICCIÓN ===")
    print(comparacion.to_string(index=False))

    resultados_df.to_csv(os.path.join(output_dir, 'resultados_modelos.csv'), index=False)
    df.to_csv('dataset_ventas_simulado.csv', index=False)

    print(f"\nGráficas guardadas en la carpeta: {output_dir}")
    print("Archivo CSV de resultados guardado como: graficas_modelos/resultados_modelos.csv")
    print("Dataset guardado como: dataset_ventas_simulado.csv")


if __name__ == "__main__":
    main()