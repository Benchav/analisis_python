import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------- Preprocesamiento -----------------------------

def preprocessing_data(input='data/Correo_n8.csv', output='customer_segmentation/data/customer_preprocessed.csv'):
    """
    Lee el CSV adaptado, valida columnas mínimas necesarias, crea 'Total Price' (proxy si no hay precio),
    convierte fechas y guarda el preprocesado.
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df = pd.read_csv(input)

    # columnas mínimas requeridas para RFM
    required = ['InvoiceNo', 'InvoiceDate', 'CustomerID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FileNotFoundError(f"El CSV de entrada no contiene las columnas requeridas: {missing}")

    # convertir InvoiceDate a datetime y eliminar filas con fecha inválida
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'InvoiceNo', 'CustomerID'])

    # Normalizar InvoiceNo y CustomerID a tipo string (evita problemas con floats)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['CustomerID'] = df['CustomerID'].astype(str)

    # Si existen Quantity y UnitPrice, calcular Total Price; si no, usar proxy = 1 por línea
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        # Intentar convertir a numérico y filtrar positivos
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        df = df.dropna(subset=['Quantity', 'UnitPrice'])
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        df['Total Price'] = df['Quantity'] * df['UnitPrice']
        print(" Se usaron 'Quantity' y 'UnitPrice' para calcular 'Total Price'.")
    else:
        # No hay información de precio/cantidad: usar proxy para que RFM funcione
        df['Total Price'] = 1.0
        # Mantener columnas Quantity/UnitPrice con valores por defecto para compatibilidad si se usan más adelante
        if 'Quantity' not in df.columns:
            df['Quantity'] = 1
        if 'UnitPrice' not in df.columns:
            df['UnitPrice'] = 1.0
        print(" No se encontró 'Quantity'/'UnitPrice'. Se creó 'Total Price' = 1 por línea como proxy.")

    # Guardar CSV preprocesado
    df.to_csv(output, index=False)
    print(f" Preprocesamiento completado. Archivo guardado en: {output}")
    return output


# --------------------------------- RFM -------------------------------------

def create_rfm(preprocessed_path='customer_segmentation/data/customer_preprocessed.csv', output='customer_segmentation/data/rfm_data.csv'):
    """
    Genera la tabla RFM (Recency, Frequency, Monetary) y la guarda.
    - Recency: días desde la última compra
    - Frequency: número de facturas (InvoiceNo) únicas por cliente
    - Monetary: suma de 'Total Price' (proxy si no hay precios)
    """
    df = pd.read_csv(preprocessed_path)
    if 'InvoiceDate' not in df.columns:
        raise KeyError("La columna 'InvoiceDate' no existe en el archivo preprocesado.")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'InvoiceNo', 'CustomerID'])

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Asegurar existencia de 'Total Price'
    if 'Total Price' not in df.columns:
        df['Total Price'] = 1.0

    # Agrupar por CustomerID
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Total Price': 'sum'
    })

    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Total Price': 'Monetary'
    }, inplace=True)

    # Filtrar clientes válidos (Monetary > 0)
    rfm = rfm[(rfm.index.notnull()) & (rfm['Monetary'] > 0)]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    rfm.to_csv(output)
    print(f" RFM creado y guardado en: {output}")
    return rfm


# ----------------------------- Escalado RFM -------------------------------

def scale_rfm(rfm_path='customer_segmentation/data/rfm_data.csv'):
    """
    Carga rfm_data.csv y estandariza Recency, Frequency, Monetary.
    Retorna rfm (DataFrame) y rfm_scaled (np.array).
    """
    rfm = pd.read_csv(rfm_path, index_col=0)

    # Verificar columnas necesarias
    for col in ['Recency', 'Frequency', 'Monetary']:
        if col not in rfm.columns:
            raise KeyError(f"La columna requerida '{col}' no existe en {rfm_path}")

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    print(" Escalado completado.")
    return rfm, rfm_scaled


# ----------------------------- Elección de K -------------------------------

def find_optimal_k(rfm_scaled, max_k=10):
    """
    Grafica Inertia (codo) y Silhouette para k en [2, max_k].
    Si silhouette falla para algún k, registra NaN y continúa.
    """
    inertia = []
    silhouette_scores = []
    K = range(2, max_k + 1)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(rfm_scaled)
        inertia.append(km.inertia_)
        try:
            score = silhouette_score(rfm_scaled, labels)
        except Exception:
            score = float('nan')
        silhouette_scores.append(score)

    plt.figure()
    plt.plot(list(K), inertia, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Método del codo')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(list(K), silhouette_scores, 'rx-')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.grid(True)
    plt.show()


# ------------------------------ Clustering -------------------------------

def run_kmeans(rfm, rfm_scaled, k=4, output='customer_segmentation/data/rfm_clustered.csv'):
    """
    Ejecuta KMeans con k clusters, añade columna 'Cluster' al DataFrame rfm y lo guarda.
    """
    if rfm_scaled.shape[0] < k:
        raise ValueError(f"Número de muestras ({rfm_scaled.shape[0]}) menor que k={k}. Reduce k o aumenta datos.")
    km = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = km.fit_predict(rfm_scaled)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    rfm.to_csv(output)
    print(f" KMeans ejecutado con k={k}. Resultado guardado en: {output}")
    return rfm


# ---------------------------- Análisis clusters ---------------------------

def analyze_clusters(rfm):
    """
    Muestra perfil promedio (Recency, Frequency, Monetary) por cluster y tamaños.
    """
    if 'Cluster' not in rfm.columns:
        raise KeyError("La columna 'Cluster' no existe en el DataFrame. Ejecuta run_kmeans antes de analyze_clusters.")
    cluster_profile = rfm.groupby('Cluster')[["Recency", "Frequency", "Monetary"]].mean()
    cluster_size = rfm['Cluster'].value_counts().sort_index()
    print(" Perfil de Clusters (promedio por cluster):")
    print(cluster_profile)
    print("\n Tamaño de cada cluster:")
    print(cluster_size)
    return cluster_profile, cluster_size


# ---------------------------- Visualizaciones ----------------------------

def plot_clusters_scatter(rfm):
    """
    Scatter Recency vs Monetary coloreado por cluster.
    """
    if 'Cluster' not in rfm.columns:
        raise KeyError("La columna 'Cluster' no existe en el DataFrame. Ejecuta run_kmeans antes de plot_clusters_scatter.")
    plt.figure()
    plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.title('Clusters de clientes (Recency vs Monetary)')
    plt.grid(True)
    plt.show()


def plot_cluster_profiles(cluster_profile):
    """
    Bar plot del perfil promedio por cluster.
    """
    plt.figure()
    cluster_profile.plot(kind='bar')
    plt.title('Perfil promedio de cada cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Valores RFM')
    plt.grid(True)
    plt.show()


# ---------------------------------- Main ----------------------------------

if __name__ == '__main__':
    try:
        # 1) Preprocesar (lee el CSV por defecto 'data/Correo_n8.csv')
        preprocessed_path = preprocessing_data(input='data/Correo_n8.csv',
                                               output='customer_segmentation/data/customer_preprocessed.csv')

        # 2) Crear RFM
        rfm = create_rfm(preprocessed_path=preprocessed_path,
                         output='customer_segmentation/data/rfm_data.csv')

        # 3) Escalar RFM
        rfm, rfm_scaled = scale_rfm(rfm_path='customer_segmentation/data/rfm_data.csv')

        # 4) Explorar k óptimo (muestra gráficas)
        find_optimal_k(rfm_scaled, max_k=6)

        # 5) Ejecutar KMeans (por defecto k=4)
        rfm = run_kmeans(rfm, rfm_scaled, k=4,
                         output='customer_segmentation/data/rfm_clustered.csv')

        # 6) Analizar y mostrar resultados
        cluster_profile, cluster_size = analyze_clusters(rfm)
        plot_clusters_scatter(rfm)
        plot_cluster_profiles(cluster_profile)

        print('\n Pipeline completado correctamente.')

    except FileNotFoundError as e:
        print(f" Archivo no encontrado o columna faltante: {e}. Comprueba la ruta y las columnas del CSV.")
    except KeyError as e:
        print(f" KeyError: {e}")
    except ValueError as e:
        print(f" ValueError: {e}")
    except Exception as e:
        print(f" Ocurrió un error: {e}")

        #agregar alemnos 5000 registros en la base de datos