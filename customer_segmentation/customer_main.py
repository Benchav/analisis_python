import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------- Preprocesamiento -----------------------------

def preprocessing_data(input='data/ventas_ejemplo.csv', output='customer_segmentation/data/customer_preprocessed.csv'):
    """Lee el CSV de ventas, filtra valores inválidos, calcula Total Price y guarda el preprocesado.

    Parámetros:
        input (str): ruta al CSV de ventas de entrada.
        output (str): ruta al CSV preprocesado de salida.
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df = pd.read_csv(input)

    # eliminar filas con valores nulos en columnas críticas
    df = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'])

    # mantener sólo ventas positivas
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # calcular total por línea y convertir fecha
    df['Total Price'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df.to_csv(output, index=False)
    print(f"✅ Preprocesamiento completado. Archivo guardado en: {output}")


# --------------------------------- RFM -------------------------------------

def create_rfm(preprocessed_path='customer_segmentation/data/customer_preprocessed.csv', output='customer_segmentation/data/rfm_data.csv'):
    """Genera la tabla RFM a partir del CSV preprocesado y la guarda.

    Retorna:
        rfm (DataFrame): tabla RFM con índice CustomerID.
    """
    df = pd.read_csv(preprocessed_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

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

    rfm = rfm[(rfm.index.notnull()) & (rfm['Monetary'] > 0)]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    rfm.to_csv(output)
    print(f"✅ RFM creado y guardado en: {output}")
    return rfm


# ----------------------------- Escalado RFM -------------------------------

def scale_rfm(rfm_path='customer_segmentation/data/rfm_data.csv'):
    """Carga rfm_data.csv, estandariza Recency/Frequency/Monetary y devuelve rfm y rfm_scaled.

    Retorna:
        rfm (DataFrame), rfm_scaled (np.array)
    """
    rfm = pd.read_csv(rfm_path, index_col=0)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    print("✅ Escalado completado.")
    return rfm, rfm_scaled


# ----------------------------- Elección de K -------------------------------

def find_optimal_k(rfm_scaled, max_k=10):
    """Grafica Inertia (codo) y Silhouette para k en [2, max_k].

    Muestra las gráficas usando matplotlib.
    """
    inertia = []
    silhouette_scores = []
    K = range(2, max_k + 1)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(rfm_scaled)
        inertia.append(km.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, labels))

    plt.figure()
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Método del codo')
    plt.show()

    plt.figure()
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.show()


# ------------------------------ Clustering -------------------------------

def run_kmeans(rfm, rfm_scaled, k=4, output='customer_segmentation/data/rfm_clustered.csv'):
    """Ejecuta KMeans con k clusters, añade columna 'Cluster' al DataFrame rfm y lo guarda."""
    km = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = km.fit_predict(rfm_scaled)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    rfm.to_csv(output)
    print(f"✅ KMeans ejecutado con k={k}. Resultado guardado en: {output}")
    return rfm


# ---------------------------- Análisis clusters ---------------------------

def analyze_clusters(rfm):
    cluster_profile = rfm.groupby('Cluster')[["Recency", "Frequency", "Monetary"]].mean()
    cluster_size = rfm['Cluster'].value_counts()
    print("📊 Perfil de Clusters")
    print(cluster_profile)
    print("\n📦 Tamaño de cada cluster")
    print(cluster_size)
    return cluster_profile, cluster_size


# ---------------------------- Visualizaciones ----------------------------

def plot_clusters_scatter(rfm):
    plt.figure()
    plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.title('Clusters de clientes (Recency vs Monetary)')
    plt.show()


def plot_cluster_profiles(cluster_profile):
    plt.figure()
    cluster_profile.plot(kind='bar')
    plt.title('Perfil promedio de cada cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Valores RFM')
    plt.show()


# ---------------------------------- Main ----------------------------------

if __name__ == '__main__':
    try:
        # 1) Preprocesar (lee el CSV original: ventas_ejemplo.csv por defecto)
        preprocessing_data()

        # 2) Crear RFM
        create_rfm()

        # 3) Escalar RFM
        rfm, rfm_scaled = scale_rfm()

        # 4) Explorar k óptimo (muestra gráficas)
        find_optimal_k(rfm_scaled)

        # 5) Ejecutar KMeans (por defecto k=4)
        rfm = run_kmeans(rfm, rfm_scaled, k=4)

        # 6) Analizar y mostrar resultados
        cluster_profile, cluster_size = analyze_clusters(rfm)
        plot_clusters_scatter(rfm)
        plot_cluster_profiles(cluster_profile)

        print('\n✅ Pipeline completado correctamente.')

    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}. Asegúrate de colocar tus CSV en la ruta correcta o ajustar los parámetros del script.")
    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")