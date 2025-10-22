import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'ventas_tecnologia_100_plus5000.csv')


OUTPUT_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_GRAPHIC_DIR = os.path.join(SCRIPT_DIR, 'graphic')



PREPROCESSED_PATH = os.path.join(OUTPUT_DATA_DIR, 'customer_preprocessed.csv')
RFM_PATH = os.path.join(OUTPUT_DATA_DIR, 'rfm_data.csv')
CLUSTERED_PATH = os.path.join(OUTPUT_DATA_DIR, 'rfm_clustered.csv')


ELBOW_PLOT_PATH = os.path.join(OUTPUT_GRAPHIC_DIR, 'elbow_plot.png')
SILHOUETTE_PLOT_PATH = os.path.join(OUTPUT_GRAPHIC_DIR, 'silhouette_plot.png')
SCATTER_PLOT_PATH = os.path.join(OUTPUT_GRAPHIC_DIR, 'clusters_scatter.png')
PROFILE_PLOT_PATH = os.path.join(OUTPUT_GRAPHIC_DIR, 'cluster_profiles.png')



# Preprocesamiento  Crea el customer_processed.csv

# Usar las nuevas rutas por defecto
def preprocessing_data(input=DEFAULT_CSV_PATH, output=PREPROCESSED_PATH):
    """
    Lee el CSV adaptado, valida columnas mínimas necesarias, crea 'Total Price',
    convierte fechas y guarda el preprocesado.
    """
    # Asegurarse de que el directorio de salida (OUTPUT_DATA_DIR) exista
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        df = pd.read_csv(input)
    except FileNotFoundError:
        print(f"Error FATAL: No se pudo encontrar el archivo de entrada en la ruta esperada:")
        print(f"{input}")
        print("Asegúrate de que la carpeta 'data' exista en 'analisis_python' y contenga el CSV.")
        raise
    except Exception as e:
        print(f"Error al leer el CSV {input}: {e}")
        raise

    # columnas mínimas requeridas para RFM
    required = ['InvoiceNo', 'InvoiceDate', 'CustomerID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FileNotFoundError(f"El CSV de entrada no contiene las columnas requeridas: {missing}")

    # convertir InvoiceDate a datetime y eliminar filas con fecha inválida
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'InvoiceNo', 'CustomerID'])

    # Normalizar InvoiceNo y CustomerID a tipo string (maneja floats)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['CustomerID'] = df['CustomerID'].apply(lambda x: str(int(x)) if pd.notnull(x) else None).astype(str)

    # Si existen Quantity y UnitPrice, calcular Total Price
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        # Intentar convertir a numérico
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        df = df.dropna(subset=['Quantity', 'UnitPrice'])
        
        # Filtrar positivos para Quantity, pero permitir 0 para UnitPrice
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] >= 0)] 
        
        if df.empty:
            raise ValueError("No hay datos válidos (Quantity > 0 y UnitPrice >= 0) después de la limpieza.")

        df['Total Price'] = df['Quantity'] * df['UnitPrice']
        print(" Se usaron 'Quantity' y 'UnitPrice' para calcular 'Total Price'.")
    else:
        # No hay información de precio/cantidad: usar proxy
        df['Total Price'] = 1.0
        if 'Quantity' not in df.columns:
            df['Quantity'] = 1
        if 'UnitPrice' not in df.columns:
            df['UnitPrice'] = 1.0
        print(" No se encontró 'Quantity'/'UnitPrice'. Se creó 'Total Price' = 1 por línea como proxy.")

    # Guardar CSV preprocesado
    df.to_csv(output, index=False)
    print(f" Preprocesamiento completado. Archivo guardado en: {output}")
    return output


#  RFM crea rfm_data.csv

# Usar las nuevas rutas por defecto
def create_rfm(preprocessed_path=PREPROCESSED_PATH, output=RFM_PATH):
    """
    Genera la tabla RFM (Recency, Frequency, Monetary) y la guarda.
    """
    try:
        df = pd.read_csv(preprocessed_path)
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo preprocesado: {preprocessed_path}")
        raise
        
    if 'InvoiceDate' not in df.columns:
        raise KeyError("La columna 'InvoiceDate' no existe en el archivo preprocesado.")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'InvoiceNo', 'CustomerID'])
    
    if df.empty:
        raise ValueError("El dataframe preprocesado está vacío después de limpiar NaNs.")

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

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

    # Se permite Monetary >= 0 
    rfm = rfm[(rfm.index.notnull()) & (rfm['Monetary'] >= 0)]
    
    if rfm.empty:
        raise ValueError("No se pudieron generar datos RFM. Verifique los datos de entrada.")

    # Asegurarse de que el directorio de salida exista
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    rfm.to_csv(output)
    print(f" RFM creado y guardado en: {output}")
    return rfm, output 


#  Escalado RFM 

# Usar la nueva ruta por defecto
def scale_rfm(rfm_path=RFM_PATH):
    """
    Carga rfm_data.csv y estandariza Recency, Frequency, Monetary.
    Retorna rfm (DataFrame) y rfm_scaled (np.array).
    """
    try:
        rfm = pd.read_csv(rfm_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo RFM: {rfm_path}")
        raise
    
    if rfm.empty:
        raise ValueError("El archivo RFM está vacío.")

    for col in ['Recency', 'Frequency', 'Monetary']:
        if col not in rfm.columns:
            raise KeyError(f"La columna requerida '{col}' no existe en {rfm_path}")

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    print(" Escalado completado.")
    return rfm, rfm_scaled


#  Elección de K 

def find_optimal_k(rfm_scaled, max_k=10):
    """
    Grafica Inertia (codo) y Silhouette para k en [2, max_k].
    Muestra las gráficas en ventanas interactivas.
    """
    inertia = []
    silhouette_scores = []
    
    n_samples = rfm_scaled.shape[0]
    
    # Asegurar que max_k sea válido para Silhouette
    safe_max_k = min(max_k, n_samples - 1)

    if safe_max_k < 2:
        print(f"No hay suficientes clientes únicos ({n_samples}) para encontrar un K óptimo (se necesita K >= 2). Omitiendo búsqueda de K.")
        return

    K = range(2, safe_max_k + 1)

    for k in K:

        km = KMeans(n_clusters=k, random_state=42, n_init='auto') 
        labels = km.fit_predict(rfm_scaled)
        inertia.append(km.inertia_)
        try:
            score = silhouette_score(rfm_scaled, labels)
        except Exception as e:
            print(f"Error al calcular silhouette para k={k}: {e}")
            score = float('nan')
        silhouette_scores.append(score)

    # --- Gráfico 1: Codo (Estilo OO) ---
    fig1, ax1 = plt.subplots(figsize=(8, 5)) 
    ax1.plot(list(K), inertia, 'bx-')        
    ax1.set_xlabel('K')                      
    ax1.set_ylabel('Inertia')
    ax1.set_title('Método del codo')
    ax1.grid(True)
    fig1.tight_layout()
    
    fig1.savefig(ELBOW_PLOT_PATH) 
    print(f"Gráfico del codo guardado en: {ELBOW_PLOT_PATH}")
    
    plt.show() 
    plt.close(fig1)                          

    # --- Gráfico 2: Silueta (Estilo OO) ---
    fig2, ax2 = plt.subplots(figsize=(8, 5)) 
    ax2.plot(list(K), silhouette_scores, 'rx-')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Method')
    ax2.grid(True)
    fig2.tight_layout()
    
    fig2.savefig(SILHOUETTE_PLOT_PATH) 
    print(f"Gráfico de silueta guardado en: {SILHOUETTE_PLOT_PATH}")

    plt.show() 
    plt.close(fig2)                          


#  Clustering 

# Usar la nueva ruta por defecto
def run_kmeans(rfm, rfm_scaled, k=4, output=CLUSTERED_PATH):
    """
    Ejecuta KMeans con k clusters, añade columna 'Cluster' al DataFrame rfm y lo guarda.
    """
    n_samples = rfm_scaled.shape[0]
    
    # Ajustar k si es mayor que el número de muestras
    if n_samples < k:
        print(f"Advertencia: Número de muestras ({n_samples}) es menor que k={k}. Se usará k={n_samples}.")
        k = n_samples
    
    if k <= 0:
         raise ValueError(f"No se puede ejecutar k-means con k={k}.")

    # Añadir n_init='auto'
    km = KMeans(n_clusters=k, random_state=42, n_init='auto') 
    rfm['Cluster'] = km.fit_predict(rfm_scaled)
    
    # Asegurarse de que el directorio de salida (OUTPUT_DATA_DIR) exista
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    rfm.to_csv(output)
    print(f" KMeans ejecutado con k={k}. Resultado guardado en: {output}")
    return rfm, output 


# Análisis clusters 

def analyze_clusters(rfm):
    """
    Muestra perfil promedio (Recency, Frequency, Monetary) por cluster y tamaños.
    """
    if 'Cluster' not in rfm.columns:
        raise KeyError("La columna 'Cluster' no existe en el DataFrame. Ejecuta run_kmeans antes de analyze_clusters.")
    cluster_profile = rfm.groupby('Cluster')[["Recency", "Frequency", "Monetary"]].mean()
    cluster_size = rfm['Cluster'].value_counts().sort_index()
    print("\n--- Perfil de Clusters (promedio por cluster) ---")
    print(cluster_profile)
    print("\n--- Tamaño de cada cluster ---")
    print(cluster_size)
    return cluster_profile, cluster_size


# Visualizaciones 

def plot_clusters_scatter(rfm):
    """
    Scatter Recency vs Monetary coloreado por cluster.
    Muestra la gráfica en una ventana interactiva Y la guarda como PNG.
    """
    if 'Cluster' not in rfm.columns:
        raise KeyError("La columna 'Cluster' no existe en el DataFrame. Ejecuta run_kmeans antes de plot_clusters_scatter.")
        
    # --- Gráfico 3: Dispersión (Estilo OO) ---
    fig, ax = plt.subplots(figsize=(8, 6)) 
    scatter = ax.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Recency')               
    ax.set_ylabel('Monetary')
    ax.set_title('Clusters de clientes (Recency vs Monetary)')
    ax.grid(True)
    
    # Añadir barra de color
    try:
        clusters_present = sorted(rfm['Cluster'].unique())
        fig.colorbar(scatter, ax=ax, ticks=clusters_present) 
    except Exception as e:
        print(f"No se pudo agregar la barra de color: {e}")
        
    fig.tight_layout()
    
    fig.savefig(SCATTER_PLOT_PATH) # <--- CAMBIO: Guardar el PNG
    print(f"Gráfico de dispersión guardado en: {SCATTER_PLOT_PATH}")
    
    plt.show() # <--- Mostrar en ventana
    plt.close(fig)                             


def plot_cluster_profiles(cluster_profile):
    """
    Bar plot del perfil promedio por cluster.
    Muestra la gráfica en una ventana interactiva Y la guarda como PNG.
    """
    # Escalar el perfil para una visualización comparable
    scaler_profile = StandardScaler()
    try:
        profile_scaled = scaler_profile.fit_transform(cluster_profile)
        profile_scaled_df = pd.DataFrame(profile_scaled, index=cluster_profile.index, columns=cluster_profile.columns)
    except ValueError:
        print("No se pudo escalar el perfil del cluster. Graficando valores brutos.")
        profile_scaled_df = cluster_profile

    # --- Gráfico 4: Perfiles (Estilo OO) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pasamos los ejes (ax) a pandas
    profile_scaled_df.plot(kind='bar', ax=ax) 
    
    # Usamos 'ax' para configurar el gráfico (mejor práctica)
    ax.set_title('Perfil promedio (Estandarizado) de cada cluster') 
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Valores RFM (Estandarizados)')
    ax.grid(True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 
    ax.legend(title='RFM')
    
    fig.tight_layout()
    
    fig.savefig(PROFILE_PLOT_PATH) 
    print(f"Gráfico de perfiles guardado en: {PROFILE_PLOT_PATH}")

    plt.show() 
    plt.close(fig)                             


#  Main 


if __name__ == '__main__':
    print("Iniciando pipeline de segmentación...")
    try:
        # Asegurarse de que los directorios de salida existan
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_GRAPHIC_DIR, exist_ok=True)

        # 1) Preprocesar
        preprocessed_path = preprocessing_data()

        # 2) Crear RFM
        rfm, rfm_path = create_rfm(preprocessed_path=preprocessed_path)

        # 3) Escalar RFM
        rfm, rfm_scaled = scale_rfm(rfm_path=rfm_path)


        # 4) Explorar k óptimo (mostrará gráficas interactivas y guardará PNGs)
        if rfm_scaled.shape[0] > 1:
            print("\n--- Mostrando gráficos de K óptimo (cierra cada ventana para continuar) ---")
            find_optimal_k(rfm_scaled, max_k=6)
            
            # 5) Ejecutar KMeans (k=4 por defecto, pero se ajusta)
            n_samples_main = rfm_scaled.shape[0]
            k_default = 4
            k_to_use = min(k_default, n_samples_main) # k seguro
            
            if k_to_use > 0:
                # Usará el output por defecto (CLUSTERED_PATH)
                rfm, clustered_path = run_kmeans(
                    rfm, 
                    rfm_scaled, 
                    k=k_to_use
                )

                # 6) Analizar y mostrar resultados
                cluster_profile, cluster_size = analyze_clusters(rfm)
                
                # 7) Visualizar (mostrará gráficos y guardará PNGs)
                print("\n--- Mostrando gráficos de Clusters (cierra cada ventana para continuar) ---")
                plot_clusters_scatter(rfm)
                plot_cluster_profiles(cluster_profile)
            
                print('\n--- Ubicaciones de los archivos generados ---')
                print(f"Datos CSV generados en: {OUTPUT_DATA_DIR}")
                print(f"Gráficos PNG generados en: {OUTPUT_GRAPHIC_DIR}")

            else:
                print("No se generaron clusters (k=0 o k=1).")

        else:
            print("No hay suficientes datos (se necesita > 1 cliente) para realizar clustering.")


        print('\nPipeline completado.')

    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado o columna faltante: {e}.")
    except KeyError as e:
        print(f"Error: KeyError: {e} - Columna esperada no encontrada.")
    except ValueError as e:
        print(f"Error: ValueError: {e} - Datos no válidos después del filtrado.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")