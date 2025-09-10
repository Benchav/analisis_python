import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Ruta al CSV (siempre dentro de data_set/data/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # apunta a carpeta data_set
file_path = os.path.join(BASE_DIR, "data", "dataset_reduccion.csv")

# Leer dataset
df = pd.read_csv(file_path)

# Separar variables predictoras y objetivo
X = df.drop("diagnostico", axis=1)
y = df["diagnostico"]

# Selección de características con Chi²
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X, y)

print("Características seleccionadas:")
print(X.columns[selector.get_support()])

# Escalar datos para PCA
X_scaled = StandardScaler().fit_transform(X)

# PCA con 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nVarianza explicada por los componentes:", pca.explained_variance_ratio_)
print("\nNuevas características (primeras 2 filas):")
print(X_pca[:2])

# Visualización
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Visualización de datos reducidos con PCA")
plt.show()


#python data_set\dataset_main.py