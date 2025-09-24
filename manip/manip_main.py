# --- Importación de Librerías ---
import pandas as pd                # Manejo de datos en DataFrames
import numpy as np                 # Cálculos numéricos y NaN
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt    # Visualización de gráficos
import seaborn as sns              # Visualización estadística (más estética)

# --- Cargar Dataset desde CSV ---
df = pd.read_csv("manip/data/dataset.csv")  # <-- Aquí se lee el CSV con los datos "malos" (NaN)

print(" 1. Dataset Original (desde CSV):")
print(df)
print("-" * 40)

# --- 1. Imputación de Valores Faltantes (con la media) ---
df['Edad'] = df['Edad'].fillna(df['Edad'].mean())
df['Ingresos'] = df['Ingresos'].fillna(df['Ingresos'].mean())

print("\n 2. Después de Imputación de Valores Faltantes:")
print(df)
print("-" * 40)

# --- 2. Codificación de Variables Categóricas ---
# A) One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=['Genero'], drop_first=True)
print("\n 3.A. Después de Codificación One-Hot:")
print(df_one_hot)
print("-" * 40)

# B) Label Encoding
le = LabelEncoder()
df['Genero_LabelEncoded'] = le.fit_transform(df['Genero'])
print("\n 3.B. Demostración de Label Encoding:")
print(df[['Nombre', 'Genero', 'Genero_LabelEncoded']])
print("-" * 40)

# --- 3. Discretización de Ingresos en Categorías ---
df['SegmentoIngresos'] = pd.cut(
    df['Ingresos'],
    bins=[0, 800, 1500, 3000],
    labels=['Bajo', 'Medio', 'Alto']
)

nuevos_bins = [0, 1000, 1800, 3000]
nuevas_labels = ['Básico', 'Avanzado', 'Premium']
df['NuevoSegmentoIngresos'] = pd.cut(
    df['Ingresos'],
    bins=nuevos_bins,
    labels=nuevas_labels
)

print("\n 4. Después de Discretización (Original y Modificada):")
print(df[['Ingresos', 'SegmentoIngresos', 'NuevoSegmentoIngresos']])
print("-" * 40)

# --- 4. Normalización y Estandarización ---
scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()

df[['Edad_norm', 'Ingresos_norm']] = scaler_minmax.fit_transform(df[['Edad', 'Ingresos']])
df[['Edad_std', 'Ingresos_std']] = scaler_std.fit_transform(df[['Edad', 'Ingresos']])

print("\n 5. Después de Normalización y Estandarización:")
print(df[['Edad', 'Ingresos', 'Edad_norm', 'Ingresos_norm', 'Edad_std', 'Ingresos_std']].round(2))
print("-" * 40)

# --- 5. Visualización Comparativa (DISPERSIÓN) ---
print("\n 6. Generando Gráfico Comparativo de Escalado (Dispersión)...")
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparación de Técnicas de Escalado para la Variable "Ingresos"', fontsize=16)

x = df.index

sns.scatterplot(x=x, y=df['Ingresos'], ax=axes[0], color='blue', s=100)
axes[0].set_title('Dispersión Original')
axes[0].set_ylabel('Ingresos')
axes[0].set_xlabel('Índice')

sns.scatterplot(x=x, y=df['Ingresos_norm'], ax=axes[1], color='green', s=100)
axes[1].set_title('Dispersión Normalizada (Min-Max)')
axes[1].set_ylabel('Ingresos Normalizados')
axes[1].set_xlabel('Índice')

sns.scatterplot(x=x, y=df['Ingresos_std'], ax=axes[2], color='red', s=100)
axes[2].set_title('Dispersión Estandarizada (Z-score)')
axes[2].set_ylabel('Ingresos Estandarizados')
axes[2].set_xlabel('Índice')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n Proceso completado.")