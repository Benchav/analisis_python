# --- Importación de Librerías ---
import pandas as pd                # Manejo de datos en DataFrames
import numpy as np                 # Cálculos numéricos y NaN
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt    # Visualización de gráficos
import seaborn as sns              # Visualización estadística (más estética)

# --- Creación del Dataset Inicial ---
# Se construye un DataFrame con algunos valores nulos (NaN)
data = {
    'Nombre': ['Ana', 'Luis', 'Carlos', 'María', 'Pedro', 'Sofía'],
    'Edad': [25, np.nan, 35, 29, 40, 31],                # Edad contiene un NaN
    'Genero': ['Mujer', 'Hombre', 'Hombre', 'Mujer', 'Hombre', 'Mujer'],
    'Ingresos': [500, 1200, np.nan, 700, 1500, 2000]     # Ingresos contiene un NaN
}
df = pd.DataFrame(data)
print(" 1. Dataset Original:")
print(df)
print("-" * 40)

# --- 1. Imputación de Valores Faltantes (con la media) ---
# Se rellenan los valores faltantes (NaN) con la media de la columna
df['Edad'] = df['Edad'].fillna(df['Edad'].mean())
df['Ingresos'] = df['Ingresos'].fillna(df['Ingresos'].mean())
print("\n 2. Después de Imputación de Valores Faltantes:")
print(df)
print("-" * 40)

# --- 2. Codificación de Variables Categóricas ---

# A) One-Hot Encoding
# Convierte la variable categórica 'Genero' en variables binarias (dummies).
# drop_first=True elimina una columna para evitar multicolinealidad.
df_one_hot = pd.get_dummies(df, columns=['Genero'], drop_first=True)
print("\n 3.A. Después de Codificación One-Hot:")
print(df_one_hot)
print("-" * 40)

# B) Label Encoding
# Convierte las categorías en números (ejemplo: Hombre=0, Mujer=1).
# Usamos el dataframe original para la demostración.
le = LabelEncoder()
df['Genero_LabelEncoded'] = le.fit_transform(df['Genero'])
print("\n 3.B. Demostración de Label Encoding:")
print(df[['Nombre', 'Genero', 'Genero_LabelEncoded']])
print("-" * 40)

# --- 3. Discretización de Ingresos en Categorías ---

# A) Discretización Original
# Se crean rangos de ingresos en categorías: Bajo, Medio, Alto
df['SegmentoIngresos'] = pd.cut(
    df['Ingresos'],
    bins=[0, 800, 1500, 3000],
    labels=['Bajo', 'Medio', 'Alto']
)

# B) Discretización Modificada
# Nuevos rangos y etiquetas: Básico, Avanzado, Premium
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
# Se preparan los datos numéricos para algoritmos que son sensibles a la escala

scaler_minmax = MinMaxScaler()   # Escala los valores al rango [0, 1]
scaler_std = StandardScaler()    # Escala los valores con media=0 y desviación estándar=1

# Se aplican los dos escalados a las columnas Edad e Ingresos
df[['Edad_norm', 'Ingresos_norm']] = scaler_minmax.fit_transform(df[['Edad', 'Ingresos']])
df[['Edad_std', 'Ingresos_std']] = scaler_std.fit_transform(df[['Edad', 'Ingresos']])

print("\n 5. Después de Normalización y Estandarización:")
print(df[['Edad', 'Ingresos', 'Edad_norm', 'Ingresos_norm', 'Edad_std', 'Ingresos_std']].round(2))
print("-" * 40)

# --- 5. Visualización Comparativa (DISPERSIÓN) ---
print("\n 6. Generando Gráfico Comparativo de Escalado (Dispersión)...")

# Estilo de seaborn
sns.set_style("whitegrid")

# Se crean 3 gráficos en una misma figura (1 fila, 3 columnas)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparación de Técnicas de Escalado para la Variable \"Ingresos\"', fontsize=16)

# Usaremos el índice del DataFrame como eje X para identificar cada observación
x = df.index

# Gráfico 1: Dispersión Original
sns.scatterplot(x=x, y=df['Ingresos'], ax=axes[0], color='blue', s=100)
axes[0].set_title('Dispersión Original')
axes[0].set_ylabel('Ingresos')
axes[0].set_xlabel('Índice')
# (Opcional) Línea de tendencia lineal: descomentar la siguiente línea para agregar
# sns.regplot(x=x, y=df['Ingresos'], ax=axes[0], scatter=False, truncate=False, ci=None, line_kws={'lw':1.5, 'ls':'--'})

# Gráfico 2: Dispersión Normalizada (Min-Max)
sns.scatterplot(x=x, y=df['Ingresos_norm'], ax=axes[1], color='green', s=100)
axes[1].set_title('Dispersión Normalizada (Min-Max)')
axes[1].set_ylabel('Ingresos Normalizados')
axes[1].set_xlabel('Índice')
# (Opcional) Línea de tendencia
# sns.regplot(x=x, y=df['Ingresos_norm'], ax=axes[1], scatter=False, truncate=False, ci=None, line_kws={'lw':1.5, 'ls':'--'})

# Gráfico 3: Dispersión Estandarizada (Z-score)
sns.scatterplot(x=x, y=df['Ingresos_std'], ax=axes[2], color='red', s=100)
axes[2].set_title('Dispersión Estandarizada (Z-score)')
axes[2].set_ylabel('Ingresos Estandarizados')
axes[2].set_xlabel('Índice')
# (Opcional) Línea de tendencia
# sns.regplot(x=x, y=df['Ingresos_std'], ax=axes[2], scatter=False, truncate=False, ci=None, line_kws={'lw':1.5, 'ls':'--'})

# Ajuste de espaciado para que no se encimen títulos
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n Proceso completado.")