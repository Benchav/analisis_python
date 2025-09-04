import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------- Datos de ejemplo -------------------------------
DEFAULT_TIEMPOS = [20, 22, 23, 25, 25, 26, 27, 28, 30, 40, 50]
DEFAULT_PUBLICIDAD = [2, 4, 6, 8, 10]
DEFAULT_VENTAS = [15, 25, 35, 50, 65]
DEFAULT_EDADES = [18, 20, 22, 23, 25, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 42, 45]


# ------------------------- Funciones gráficas -----------------------------

def plot_delivery_boxplot(tiempos, show=True, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    plt.boxplot(tiempos, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='orange', markersize=8, linestyle='none'))
    plt.title('Boxplot: Tiempos de Entrega')
    plt.ylabel('Minutos')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Guardado boxplot en: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_ad_vs_sales(publicidad, ventas, show=True, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    plt.scatter(publicidad, ventas, s=100)
    plt.title('Diagrama de dispersión: Publicidad vs Ventas')
    plt.xlabel('Gastos en Publicidad (miles)')
    plt.ylabel('Ventas (miles)')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Guardado scatter en: {save_path}")
    if show:
        plt.show()
    plt.close()


def compute_stats_and_hist(edades, bin_width=5, show=True, save_path: Optional[str] = None):
    edades = np.array(edades)
    media = np.mean(edades)
    mediana = np.median(edades)
    moda = pd.Series(edades).mode().iloc[0]
    varianza = np.var(edades, ddof=1)
    desviacion = np.std(edades, ddof=1)

    print('--- Estadísticos ---')
    print(f'Media: {media}')
    print(f'Mediana: {mediana}')
    print(f'Moda: {moda}')
    print(f'Varianza (muestra): {varianza}')
    print(f'Desviación estándar (muestra): {desviacion}')

    # construir intervalos
    ancho = bin_width
    min_edad = int(edades.min())
    max_edad = int(edades.max())
    inicio = (min_edad // ancho) * ancho
    fin = ((max_edad // ancho) + 1) * ancho
    bordes = np.arange(inicio, fin + 1, ancho)

    cats = pd.cut(edades, bins=bordes, right=False, include_lowest=True)
    frecuencias = cats.value_counts().sort_index()

    etiquetas = [f"{bordes[i]}–{bordes[i+1]-1}" for i in range(len(bordes)-1)]
    tabla = pd.DataFrame({
        'Intervalo (años)': etiquetas,
        'Frecuencia': frecuencias.values,
    })
    tabla['% relativo'] = (tabla['Frecuencia'] / len(edades) * 100).round(2)
    tabla['Frec acumulada'] = tabla['Frecuencia'].cumsum()

    print('\nTabla de frecuencias:')
    print(tabla.to_string(index=False))

    # histograma
    plt.figure(figsize=(8, 5))
    sns.histplot(edades, bins=bordes, kde=False, edgecolor='black')
    plt.title('Histograma de edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.xticks(bordes)

    for i, freq in enumerate(frecuencias.values):
        left = bordes[i]
        right = bordes[i+1]
        x = (left + right) / 2
        plt.text(x, freq + 0.1, str(int(freq)), ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Guardado histograma en: {save_path}")
    if show:
        plt.show()
    plt.close()


# ------------------------- Utilidades de lectura --------------------------

def read_optional_csv(input_path: Optional[str]):
    if not input_path:
        return None
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    return pd.read_csv(input_path)


def extract_columns_from_df(df: pd.DataFrame):
    """Intenta mapear columnas del CSV a las variables que usamos.
    Busca nombres comunes en español o inglés.
    """
    col_map = {}
    cols = [c.lower() for c in df.columns]

    # tiempos entrega
    for candidate in ['deliverytime', 'tiempodeentrega', 'tiempos_entrega', 'tiempo_entrega', 'delivery_time', 'delivery']:
        if candidate in cols:
            col_map['tiempos'] = df[df.columns[cols.index(candidate)]].dropna().astype(float).tolist()
            break

    # publicidad / ventas
    for p in ['publicidad', 'ad', 'ads', 'advertising']:
        if p in cols:
            col_map['publicidad'] = df[df.columns[cols.index(p)]].dropna().astype(float).tolist()
            break
    for v in ['ventas', 'sales']:
        if v in cols:
            col_map['ventas'] = df[df.columns[cols.index(v)]].dropna().astype(float).tolist()
            break

    # edades
    for e in ['edad', 'age', 'edades']:
        if e in cols:
            col_map['edades'] = df[df.columns[cols.index(e)]].dropna().astype(int).tolist()
            break

    return col_map


# ---------------------------------- Main ---------------------------------

def main():
    parser = argparse.ArgumentParser(description='Process Data: visualizaciones y estadísticas')
    parser.add_argument('--input', default=None, help='ruta opcional a CSV con columnas compatibles')
    parser.add_argument('--mode', choices=['boxplot', 'scatter', 'hist', 'all'], default='all', help='qué gráficas ejecutar')
    parser.add_argument('--save', action='store_true', help='si se activa guarda las imágenes en la carpeta outputs/')
    args = parser.parse_args()

    df = None
    if args.input:
        df = read_optional_csv(args.input)

    mapped = extract_columns_from_df(df) if df is not None else {}

    # preparar datos (usar CSV si tiene, sino datos por defecto)
    tiempos = mapped.get('tiempos', DEFAULT_TIEMPOS)
    publicidad = mapped.get('publicidad', DEFAULT_PUBLICIDAD)
    ventas = mapped.get('ventas', DEFAULT_VENTAS)
    edades = mapped.get('edades', DEFAULT_EDADES)

    out_dir = 'outputs'
    if args.save:
        os.makedirs(out_dir, exist_ok=True)

    if args.mode in ('boxplot', 'all'):
        save_path = os.path.join(out_dir, 'boxplot_delivery.png') if args.save else None
        plot_delivery_boxplot(tiempos, save_path=save_path)

    if args.mode in ('scatter', 'all'):
        save_path = os.path.join(out_dir, 'scatter_ad_sales.png') if args.save else None
        plot_ad_vs_sales(publicidad, ventas, save_path=save_path)

    if args.mode in ('hist', 'all'):
        save_path = os.path.join(out_dir, 'hist_edades.png') if args.save else None
        compute_stats_and_hist(edades, save_path=save_path)


if __name__ == '__main__':
    try:
        main()
        print('\n✅ Process Data completado correctamente.')
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}. Asegúrate de colocar tu CSV en la ruta correcta o ajustar --input.")
    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")