import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


# --------------------------- Preprocesamiento -----------------------------

def preprocess_data(input_file='data/ventas_ejemplo.csv', output_file='data/market_preprocesados.csv'):
    """Lee CSV de ventas, filtra y guarda preprocesado en output_file."""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df = pd.read_csv(input_file)

    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df.to_csv(output_file, index=False)
    print(f"✅ Preprocesamiento completado. Archivo guardado en: {output_file}")


# ------------------------ Matriz transaccional ----------------------------

def create_transaction_matrix(preprocessed_path='data/market_preprocesados.csv'):
    """Crea matriz booleana de transacciones: filas=InvoiceNo, columnas=Description."""
    df = pd.read_csv(preprocessed_path)
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket_bool = basket > 0
    print(f"✅ Matriz transaccional creada. Tamaño: {basket_bool.shape}")
    return basket_bool


# ------------------------------ Apriori ---------------------------------

def run_apriori(basket_bool, min_support=0.002, min_conf=0.1, min_lift=1.0):
    """Ejecuta apriori y devuelve reglas filtradas por lift.

    Parámetros:
        basket_bool (DataFrame): matriz booleana de transacciones.
    """
    frequent_itemsets = apriori(basket_bool, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_conf)
    rules = rules[rules['lift'] > min_lift]
    print(f"✅ Reglas generadas. Reglas finales: {len(rules)}")
    return rules


# ---------------------------- Visualizaciones ---------------------------

def fs_to_str(fs):
    return ', '.join(list(fs))


def plot_top_rules(top_rules):
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_rules)), top_rules['support'])
    plt.yticks(range(len(top_rules)), [f"{fs_to_str(a)} → {fs_to_str(c)}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])])
    plt.gca().invert_yaxis()
    plt.xlabel('Support')
    plt.title('Top Reglas de Asociación por Support')
    plt.tight_layout()
    plt.show()


def plot_scatter(top_rules):
    plt.figure(figsize=(10, 6))
    plt.scatter(top_rules['confidence'], top_rules['lift'], s=top_rules['support'] * 2000, alpha=0.6)
    for i, row in top_rules.iterrows():
        plt.text(row['confidence'], row['lift'], f"{fs_to_str(row['antecedents'])} → {fs_to_str(row['consequents'])}", fontsize=8, ha='right')
    plt.xlabel('Confidence')
    plt.ylabel('Lift')
    plt.title('Confidence vs Lift (Tamaño = Support)')
    plt.tight_layout()
    plt.show()


# ---------------------------------- Main ---------------------------------

def main(args):
    preprocessed_path = args.output

    preprocess_data(input_file=args.input, output_file=preprocessed_path)
    basket_bool = create_transaction_matrix(preprocessed_path=preprocessed_path)

    rules = run_apriori(basket_bool, min_support=args.min_support, min_conf=args.min_conf, min_lift=args.min_lift)
    top_rules = rules.sort_values(by='support', ascending=False).head(args.top_n)

    if not top_rules.empty:
        plot_top_rules(top_rules)
        plot_scatter(top_rules)
        print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        print('⚠️ No se generaron reglas con los parámetros actuales. Intenta reducir min_support/min_conf/min_lift.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline Market Basket Analysis')
    parser.add_argument('--input', default='data/ventas_ejemplo.csv', help='ruta al CSV de ventas (por defecto: data/ventas_ejemplo.csv)')
    parser.add_argument('--output', default='data/market_preprocesados.csv', help='ruta del CSV preprocesado a generar')
    parser.add_argument('--min_support', type=float, default=0.002, help='min_support para apriori')
    parser.add_argument('--min_conf', type=float, default=0.1, help='min confidence para association_rules')
    parser.add_argument('--min_lift', type=float, default=1.0, help='filtro mínimo de lift para las reglas')
    parser.add_argument('--top_n', type=int, default=10, help='cantidad de top reglas a visualizar')
    args = parser.parse_args()

    try:
        main(args)
        print('\n✅ Pipeline Market Basket completado correctamente.')
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}. Asegúrate de colocar tus CSV en la ruta correcta o ajustar los parámetros del script.")
    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")