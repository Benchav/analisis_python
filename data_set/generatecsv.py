# generatecsv.py
import os
import pandas as pd
import numpy as np

def main():
    np.random.seed(42)

    data = {
        "edad": np.random.randint(18, 65, 100),
        "ingresos": np.random.randint(300, 5000, 100),
        "gasto_salud": np.random.randint(100, 2000, 100),
        "horas_trabajo": np.random.randint(20, 60, 100),
        "horas_estudio": np.random.randint(0, 30, 100),
        "actividad_fisica": np.random.randint(0, 10, 100),
        "consumo_frutas": np.random.randint(0, 7, 100),
        "consumo_alcohol": np.random.randint(0, 5, 100),
        "diagnostico": np.random.randint(0, 2, 100),
    }

    df = pd.DataFrame(data)

    # carpeta "data" dentro de la carpeta donde está este script
    base_dir = os.path.dirname(os.path.abspath(__file__))  # esto apunta a ANALISIS_PYTHON/data_set
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)  # crea la carpeta si no existe

    file_path = os.path.join(data_dir, "dataset_reduccion.csv")
    df.to_csv(file_path, index=False)
    print(f"CSV generado en: {file_path}")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()


#python data_set\generatecsv.py