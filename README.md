Proyecto de Minería de Datos

Este repositorio contiene tres pipelines independientes pero relacionados para análisis de datos y minería:

Customer Segmentation (customer_segmentation) — RFM + KMeans.

Market Basket Analysis (market_analitic) — Apriori / reglas de asociación.

Process Data (process_data) — visualizaciones descriptivas (boxplot, scatter, histograma) con datos de ejemplo o CSV.

A continuación encontrarás instrucciones detalladas de instalación, estructura de carpetas, ejecución y resolución de problemas.


Estructura recomendada del proyecto


Analisis_python/
├─ customer_segmentation/
│  ├─ customer_main.py
│  └─ data/
│     └─
│  
│──data/
│   └─ventas_ejemplo.csv
│  
│  
├─ market_analitic/
│  ├─ market_main.py
│  └─ data/
│     └─
└─ process_data/
   ├─ processdata_main.py




Dependencias / Requisitos

Paquetes principales (combinados para los tres pipelines):

1. pip install pandas scikit-learn matplotlib mlxtend 
2. pip install pandas scikit-learn matplotlib 
3. pip install numpy pandas matplotlib seaborn

Proceso de carga de entorno virtual de pyhton: 
1. python -m venv venv 
2. .\venv\Scripts\Activate.ps1


Ejecución de customer_segmentation: 

1. python customer_segmentation/customer_main.py 

Ejecución de market_analitic: 

1. python .\market_analitic\market_main.py 

Ejecución de process_data: 

1. python .\process_data\processdata_main.py --mode all
   

Ejecución de data_set: 

explicación: https://unanmanagua-my.sharepoint.com/:w:/g/personal/joshua_chavez22906906_estu_unan_edu_ni/EfKjbWbC_edKvzWvftX55kABmyK25ceRKjBwXVZO40wwkQ?e=nXQSba

Leer el csv:

1. python data_set\dataset_main.py


Generarlo:

1. python data_set\generatecsv.py
