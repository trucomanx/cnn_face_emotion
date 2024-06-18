#!/usr/bin/python3

import pandas as pd
import os
from PIL import Image

# Ruta al archivo CSV
csv_file_path = 'train.csv'

# Leer el archivo CSV usando pandas
df = pd.read_csv(csv_file_path)

Nel = df.shape[0]

# Función para verificar si el archivo es un PNG válido
def is_valid_png(file_path):
    if os.path.exists(file_path):
        try:
            with Image.open(file_path) as img:
                return img.format == 'PNG'
        except IOError:
            return False
    return False

# Verificar la existencia de los archivos
count=0;
for index, row in df.iterrows():
    relative_path = row.iloc[0]  # Primera columna con la ruta relativa
    if is_valid_png(relative_path):
        #print(f"El archivo en '{relative_path}' existe.")
        count=count+1;
    else:
        sys.system(f"El archivo en '{relative_path}' NO existe ou es una imagen invalida.")
    
if count==Nel:
    print('Todos los',count,'archivos de imagen existen y son validos');
