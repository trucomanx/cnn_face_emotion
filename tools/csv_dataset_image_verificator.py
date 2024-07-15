#!/usr/bin/python3

import pandas as pd
import os
from PIL import Image

# Ruta al archivo CSV
csv_file_path = 'train.csv'
image_format = 'PNG'

import argparse

parser = argparse.ArgumentParser(   prog='ProgramName',
                                    description='Lee um aquivo train.csv e verifica se cada file listado existe e verifica se o formato e` PNG.',
                                    epilog='Text at the bottom of help',
                                    add_help=False)

# Definindo os argumentos
parser.add_argument('--input'  , type=str, default=csv_file_path, help='Caminho do arquivo *.cvs de entrada')
parser.add_argument('--format' , type=str, default=image_format , help='Formato dos arquivos de imagem de entrada, retorna True se o formato e assim.')
parser.add_argument('--verbose', action='store_true', help='Exibir informações detalhadas.')
parser.add_argument('--help', action='store_true',  help='Show this help message and exit.')

# Parse dos argumentos
args = parser.parse_args()

# Usando os argumentos
if args.verbose:
    print(f"Arquivo de entrada: {args.input}")
    print(f"Formato de imagem: {args.format}")
    print('')
    
if args.help:
    parser.print_help();
    exit();
################################################################################

if os.path.exists(args.input)==False:
    print('No exist',args.input)
    exit();
    
    
# Leer el archivo CSV usando pandas
df = pd.read_csv(csv_file_path)

Nel = df.shape[0]

# Función para verificar si el archivo es un PNG válido
def is_valid_png(file_path):
    if os.path.exists(file_path):
        try:
            with Image.open(file_path) as img:
                return img.format == args.format
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
