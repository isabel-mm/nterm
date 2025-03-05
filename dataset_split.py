import os
import json
from sklearn.model_selection import train_test_split

# Ruta del repositorio (Colab o entorno local)
repo_path = os.getcwd()  # Se asume que el script se ejecuta desde el directorio del repo
input_file = os.path.join(repo_path, "annotated_dataset.json")  # Archivo JSON de entrada
output_dir = os.path.join(repo_path, "output")  # Carpeta de salida para los archivos divididos

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Verificar si el archivo JSON existe
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ No se encontró el archivo {input_file}. Verifica que el repo esté correctamente clonado.")

# Cargar el dataset
with open(input_file, "r", encoding="utf-8") as f:
    dataset_anotado = json.load(f)

# Dividir en entrenamiento (80%) y prueba (20%)
train_data, test_data = train_test_split(dataset_anotado, test_size=0.2, random_state=42)

# Definir rutas de salida
train_path = os.path.join(output_dir, "train_data.json")
test_path = os.path.join(output_dir, "test_data.json")

# Guardar los datasets divididos
with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"✅ Dataset de entrenamiento guardado en: {train_path}")
print(f"✅ Dataset de prueba guardado en: {test_path}")
