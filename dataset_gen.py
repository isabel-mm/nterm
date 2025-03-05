import os
import json

# Rutas de los archivos
corpus_dir = "processed_corpus"  # Directorio con los textos procesados
terminos_path = "terminos_extraidos.txt"  # Archivo con la lista de términos
output_dir = "spacy_annotated_dataset"  # Directorio para guardar el dataset anotado

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    print(f"Creando directorio de salida: {output_dir}")
    os.makedirs(output_dir)
    print("Directorio creado correctamente.\n")

# Leer la lista de términos
print("Leyendo la lista de términos...")
with open(terminos_path, "r", encoding="utf-8") as f:
    terminos = [line.strip().lower() for line in f if line.strip()]  # Normalizar términos a minúsculas
print(f"Se han cargado {len(terminos)} términos.\n")

def anotar_terminos(texto, terminos, etiqueta="TERMINO"):
    texto_normalizado = texto.lower()  # Normalizar el texto a minúsculas
    entidades = []
    
    for term in terminos:
        start = 0
        while True:
            start = texto_normalizado.find(term, start)  # Buscar el término en el texto normalizado
            if start == -1:
                break
            end = start + len(term)

            # Añadir la entidad encontrada
            entidades.append((start, end, etiqueta))
            start += len(term)

    # Filtrar entidades solapadas
    entidades = sorted(entidades, key=lambda x: (x[0], x[1]))
    entidades_filtradas = []
    prev_end = -1

    for ent in entidades:
        if ent[0] >= prev_end:  # Evitar solapamientos
            entidades_filtradas.append(ent)
            prev_end = ent[1]

    return entidades_filtradas

# Procesar cada archivo en el corpus
print("Iniciando la anotación del corpus...")
archivos = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")]
total_archivos = len(archivos)
print(f"Se encontraron {total_archivos} archivos para procesar.\n")

dataset_anotado = []

for i, filename in enumerate(archivos, 1):
    filepath = os.path.join(corpus_dir, filename)
    print(f"Procesando archivo {i} de {total_archivos}: {filename}")

    with open(filepath, "r", encoding="utf-8") as f:
        texto = f.read()

    # Anotar términos en el texto
    entidades = anotar_terminos(texto, terminos)

    # Añadir al dataset anotado
    dataset_anotado.append({
        "text": texto,
        "entities": entidades
    })

# Guardar el dataset anotado en un archivo JSON
output_path = os.path.join(output_dir, "annotated_dataset.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset_anotado, f, ensure_ascii=False, indent=4)

print(f"Dataset anotado guardado en {output_path}\n")
