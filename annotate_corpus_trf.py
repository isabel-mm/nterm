import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignorar advertencias de PyTorch

import os
import time
import spacy
from spacy.tokens import Doc, Span

# Iniciar el contador de tiempo
start_time = time.time()

# Cargar el modelo transformer de spaCy
print("Cargando el modelo 'en_core_web_trf'...")
nlp = spacy.load("en_core_web_trf")
print("Modelo cargado correctamente.\n")

# Rutas de los archivos
corpus_dir = "original_texts"  # Directorio con los textos del corpus
terminos_path = "terminos_extraidos.txt"  # Archivo con la lista de términos
output_dir = "annotated_corpus_trf"  # Directorio para guardar los textos anotados

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    print(f"¡A ver! Creando directorio de salida: {output_dir}")
    os.makedirs(output_dir)
    print("Directorio creado correctamente.\n")

# Leer la lista de términos
print("Leyendo la lista de términos...")
with open(terminos_path, "r", encoding="utf-8") as f:
    terminos = [line.strip() for line in f if line.strip()]
print(f"Se han cargado {len(terminos)} términos.\n")

# Función para anotar términos en un texto
def anotar_terminos(texto, terminos, etiqueta="TERMINO"):
    doc = nlp(texto)
    nuevas_entidades = []
    for term in terminos:
        start = 0
        while True:
            start = texto.lower().find(term.lower(), start)  # Buscar el término (insensible a mayúsculas)
            if start == -1:  # Si no se encuentra más ocurrencias
                break
            end = start + len(term)
            span = doc.char_span(start, end, label=etiqueta)  # Crear un Span
            if span is not None:
                # Verificar si el span se superpone con alguna entidad ya existente
                if not any(span.start < ent.end and span.end > ent.start for ent in nuevas_entidades):
                    nuevas_entidades.append(span)
            start += 1  # Continuar buscando después de esta ocurrencia

    # Añadir las nuevas entidades al documento
    try:
        doc.ents = list(doc.ents) + nuevas_entidades
    except ValueError as e:
        print(f"Error al añadir entidades: {e}")
        # Si hay un error, omitir las entidades problemáticas
        doc.ents = list(doc.ents)
    return doc

# Procesar cada archivo en el corpus
print("Iniciando el procesamiento del corpus...")
archivos = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")]
total_archivos = len(archivos)
print(f"Se encontraron {total_archivos} archivos para procesar.\n")

for i, filename in enumerate(archivos, 1):
    filepath = os.path.join(corpus_dir, filename)
    print(f"Procesando archivo {i} de {total_archivos}: {filename}")

    with open(filepath, "r", encoding="utf-8") as f:
        texto = f.read()

    # Preprocesamiento mínimo: eliminar espacios extra y saltos de línea
    texto = " ".join(texto.split())

    # Anotar términos
    try:
        doc = anotar_terminos(texto, terminos)
    except Exception as e:
        print(f"Error al procesar el archivo {filename}: {e}")
        continue

    # Guardar el texto anotado en formato spaCy
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in doc.sents:
            f.write(f"Oración: {sent.text}\n")
            if sent.ents:
                f.write("Entidades:\n")
                for ent in sent.ents:
                    f.write(f"  - {ent.text} ({ent.label_})\n")
            f.write("\n")

    print(f"Archivo {filename} procesado y guardado en {output_path}\n")

# Mostrar el tiempo total de ejecución
end_time = time.time()
total_time = end_time - start_time
print(f"Procesamiento completado en {total_time:.2f} segundos.")
print("Los textos anotados se han guardado en:", output_dir)
