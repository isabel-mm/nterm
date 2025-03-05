import os
import spacy
from spacy.tokens import Span

# Cargar el modelo transformer de spaCy
print("Cargando el modelo 'en_core_web_trf'...")
nlp = spacy.load("en_core_web_trf")
print("Modelo cargado correctamente.\n")

# Rutas de los archivos
corpus_dir = "processed_corpus"  # Directorio con los textos procesados
terminos_path = "terminos_extraidos.txt"  # Archivo con la lista de términos
output_dir = "annotated_sentences"  # Directorio para guardar las oraciones anotadas

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

# Función para anotar términos en un texto
def anotar_terminos(texto, terminos, etiqueta="TERMINO"):
    texto_normalizado = texto.lower()  # Normalizar el texto a minúsculas
    doc = nlp(texto)
    nuevas_entidades = []
    for term in terminos:
        start = 0
        while True:
            start = texto_normalizado.find(term, start)  # Buscar el término en el texto normalizado
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
print("Iniciando la segmentación y anotación del corpus...")
archivos = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")]
total_archivos = len(archivos)
print(f"Se encontraron {total_archivos} archivos para procesar.\n")

for i, filename in enumerate(archivos, 1):
    filepath = os.path.join(corpus_dir, filename)
    print(f"Procesando archivo {i} de {total_archivos}: {filename}")

    with open(filepath, "r", encoding="utf-8") as f:
        texto = f.read()

    # Segmentar el texto en oraciones
    doc = nlp(texto)
    oraciones = list(doc.sents)

    # Anotar términos en cada oración
    oraciones_anotadas = []
    for oracion in oraciones:
        oracion_anotada = anotar_terminos(oracion.text, terminos)
        oraciones_anotadas.append(oracion_anotada)

    # Guardar las oraciones anotadas en un archivo
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for oracion in oraciones_anotadas:
            f.write(f"Oración: {oracion.text}\n")  # Conservar el texto original
            if oracion.ents:
                f.write("Entidades:\n")
                for ent in oracion.ents:
                    f.write(f"  - {ent.text} ({ent.label_})\n")
            f.write("\n")

    print(f"Archivo {filename} procesado y guardado en {output_path}\n")

print("Segmentación y anotación completadas. Las oraciones anotadas se han guardado en:", output_dir)
