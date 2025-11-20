import spacy
import nltk
import os

def download_nltk_data():
    """Descarga los sets de datos necesarios para NLTK."""
    print("Iniciando descarga de datos de NLTK...")
    try:
        nltk.data.find('tokenizers/punkt')
        print("Datos 'punkt' ya descargados.")
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=False)

    try:
        nltk.data.find('corpora/stopwords')
        print("Datos 'stopwords' ya descargados.")
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=False)
    print("Descarga de NLTK completada.")


def download_spacy_model():
    """Descarga el modelo de lenguaje de SpaCy."""
    model_name = "en_core_web_sm"
    print(f"Iniciando descarga del modelo de spaCy: {model_name}...")
    try:
        spacy.load(model_name)
        print(f"Modelo '{model_name}' ya instalado.")
    except OSError:
        spacy.cli.download(model_name)
        print(f"Modelo '{model_name}' descargado e instalado.")


if __name__ == "__main__":
    download_nltk_data()
    download_spacy_model()
    print("\nTodos los assets y modelos adicionales han sido configurados exitosamente.")

