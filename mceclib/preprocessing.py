import re
import string
import contractions
import spacy
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    """
    Esta función limpia el texto, remueve html, caracteres especiales, urls, puntuaciones y espacios de mas.

    Args:
        text: texto que se va a limpiar.

    Returns:
        text: texto limpio.
    """
    if isinstance(text, float):
        text = str(text)
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text) #etiqueta html
    text = re.sub(r'[^a-zA-Z\s]', '', text) #caracteres especiales
    text = re.sub(r'http\S+|www\S+', '', text) #urls
    text = text.lower() # todo a minusculas
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return ' '.join(text.split())

def normalized_text(text):
    """
    intenta normalizar el texto, separando las contracciones del ingles.

    Args:
        text: texto que se va a normalizar.

    Returns:
        text: texto normalizado.
    """
    expanded_text = contractions.fix(text)
    return expanded_text

def tokenize_text(text):
    """
    tokeniza el texto, separando por palabras.

    Args:
        text: texto que se va a tokenizar.

    Returns:
        tokens: texto tokenizado.
    """
    return word_tokenize(text, language='english')

porter_stemmer = PorterStemmer()
def stemming_tokens(token_list):
    """
    Aplica el algoritmo Porter Stemmer a una lista de tokens.

    Args:
        token_list: texto tokenizado por palabras.

    Returns:
        stemmed_tokens: lista de tokes con stemming.
    """
    stemmed_tokens = [porter_stemmer.stem(token) for token in token_list]
    return stemmed_tokens

nlp = spacy.load("en_core_web_sm")
def lemmatization_text(text):
    """
    Procesa el texto completo con SpaCy y extrae el lema de cada token.

    Args:
        text: texto preprocesado anteriormente.

    Returns:
        lemmas: lista de tokes con lemmas.
    """
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return lemmas

STOP_WORDS_ENGLISH = set(stopwords.words('english'))
def remove_stopword(token_list):
    """
    Elimina las stop words de una lista de tokens

    Args:
        token_list: texto tokenizado por palabras.

    Returns:
        filtered_tokens: lista de tokes sin stopwords.
    """
    filtered_tokens = [word for word in token_list if word.lower() not in STOP_WORDS_ENGLISH]
    return filtered_tokens

def preprocessing2text(text, op=0):
    """
    Devuelve el texto con preprocesamiento: limpiado, remove stopwords, and stemming or lemmatization.

    Args:
        text: texto crudo.
        op: stemming = 0, Lemmatization = 1

    Returns:
        text: texto preprocesado.
    """
    text = clean_text(text)
    text = normalized_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopword(tokens)
    if op == 0:
        tokens = stemming_tokens(tokens)
    elif  op == 1:
        text = " ".join(tokens)
        tokens = lemmatization_text(text)
    return " ".join(tokens)

def preprocessing2tokens(text, op=0):
    """
    Devuelve los tokes del texto con preprocesamiento: limpiado, remove stopwords, and stemming or lemmatization.

    Args:
        text: texto crudo.
        op: stemming = 0, Lemmatization = 1

    Returns:
        tokens: tokens del texto preprocesado.
    """
    text = clean_text(text)
    text = normalized_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopword(tokens)
    if op == 0:
        tokens = stemming_tokens(tokens)
    elif  op == 1:
        text = " ".join(tokens)
        tokens = lemmatization_text(text)
    else:
        pass
    return tokens

def build_inverted_index(corpus):
    """
        Crea un indice inverido para el corpus.

        Args:
            corpus: corpus previamente preprocesado.

        Returns:
            dict: indice inverdido.
    """
    indice = defaultdict(dict)
    for doc_id, tokens in corpus.items():
        contador = Counter(tokens)
        for termino, freq in contador.items():
            indice[termino][doc_id] = freq
    return dict(indice)