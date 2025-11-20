import pandas as pd
import numpy as np
from mceclib.preprocessing import preprocessing2tokens


def jaccard_similarity(tokens_a, tokens_b):
    """
    Calcula el coeficiente de similitud de Jaccard entre dos listas de tokens.

    Args:
        tokens_a: Lista de tokens del primer texto.
        tokens_b: Lista de tokens del segundo texto.

    Returns:
        float: El coeficiente de Jaccard (entre 0.0 y 1.0).
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    interseccion = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return interseccion / union

def search_jaccard(query_text, corpus_preprocessing, corpus, op=0):
    """
    Procesa un query, calcula la similitud Jaccard y devuelve los resultados en un DataFrame.

    Args:
        query_text (str): El texto de la consulta cruda.
        corpus_preprocessing: corpus previamente limpiado y tokenizado.
        corpus (list[str]): El corpus original (lista de textos crudos).
        op: Opción para stemming (0) o lematización (1) en el preprocesamiento.

    Returns:
        results_df: DataFrame con textos originales y scores de Jaccard.
    """
    query_tokens = preprocessing2tokens(query_text, op=op)
    jaccard_scores = []
    for doc_tokens in corpus_preprocessing:
        score = jaccard_similarity(query_tokens, doc_tokens)
        jaccard_scores.append(score)
    results_df = pd.DataFrame({
        'doc_index': corpus.index,
        'reviews': corpus,
        'scores': jaccard_scores
    })
    results_df = results_df.sort_values(by='scores', ascending=False)
    results_df = results_df.reset_index(drop=True)
    return results_df

def simple_search_jaccard(query_text, corpus_preprocessing, top_k=5, op=0):
    """
    Procesa un query, calcula la similitud de Jaccard y devuelve los primeros k resultados.

    Args:
        query_text (str): El texto de la consulta cruda.
        corpus_preprocessing: corpus previamente limpiado y tokenizado.
        top_k (int): El número de documentos principales (ranking k) a devolver.
        op: Opción para stemming (0) o lematización (1) en el preprocesamiento.

    Returns:
        list[tuple]: muestra los indices de ranking k de documentos recuperados com mayor indice
    """
    query_tokens = preprocessing2tokens(query_text, op=op)
    jaccard_scores = []
    for doc_tokens in corpus_preprocessing:
        score = jaccard_similarity(query_tokens, doc_tokens)
        jaccard_scores.append(score)
    scores_array = np.array(jaccard_scores)
    ranking_indices = scores_array.argsort()[::-1][:top_k]
    results = []
    for i in ranking_indices:
        results.append((i, scores_array[i]))
    return results
