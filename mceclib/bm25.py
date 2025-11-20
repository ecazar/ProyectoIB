from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from mceclib.preprocessing import preprocessing2tokens


def build_bm25_model(corpus_preprocessing):
    """
    Procesa el corpus, tokeniza y entrena el modelo BM25.

    Args:
        corpus_preprocessing: corpus previamente limpiado y tokenizado.

    Returns:
        bm25_model (BM25Okapi): El modelo BM25 entrenado.
        corpus_tokens (list[list[str]]): El corpus tokenizado usado para el entrenamiento.
    """
    bm25_model = BM25Okapi(corpus_preprocessing)
    return bm25_model


# Asume que build_bm25_model y preprocessing2tokens están definidos arriba

def search_bm25(query_text, bm25_model, corpus, op=0):
    """
    Procesa un query, calcula la similitud BM25 y devuelve los resultados en un DataFrame.

    Args:
        query_text (str): El texto de la consulta cruda.
        bm25_model (BM25Okapi): El modelo BM25 entrenado.
        corpus (list[str]): El corpus original (lista de textos crudos).
        op: Opción para stemming (0) o lematización (1) en el preprocesamiento.

    Returns:
        results_df: DataFrame con textos originales y scores de BM25.
    """
    query_tokens = preprocessing2tokens(query_text, op=op)
    doc_scores = bm25_model.get_scores(query_tokens)
    results_df = pd.DataFrame({
        'doc_index': corpus.index,
        'reviews': corpus,  # Texto original del documento
        'scores': doc_scores  # Puntuación BM25 obtenida
    })
    results_df = results_df.sort_values(by='scores', ascending=False)
    results_df = results_df.reset_index(drop=True)
    return results_df



def simple_search_bm25(query_text, bm25_model, top_k=5, op=0):
    """
    Procesa un query, calcula la similitud BM25 y devuelve los primeros k resultados.

    Args:
        query_text (str): El texto de la consulta cruda.
        bm25_model (BM25Okapi): El modelo BM25 entrenado.
        corpus (list[str]): El corpus original (lista de textos crudos).
        top_k (int): El número de documentos principales (ranking k) a devolver.
        op: Opción para stemming (0) o lematización (1) en el preprocesamiento.

    Returns:
        list[tuple]: muestra los indices de ranking k de documentos recuperados com mayor indice
    """
    query_tokens = preprocessing2tokens(query_text, op=op)
    doc_scores = bm25_model.get_scores(query_tokens)
    scores_array = np.array(doc_scores)
    ranking_indices = scores_array.argsort()[::-1][:top_k]
    results = []
    for i in ranking_indices:
        results.append((i, scores_array[i]))
    return results
