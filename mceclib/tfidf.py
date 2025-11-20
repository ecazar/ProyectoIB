import pandas as pd
from mceclib.preprocessing import preprocessing2text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf(corpus_preprocessing):
    """
        Procesa el corpus y genera la matriz TF-IDF y el vectorizador.

        Args:
            corpus_preprocessing (Series): corpus previamente limpiado y tokenizado.

        Returns:
            tfidf_matrix: matriz de TF-IDF.
            tfidf_vectorizer: vectorizador entrenado de TF-IDF.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_preprocessing)
    # El vectorizer_instance guarda el vocabulario y pesos IDF
    return tfidf_matrix, tfidf_vectorizer

def search_tdifd(query_text, vectorizer, tfidf_matrix, corpus, op=0):
    """
            Procesa un query, calcula la similitud TF-IDF y devuelve los resultados en un DataFrame.

            Args:
                query_text (str): El texto de la consulta cruda.
                vectorizer: vectorizador entrenado de TF-IDF.
                tfidf_matrix: matriz de TF-IDF.
                corpus (DataFrame): El corpus original (lista de textos crudos).
                op: Opción para stemming (0) o lematización (1) en el preprocesamiento.

            Returns:
                results_df: DataFrame con textos originales y scores de TF-IDF.
    """
    query = preprocessing2text(query_text, op=op)
    query_vector = vectorizer.transform([query])
    cosine_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    results_df = pd.DataFrame({
        'id': corpus[corpus.columns[0]],
        'text': corpus[corpus.columns[1]],
        'scores': cosine_scores
    })
    results_df = results_df.sort_values(by='scores', ascending=False)
    results_df = results_df.reset_index(drop=True)
    return results_df

def simple_search_tdifd(query_text, vectorizer, tfidf_matrix, top_k=5, op=0):
    """
            Procesa un query, lo vectoriza, calcula la similitud y devuelve los primeros k resultados.

            Args:
                query_text (str): El texto de la consulta cruda.
                vectorizer: vectorizador entrenado de TF-IDF.
                tfidf_matrix: matriz de TF-IDF.
                top_k (int): El número de documentos principales (ranking k) a devolver.
                op: Opción para stemming (0) o lematización (1) en el preprocesamiento.

            Returns:
                list[tuple]: muestra los indices de ranking k de documentos recuperados com mayor indice
    """
    query = preprocessing2text(query_text, op=op)
    query_vector = vectorizer.transform([query])
    sim = cosine_similarity(query_vector, tfidf_matrix)[0]
    ranking = sim.argsort()[::-1][:top_k]
    return [(i, sim[i]) for i in ranking]
