import pandas as pd
from mceclib.jaccard import search_jaccard

from typing import List, Dict, Set, Any, Callable


def calcular_precision_recall(resultados_df, ids_relevantes, k = 10) -> Dict[str, float]:
    """
    Calcula la precisión y el recall en el top K para una sola consulta.
    """
    top_k_resultados = resultados_df.head(k)
    ids_obtenidos = set(top_k_resultados['id'])

    interseccion = ids_obtenidos.intersection(ids_relevantes)

    precision = len(interseccion) / k if k > 0 else 0.0
    total_relevantes = len(ids_relevantes)
    recall = len(interseccion) / total_relevantes if total_relevantes > 0 else 0.0

    return {'precision@k': precision, 'recall@k': recall}


def calcular_ap(resultados_df, ids_relevantes) -> float:
    """
    Calcula la Average Precision (AP) para una sola consulta.
    """
    if not ids_relevantes:
        return 0.0

    ids_obtenidos = resultados_df['id'].tolist()
    ap = 0.0
    relevantes_encontrados = 0

    for i, doc_id in enumerate(ids_obtenidos):
        if doc_id in ids_relevantes:
            relevantes_encontrados += 1
            # La precisión en este punto de la lista
            precision_at_i = relevantes_encontrados / (i + 1)
            ap += precision_at_i

    # Normalizamos por el número total de documentos relevantes
    return ap / len(ids_relevantes)


def calcular_map(qrels, search_function: Callable, **search_args: Any) -> Dict[
    str, float]:
    """
    Calcula el Mean Average Precision (MAP) para todo el sistema.
    """
    aps = {}
    total_ap = 0.0

    for i, (query_text, ids_relevantes_list) in enumerate(qrels.items()):
        # --- Punto clave: Llamada dinámica a la función de búsqueda ---
        # Pasamos la query y todos los argumentos adicionales almacenados en search_args
        resultados_df = search_function(query_text, **search_args)
        # -----------------------------------------------------------

        ids_relevantes_set = set(ids_relevantes_list)
        ap = calcular_ap(resultados_df, ids_relevantes_set)
        aps[query_text] = ap
        total_ap += ap

    map_score = total_ap / len(qrels) if len(qrels) > 0 else 0.0

    return {'MAP': map_score, 'APs_individuales': aps}