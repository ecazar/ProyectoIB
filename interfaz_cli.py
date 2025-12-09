from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Button, Static, Input, DataTable,
    TabbedContent, TabPane, Select, TextArea, Label
)
from textual.binding import Binding
from textual.screen import Screen
from textual.reactive import reactive

import gzip
import json
import pandas as pd
from mceclib.preprocessing import preprocessing2text, tokenize_text
from mceclib.jaccard import search_jaccard
from mceclib.tfidf import build_tfidf, search_tdifd
from mceclib.bm25 import build_bm25_model, search_bm25
from mceclib.evaluate import calcular_map, calcular_precision_recall


class DocumentDetailScreen(Screen):
    """Pantalla para ver documento completo"""

    def __init__(self, doc_id, text, preprocessed=None, tokenized=None):
        super().__init__()
        self.doc_id = doc_id
        self.text = text
        self.preprocessed = preprocessed
        self.tokenized = tokenized

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer():
            yield Label(f"[bold]Documento ID:[/bold] {self.doc_id}")
            yield Label("[bold cyan]Texto Original:[/bold cyan]")
            yield Static(self.text, classes="document-text")

            if self.preprocessed:
                yield Label("[bold green]Texto Preprocesado:[/bold green]")
                yield Static(self.preprocessed, classes="document-text")

            if self.tokenized:
                yield Label("[bold yellow]Tokens:[/bold yellow]")
                yield Static(str(self.tokenized), classes="document-text")

            yield Button("Cerrar", variant="primary", id="close")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.app.pop_screen()


class DataFrameViewScreen(Screen):
    """Pantalla para ver DataFrame completo de resultados"""

    def __init__(self, method, df_text, df):
        super().__init__()
        self.method = method
        self.df_text = df_text
        self.df = df

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer():
            yield Static(self.df_text, classes="dataframe-view")
            yield Button("Cerrar", variant="primary", id="close")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.app.pop_screen()


class CorpusPanel(Static):
    """Panel para visualizar el corpus"""

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]📚 Panel de Corpus[/bold cyan]")
        yield Label(f"Total de documentos: {len(self.app_ref.corpus) if self.app_ref.corpus is not None else 0}")

        with Horizontal(classes="button-group"):
            yield Button("Ver Original", id="view_original", variant="success")
            yield Button("Ver Preprocesado", id="view_preprocessed", variant="success")
            yield Button("Ver Tokenizado", id="view_tokenized", variant="success")

        yield DataTable(id="corpus_table")

    def on_mount(self) -> None:
        table = self.query_one("#corpus_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Texto (preview)")

        if self.app_ref.corpus is not None:
            for idx, row in self.app_ref.corpus.head(50).iterrows():
                text_preview = row['text'][:200] + "..." if len(row['text']) > 200 else row['text']
                table.add_row(row['_id'], text_preview)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        table = self.query_one("#corpus_table", DataTable)

        if table.cursor_row < 0:
            self.app_ref.notify("Selecciona un documento en la tabla", severity="warning")
            return

        row_key = table.get_row_at(table.cursor_row)
        doc_id = row_key[0]

        # Find document
        doc = self.app_ref.corpus[self.app_ref.corpus['_id'] == doc_id].iloc[0]
        text = doc['text']

        preprocessed = None
        tokenized = None

        if event.button.id == "view_preprocessed" or event.button.id == "view_tokenized":
            idx = self.app_ref.corpus[self.app_ref.corpus['_id'] == doc_id].index[0]
            preprocessed = self.app_ref.corpus_clean.iloc[idx] if self.app_ref.corpus_clean is not None else None

            if event.button.id == "view_tokenized":
                tokenized = self.app_ref.corpus_clean_tokenized.iloc[
                    idx] if self.app_ref.corpus_clean_tokenized is not None else None

        self.app_ref.push_screen(DocumentDetailScreen(doc_id, text, preprocessed, tokenized))


class PreprocessingPanel(Static):
    """Panel para probar preprocesamiento"""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]🔧 Panel de Preprocesamiento[/bold cyan]")
        yield Label("Texto Original:")
        yield TextArea(id="original_text", classes="textarea")

        yield Button("Aplicar Preprocesamiento", id="apply_preprocess", variant="primary")

        yield Label("Texto Preprocesado:")
        yield TextArea(id="preprocessed_text", classes="textarea", read_only=True)

        yield Label("Tokens Resultantes:")
        yield TextArea(id="tokens_text", classes="textarea", read_only=True)


class SearchPanel(Static):
    """Panel principal de búsqueda"""

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref
        self.last_results = {}  # Para almacenar los últimos resultados

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]🔍 Panel de Búsqueda[/bold cyan]")

        with Horizontal(classes="search-controls"):
            yield Label("Query:")
            yield Input(placeholder="Ingresa tu consulta...", id="search_query")
            yield Label("Top-k:")
            yield Select(
                [(str(k), k) for k in [5, 10, 20, 50]],
                value=10,
                id="topk_select"
            )

        with TabbedContent():
            with TabPane("Jaccard"):
                yield Button("Buscar (Jaccard)", id="search_jaccard", variant="primary")
                yield DataTable(id="results_jaccard")

            with TabPane("TF-IDF"):
                yield Button("Buscar (TF-IDF)", id="search_tfidf", variant="primary")
                yield DataTable(id="results_tfidf")

            with TabPane("BM25"):
                yield Button("Buscar (BM25)", id="search_bm25", variant="primary")
                yield DataTable(id="results_bm25")

            with TabPane("Comparar"):
                with Vertical():
                    yield Button("Comparar Todos", id="compare_all", variant="success")
                    with ScrollableContainer(classes="compare-scroll-container"):
                        yield Label("[bold]Jaccard[/bold]", classes="compare-label")
                        yield DataTable(id="compare_jaccard", classes="compare-table")
                        yield Label("[bold]TF-IDF[/bold]", classes="compare-label")
                        yield DataTable(id="compare_tfidf", classes="compare-table")
                        yield Label("[bold]BM25[/bold]", classes="compare-label")
                        yield DataTable(id="compare_bm25", classes="compare-table")

    def on_mount(self) -> None:
        for table_id in ["results_jaccard", "results_tfidf", "results_bm25",
                         "compare_jaccard", "compare_tfidf", "compare_bm25"]:
            table = self.query_one(f"#{table_id}", DataTable)
            table.add_columns("Rank", "Score", "ID", "Texto")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        query = self.query_one("#search_query", Input).value
        topk = self.query_one("#topk_select", Select).value

        if event.button.id in ["search_jaccard", "search_tfidf", "search_bm25"]:
            if not query.strip():
                self.app.notify("Ingresa una consulta", severity="warning")
                return

        if event.button.id == "search_jaccard":
            self._search_method("jaccard", query, topk)
        elif event.button.id == "search_tfidf":
            self._search_method("tfidf", query, topk)
        elif event.button.id == "search_bm25":
            self._search_method("bm25", query, topk)
        elif event.button.id == "compare_all":
            if not query.strip():
                self.app.notify("Ingresa una consulta", severity="warning")
                return
            self._compare_all(query, topk)

    def _search_method(self, method, query, topk):
        try:
            if method == "jaccard":
                results = search_jaccard(query, self.app_ref.corpus_clean_tokenized, self.app_ref.corpus)
                table = self.query_one("#results_jaccard", DataTable)
                self.last_results['jaccard'] = results
            elif method == "tfidf":
                results = search_tdifd(query, self.app_ref.vectorizer, self.app_ref.matriz_tfidf, self.app_ref.corpus)
                table = self.query_one("#results_tfidf", DataTable)
                self.last_results['tfidf'] = results
            elif method == "bm25":
                results = search_bm25(query, self.app_ref.bm25_model, self.app_ref.corpus)
                table = self.query_one("#results_bm25", DataTable)
                self.last_results['bm25'] = results

            table.clear()

            # Obtener las primeras topk filas
            top_results = results.head(topk)

            # Determinar nombres de columnas (pueden variar entre métodos)
            score_col = 'scores' if 'scores' in top_results.columns else 'score'
            id_col = 'id' if 'id' in top_results.columns else '_id'

            # Obtener índices de columnas
            col_score = top_results.columns.get_loc(score_col)
            col_id = top_results.columns.get_loc(id_col)
            col_text = top_results.columns.get_loc('text')

            # Iterar usando valores directos
            for rank in range(len(top_results)):
                row_values = top_results.iloc[rank].values

                score_val = row_values[col_score]
                doc_id = row_values[col_id]
                text_val = row_values[col_text]

                text_preview = str(text_val)[:100] + "..." if len(str(text_val)) > 100 else str(text_val)
                score_str = f"{float(score_val):.4f}"
                doc_id_str = str(doc_id)

                table.add_row(str(rank + 1), score_str, doc_id_str, text_preview)

            self.app.notify(f"Búsqueda {method.upper()} completada - {len(results)} resultados totales",
                            severity="information")
        except Exception as e:
            self.app.notify(f"Error en búsqueda: {str(e)}", severity="error")
            import traceback
            self.app.notify(f"Detalle: {traceback.format_exc()}", severity="error")

    def _view_full_dataframe(self, method):
        """Muestra el DataFrame completo de resultados"""
        if method not in self.last_results:
            self.app.notify(f"No hay resultados de {method.upper()} para mostrar. Realiza una búsqueda primero.",
                            severity="warning")
            return

        df = self.last_results[method]

        # Crear texto formateado del DataFrame
        df_text = f"[bold cyan]DataFrame completo de {method.upper()}[/bold cyan]\n"
        df_text += f"[bold]Shape:[/bold] {df.shape[0]} filas x {df.shape[1]} columnas\n\n"
        df_text += f"[bold]Columnas:[/bold] {list(df.columns)}\n\n"
        df_text += "[bold]Primeras 20 filas:[/bold]\n"
        df_text += str(df.head(20))

        # Mostrar en una pantalla modal
        self.app.push_screen(DataFrameViewScreen(method, df_text, df))

    def _compare_all(self, query, topk):
        try:
            results = {
                'jaccard': search_jaccard(query, self.app_ref.corpus_clean_tokenized, self.app_ref.corpus),
                'tfidf': search_tdifd(query, self.app_ref.vectorizer, self.app_ref.matriz_tfidf, self.app_ref.corpus),
                'bm25': search_bm25(query, self.app_ref.bm25_model, self.app_ref.corpus)
            }

            for method, result_df in results.items():
                table = self.query_one(f"#compare_{method}", DataTable)
                table.clear()

                # Obtener las primeras topk filas
                top_results = result_df.head(topk)

                # Determinar nombres de columnas (pueden variar entre métodos)
                score_col = 'scores' if 'scores' in top_results.columns else 'score'
                id_col = 'id' if 'id' in top_results.columns else '_id'

                # Obtener índices de columnas
                col_score = top_results.columns.get_loc(score_col)
                col_id = top_results.columns.get_loc(id_col)
                col_text = top_results.columns.get_loc('text')

                # Iterar usando valores directos
                for rank in range(len(top_results)):
                    row_values = top_results.iloc[rank].values

                    score_val = row_values[col_score]
                    doc_id = row_values[col_id]
                    text_val = row_values[col_text]

                    text_preview = str(text_val)[:80] + "..." if len(str(text_val)) > 80 else str(text_val)
                    score_str = f"{float(score_val):.4f}"
                    doc_id_str = str(doc_id)

                    table.add_row(str(rank + 1), score_str, doc_id_str, text_preview)

            self.app.notify("Comparación completada", severity="information")
        except Exception as e:
            self.app.notify(f"Error en comparación: {str(e)}", severity="error")
            import traceback
            self.app.notify(f"Detalle: {traceback.format_exc()}", severity="error")


class EvaluationPanel(Static):
    """Panel de evaluación"""

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]📊 Panel de Evaluación[/bold cyan]")

        with Horizontal(classes="button-group"):
            yield Button("Calcular MAP", id="calc_map", variant="primary")
            yield Button("Calcular Precision@k/Recall@k", id="calc_pr", variant="primary")

        yield Label("[bold]Configuración Precision/Recall:[/bold]", classes="section-label")
        with Vertical(classes="pr-config"):
            with Horizontal(classes="input-row"):
                yield Label("Query:", classes="input-label")
                yield Input(placeholder="Ej: benefits of vegetarian diet", id="pr_query", classes="pr-input")
            with Horizontal(classes="input-row"):
                yield Label("k:", classes="input-label")
                yield Input(placeholder="10", id="pr_k", value="10", classes="k-input")

        # Contenedor horizontal para MAP y Precision/Recall
        with Horizontal(classes="results-container"):
            with Vertical(classes="map-section"):
                yield Label("[bold]Resultados MAP:[/bold]", classes="section-label")
                yield DataTable(id="map_results", classes="map-table")

            with Vertical(classes="pr-section"):
                yield Label("[bold]Precision@k y Recall@k:[/bold]", classes="section-label")
                with ScrollableContainer(classes="pr-scroll-container"):
                    yield Static(id="pr_results", classes="pr-results-content")

        yield Label("[bold]QRels (muestra):[/bold]", classes="section-label")
        yield DataTable(id="qrels_table")

    def on_mount(self) -> None:
        map_table = self.query_one("#map_results", DataTable)
        map_table.add_columns("Método", "MAP")

        qrels_table = self.query_one("#qrels_table", DataTable)
        qrels_table.add_columns("Query", "IDs Relevantes")

        if self.app_ref.qrels_dict:
            for query, ids in list(self.app_ref.qrels_dict.items())[:10]:
                qrels_table.add_row(query, str(ids[:5]) + "...")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "calc_map":
            self._calculate_map()
        elif event.button.id == "calc_pr":
            self._calculate_precision_recall()

    def _calculate_map(self):
        try:
            self.app.notify("Calculando MAP... esto puede tomar un momento", severity="information")

            map_jaccard = calcular_map(
                self.app_ref.qrels_dict,
                search_function=search_jaccard,
                corpus_preprocessing=self.app_ref.corpus_clean_tokenized,
                corpus=self.app_ref.corpus
            )

            map_tfidf = calcular_map(
                self.app_ref.qrels_dict,
                search_function=search_tdifd,
                vectorizer=self.app_ref.vectorizer,
                tfidf_matrix=self.app_ref.matriz_tfidf,
                corpus=self.app_ref.corpus
            )

            map_bm25 = calcular_map(
                self.app_ref.qrels_dict,
                search_function=search_bm25,
                bm25_model=self.app_ref.bm25_model,
                corpus=self.app_ref.corpus
            )

            table = self.query_one("#map_results", DataTable)
            table.clear()
            table.add_row("Jaccard", f"{map_jaccard['MAP']:.6f}")
            table.add_row("TF-IDF", f"{map_tfidf['MAP']:.6f}")
            table.add_row("BM25", f"{map_bm25['MAP']:.6f}")

            self.app.notify("MAP calculado exitosamente", severity="information")
        except Exception as e:
            self.app.notify(f"Error calculando MAP: {str(e)}", severity="error")

    def _calculate_precision_recall(self):
        query = self.query_one("#pr_query", Input).value
        k_str = self.query_one("#pr_k", Input).value

        if not query.strip():
            self.app.notify("Ingresa una query", severity="warning")
            return

        try:
            k = int(k_str)
        except:
            self.app.notify("k debe ser un número entero", severity="error")
            return

        if query not in self.app_ref.qrels_dict:
            self.app.notify("Query no encontrada en qrels", severity="warning")
            return

        try:
            ids_relevantes = set(self.app_ref.qrels_dict[query])

            result_jaccard = search_jaccard(query, self.app_ref.corpus_clean_tokenized, self.app_ref.corpus)
            result_tfidf = search_tdifd(query, self.app_ref.vectorizer, self.app_ref.matriz_tfidf, self.app_ref.corpus)
            result_bm25 = search_bm25(query, self.app_ref.bm25_model, self.app_ref.corpus)

            metrics_j = calcular_precision_recall(result_jaccard, ids_relevantes, k=k)
            metrics_t = calcular_precision_recall(result_tfidf, ids_relevantes, k=k)
            metrics_b = calcular_precision_recall(result_bm25, ids_relevantes, k=k)

            result_text = f"""
[bold]Query:[/bold] {query}
[bold]k:[/bold] {k}

[bold cyan]Jaccard:[/bold cyan]
  Precision@{k}: {metrics_j['precision@k']:.6f}
  Recall@{k}: {metrics_j['recall@k']:.6f}

[bold green]TF-IDF:[/bold green]
  Precision@{k}: {metrics_t['precision@k']:.6f}
  Recall@{k}: {metrics_t['recall@k']:.6f}

[bold yellow]BM25:[/bold yellow]
  Precision@{k}: {metrics_b['precision@k']:.6f}
  Recall@{k}: {metrics_b['recall@k']:.6f}


─────────────────────────────────────────────────





"""

            self.query_one("#pr_results", Static).update(result_text)
            self.app.notify("Precision/Recall calculado", severity="information")
        except Exception as e:
            self.app.notify(f"Error: {str(e)}", severity="error")


class SearchApp(App):
    """Aplicación principal"""

    CSS = """
    Screen {
        background: $surface;
    }

    .textarea {
        height: 6;
        margin: 1;
    }

    .button-group {
        height: auto;
        margin: 1;
    }

    .search-controls {
        height: auto;
        margin: 1;
        align: center middle;
    }

    DataTable {
        height: 15;
        margin: 1;
    }

    .document-text {
        margin: 1;
        padding: 1;
        background: $boost;
    }

    .results-text {
        margin: 1;
        padding: 1;
        background: $boost;
        height: auto;
        min-height: 10;
    }

    .dataframe-view {
        margin: 1;
        padding: 1;
        background: $boost;
        height: auto;
    }

    Input {
        width: 1fr;
    }

    Select {
        width: 15;
    }

    .section-label {
        margin-top: 2;
        margin-bottom: 1;
    }

    .pr-config {
        margin: 1;
        padding: 1;
        background: $boost;
        height: auto;
    }

    .input-row {
        height: auto;
        margin: 1;
        align: left middle;
    }

    .input-label {
        width: 10;
        margin-right: 1;
    }

    .pr-input {
        width: 1fr;
    }

    .k-input {
        width: 15;
    }

    .results-container {
        height: 40;
        margin: 1;
    }

    .map-section {
        width: 1fr;
        margin-right: 1;
        height: 100%;
    }

    .map-table {
        height: 8;
        margin: 1;
    }

    .pr-section {
        width: 1fr;
        height: 100%;
        margin-left: 1;
    }

    .pr-scroll-container {
        width: 100%;
        height: 25;
        border: tall $accent;
        margin-top: 1;
    }

    .pr-results-content {
        margin: 1;
        padding: 1;
        background: $boost;
        height: 50;
    }

    .compare-scroll-container {
        height: 80vh;
        width: 100%;
        border: tall $accent;
        margin-top: 1;
        overflow-y: auto;
    }

    .compare-table {
        height: 25;
        margin: 1 2;
        width: auto;
        min-height: 25;
    }

    .compare-label {
        margin: 2 2 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Salir"),
        Binding("1", "show_corpus", "Corpus"),
        Binding("2", "show_search", "Buscar"),
        Binding("3", "show_evaluation", "Evaluar"),
    ]

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.corpus_clean = None
        self.corpus_clean_tokenized = None
        self.vectorizer = None
        self.matriz_tfidf = None
        self.bm25_model = None
        self.qrels_dict = None
        self.current_panel = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(classes="button-group"):
            yield Button("1. Corpus", id="btn_corpus", variant="success")
            yield Button("2. Buscar", id="btn_search", variant="success")
            yield Button("3. Evaluar", id="btn_eval", variant="success")

        yield Container(id="main_panel")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Sistema de Búsqueda - IR"
        self.notify("Cargando datos...", severity="information")
        self._load_data()
        self.action_show_corpus()

    def _load_data(self):
        try:
            # Load corpus
            with gzip.open('./data/corpus.jsonl.gz', 'rt', encoding='utf-8') as file:
                lines = file.readlines()
            data = [json.loads(line) for line in lines]
            self.corpus = pd.DataFrame(data)[['_id', 'text']]

            # Preprocess
            corpus_text = self.corpus["text"]
            self.corpus_clean = corpus_text.apply(preprocessing2text)
            self.corpus_clean_tokenized = self.corpus_clean.apply(tokenize_text)

            # Build models
            self.matriz_tfidf, self.vectorizer = build_tfidf(self.corpus_clean)
            self.bm25_model = build_bm25_model(self.corpus_clean_tokenized)

            # Load qrels
            with gzip.open('./data/train.jsonl.gz', 'rt', encoding='utf-8') as file:
                lines = file.readlines()
            queriesdt = [json.loads(line) for line in lines]
            qrels = pd.DataFrame(queriesdt)[['_id', 'text', 'query']].head(100)
            self.qrels_dict = qrels.groupby('query')['_id'].apply(list).to_dict()

            self.notify("Datos cargados exitosamente", severity="information")
        except Exception as e:
            self.notify(f"Error cargando datos: {str(e)}", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_corpus":
            self.action_show_corpus()
        elif event.button.id == "btn_search":
            self.action_show_search()
        elif event.button.id == "btn_eval":
            self.action_show_evaluation()

    def action_show_corpus(self) -> None:
        container = self.query_one("#main_panel", Container)
        container.remove_children()
        container.mount(CorpusPanel(self))

    def action_show_search(self) -> None:
        container = self.query_one("#main_panel", Container)
        container.remove_children()
        container.mount(SearchPanel(self))

    def action_show_evaluation(self) -> None:
        container = self.query_one("#main_panel", Container)
        container.remove_children()
        container.mount(EvaluationPanel(self))


if __name__ == "__main__":
    app = SearchApp()
    app.run()