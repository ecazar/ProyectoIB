import json
import csv
import os
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "FIQA Dataset"
_SPLITS = ["corpus", "queries"]

URL = ""
_URLs = {subset: URL + f"{subset}.jsonl.gz" for subset in _SPLITS}

class BEIR(datasets.GeneratorBasedBuilder):
    """BEIR BenchmarkDataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
            description=f"This is the {name} in the FiQA dataset.",
        ) for name in _SPLITS
    ]
    
    def _info(self):

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
             features=datasets.Features({
                "_id": datasets.Value("string"), 
                "title": datasets.Value("string"),
                "text": datasets.Value("string"),
             }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=self.config.name,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            texts = f.readlines()
        for i, text in enumerate(texts):
            text = json.loads(text)
            if 'metadata' in text: del text['metadata']
            if "title" not in text: text["title"] = ""
            yield i, text