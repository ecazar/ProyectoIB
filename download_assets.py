import spacy
import nltk
import os


nltk.download('punkt', quiet=False)
nltk.download('stopwords', quiet=False)
nltk.download('punkt_tab')
model_name = "en_core_web_sm"
spacy.cli.download(model_name)

