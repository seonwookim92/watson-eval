import os
from relation_miner import relation_miner
from nltk.stem import WordNetLemmatizer, PorterStemmer
import itertools

# Core NLP Configuration
STANFORD_SERVER = 'http://127.0.0.1:9000' # Not used with Spacy but kept for compatibility
model_STANFORD = None # relation_miner will load Spacy automatically

# SRL fallback (disabled)
model_AllenNLP_SRL = None
model_AllenNLP_Coref = None

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

isLemmatize = True
isStem = True
# if reading from server, isFile=False
isFile = True
isGhaithOntology = False
isDependencyParser = True # CRITICAL: Use Dependency Parser (Spacy)
BM25_THRESHOLD = 0.1
# top_n represents the top matches with the mitre
top_n = 5

import preProcessTool as pretool
preprocessOntologies = pretool.preProcessTool(model_AllenNLP_SRL, model_STANFORD, isGhaithOntology)

import cyber_object
try:
    cyber_object.combine_cyber_object()
except:
    pass

import bm25_match
cyber_model, cyber_corpus = bm25_match.create_cyber_model()

# Ontology Model (Default to UCO)
ONTOLOGY_FILE = 'resources/ontology_details_uco.csv'
bm25_model, tokenized_corpus, ttp_id, bow_mapped = bm25_match.create_ontology_bm_model(file_name=ONTOLOGY_FILE)

# Empty extracted_list to avoid large blob
extracted_list = []

if __name__ == '__main__':
    pass
