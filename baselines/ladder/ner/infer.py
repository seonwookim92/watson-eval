import cyner
import os

model = cyner.CyNER(transformer_model='xlm-roberta-base', use_heuristic=True, flair_model=None, spacy_model=None, priority='HTFS')

with open('sample_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

entities = model.get_entities(text)

with open('sample_output.txt', 'w', encoding='utf-8') as f:
    for e in entities:
        f.write(f"{e}\n")
        print(e)
