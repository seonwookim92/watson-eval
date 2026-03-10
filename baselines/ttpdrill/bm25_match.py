import BM25
import helper
import os

def create_cyber_model():
    filepath = 'ontology/combined_cyber_object.txt'
    if not os.path.exists(filepath):
        return None, []
    
    try:
        content = helper.read_file(filepath).split('\n')
        corpus = [line.split() for line in content if line]
        return BM25.BM25Okapi(corpus), corpus
    except Exception as e:
        print(f"Error creating cyber model: {e}")
        return None, []

def query_cyber_object(model, query_text, corpus):
    if not model or not query_text:
        return True
    
    tokens = query_text.lower().split()
    scores = model.bm25.get_scores(tokens)
    # If any score is above a threshold, consider it a cyber object
    if any(s > 0.1 for s in scores):
        return True
    return False

def create_ontology_bm_model(file_name='resources/ontology_details.csv'):
    from ontology_reader import ReadOntology
    if not os.path.exists(file_name):
        return None, [], [], []
        
    try:
        ontology = ReadOntology()
        ontology_df = ontology.read_csv(file_name)
        # Convert to list of dicts
        ontology_list = ontology_df.to_dict('records')
        
        # Create a dict indexed by ID for TTPDrill's select_best_match
        # It expects ttp_df[ttp_id] = {'TECHNIQUE': ..., 'TACTIC': ...}
        ontology_dict = {}
        for row in ontology_list:
            ontology_dict[str(row['Id']).lower()] = {
                'TECHNIQUE': row.get('technique', ''),
                'TACTIC': row.get('tactic', '')
            }
        
        corpus = [str(row['action_what']).split() for row in ontology_list]
        ids = [row['Id'] for row in ontology_list]
        
        model = BM25.BM25Okapi(corpus)
        return model, corpus, ids, ontology_dict
    except Exception as e:
        print(f"Error creating ontology BM25 model: {e}")
        return None, [], [], []
