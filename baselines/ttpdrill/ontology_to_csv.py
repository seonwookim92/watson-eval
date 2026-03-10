import json
import pandas as pd
import re
import os

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def normalize_name(name):
    # Remove namespace if present
    if ':' in name:
        name = name.split(':')[-1]
    # Split camel case
    words = camel_case_split(name)
    return ' '.join(words).lower()

def convert_ontology_to_csv(data_path, output_dir):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for ont_name, ont_data in data.items():
        rows = []
        # Process Classes
        for cls in ont_data.get('classes', []):
            norm_name = normalize_name(cls)
            rows.append({
                'tactic': 'Class',
                'technique': ont_name,
                'Id': cls,
                'action_what': norm_name,
                'action_where': norm_name,
                'where_attribute': '',
                'when': '',
                'when_condition': '',
                'when_object': '',
                'why_state': '',
                'why_what': '',
                'why_where': '',
                'how': '',
                'how_what': '',
                'how_where': '',
                'precondition': '',
                'postcondition': '',
                'alternative_technique': ''
            })
            
        # Process Properties
        for prop in ont_data.get('object_properties', []):
            norm_name = normalize_name(prop)
            rows.append({
                'tactic': 'Property',
                'technique': ont_name,
                'Id': prop,
                'action_what': norm_name,
                'action_where': norm_name,
                'where_attribute': '',
                'when': '',
                'when_condition': '',
                'when_object': '',
                'why_state': '',
                'why_what': '',
                'why_where': '',
                'how': '',
                'how_what': '',
                'how_where': '',
                'precondition': '',
                'postcondition': '',
                'alternative_technique': ''
            })
            
        df = pd.DataFrame(rows)
        # TTPDrill expects some extra header info in line 1? 
        # Looking at original ontology_details.csv:
        # 1: 0,0,1,2,tactic,technique,Id,action_what,action_where,where_attribute,when,when_condition,when_object,why_state,why_what,why_where,how,how_what,how_where,precondition,postcondition,alternative_technique,
        
        output_path = os.path.join(output_dir, f'ontology_details_{ont_name}.csv')
        df.to_csv(output_path, index=False)
        print(f"Generated {output_path} with {len(rows)} entries.")

if __name__ == "__main__":
    convert_ontology_to_csv('/tmp/cleaned_ontology_data.json', 'resources')
