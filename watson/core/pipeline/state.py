from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    documents: List[Document]
    processed_text: str
    ontology_output: str
    source_info: str

    use_paraphrasing: bool
    verbose_graph: bool
    verbose: bool
    include_summaries: bool

    current_chunk_index: int
    accumulated_graph: str

    # Evaluation output: natural-language SRO triples [{subject, relation, object}]
    sro_triples: List[dict]
    # Evaluation output: extracted entities with ontology class [{name, ontology_class_short, ontology_class_uri}]
    entity_extractions: List[dict]
