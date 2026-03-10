from langgraph.graph import StateGraph, END
from core.pipeline.state import GraphState
from core.pipeline.nodes import (
    paraphrasing_node,
    ontology_mapping_node,
    visualization_node,
    next_step_router
)

def create_pipeline():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("paraphrase", paraphrasing_node)
    workflow.add_node("map_ontology", ontology_mapping_node)
    workflow.add_node("visualize", visualization_node)

    # Define edges
    workflow.set_entry_point("paraphrase")
    workflow.add_edge("paraphrase", "map_ontology")
    
    # Router for chunking
    workflow.add_conditional_edges(
        "map_ontology",
        next_step_router,
        {
            "paraphrase": "paraphrase",
            "visualize": "visualize"
        }
    )
    
    workflow.add_edge("visualize", END)

    return workflow.compile()
