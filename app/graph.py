from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph
from app.nodes.entity_extraction import entity_extraction_node
from app.nodes.clause_retrieval import clause_retrieval_node
from app.nodes.decision_engine import decision_engine_node
from app.nodes.output_formatter import output_formatter_node

# Define pipeline state schema
class ClaimState(TypedDict):
    query: str
    entities: Dict[str, Any]
    clauses: List[Dict[str, Any]]
    decision: Dict[str, Any]  # Structured JSON from LLM

def build_claim_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("clause_retrieval", clause_retrieval_node)
    graph.add_node("decision_engine", decision_engine_node)
    graph.add_node("output_formatter", output_formatter_node)

    graph.add_edge("entity_extraction", "clause_retrieval")
    graph.add_edge("clause_retrieval", "decision_engine")
    graph.add_edge("decision_engine", "output_formatter")

    graph.set_entry_point("entity_extraction")
    graph.set_finish_point("output_formatter")

    return graph.compile()
