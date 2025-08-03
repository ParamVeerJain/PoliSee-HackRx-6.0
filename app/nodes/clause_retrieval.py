from app.utils.vector_store_loader import load_vectorstore

async def clause_retrieval_node(state):
    vectorstore = load_vectorstore()
    entities = state["entities"]
    search_terms = " ".join([str(v) for v in entities.values() if v])
    results = vectorstore.similarity_search(search_terms, k=5)
    clauses = [{"text": r.page_content, "metadata": r.metadata} for r in results]
    return {"query": state["query"], "entities": entities, "clauses": clauses}
