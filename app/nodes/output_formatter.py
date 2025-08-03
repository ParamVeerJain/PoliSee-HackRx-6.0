async def output_formatter_node(state):
    return {
        "query": state["query"],
        "entities": state["entities"],
        "clauses": state["clauses"],
        "decision": state["decision"]
    }
