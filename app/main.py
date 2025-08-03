#main.py
from fastapi import FastAPI, Form
from graph import build_claim_graph

app = FastAPI(title="PoliSee - HackRx")

claim_graph = build_claim_graph()

@app.post("/analyze-claim/")
async def analyze_claim(query: str = Form(...)):
    result = await claim_graph.ainvoke({"query": query})
    return result
