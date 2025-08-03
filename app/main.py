from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from app.graph import build_claim_graph

app = FastAPI(title="PoliSee - HackRx")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

claim_graph = build_claim_graph()

@app.post("/analyze-claim/")
async def analyze_claim(query: str = Form(...)):
    result = await claim_graph.ainvoke({"query": query})
    return result
