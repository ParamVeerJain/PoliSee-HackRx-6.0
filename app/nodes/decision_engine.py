from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

async def decision_engine_node(state):
    entities = state["entities"]
    clauses = state["clauses"]
    response_schemas = [
        ResponseSchema(name="Decision", description="approved/rejected/partially approved"),
        ResponseSchema(name="Amount", description="Payout amount in INR"),
        ResponseSchema(name="Justification", description="Reasoning mapped to clauses and conditions")
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_template("""
You are an insurance claim evaluator.
Use the retrieved clauses and extracted entities to decide claim eligibility and payout.
Return structured JSON strictly in this format:
{format_instructions}
Clauses:
{clauses}
Entities:
{entities}
""")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    response = await llm.ainvoke(
        prompt.format_prompt(
            clauses=clauses,
            entities=entities,
            format_instructions=format_instructions
        ).to_messages()
    )
    decision = parser.parse(response.content)
    return {"query": state["query"], "entities": entities, "clauses": clauses, "decision": decision}
