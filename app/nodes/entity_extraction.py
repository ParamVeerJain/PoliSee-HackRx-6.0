from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

async def entity_extraction_node(state):
    query = state["query"]
    response_schemas = [
        ResponseSchema(name="age", description="Age of the person (integer)"),
        ResponseSchema(name="procedure", description="Medical procedure or treatment"),
        ResponseSchema(name="cause", description="Cause of claim, e.g., accident or illness"),
        ResponseSchema(name="location", description="Location of treatment, domestic/international/city"),
        ResponseSchema(name="policy_tenure", description="Policy active duration in months/years")
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_template("""
Extract the following details from the user query and return in JSON:
{format_instructions}

Query: {query}
""")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    response = await llm.ainvoke(
        prompt.format_prompt(query=query, format_instructions=format_instructions).to_messages()
    )
    entities = parser.parse(response.content)
    return {"query": query, "entities": entities}
