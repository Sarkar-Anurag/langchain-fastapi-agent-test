from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

# LangChain Imports
from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 1. Define Request and Response Schemas using Python 3.10 syntax
class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    answer: str
    intermediate_steps: list[Any] | None = None  # Python 3.10 Union syntax

# 2. Define Custom Tools
# We create a simple tool to give the agent an action it can perform
@tool
def calculate_string_length(text: str) -> int:
    """Returns the exact number of characters in a string."""
    return len(text)

tools = [calculate_string_length]

# 3. Initialize Local LLM and Agent
# We use Llama 3.1 as it has native support for tool calling
llm = ChatOllama(
    # model="gemma3:4b",
    model="llama3.1",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful local AI assistant. Use the provided tools when necessary to answer the user's questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent using tool calling capabilities
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. Initialize FastAPI Application
app = FastAPI(
    title="test 2: Local Ollama Agent API",
    description="A FastAPI server exposing a LangChain agent powered by Ollama",
    version="1.0.0"
)

# 5. Create the API Endpoint
@app.post("/api/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    try:
        # Use ainvoke for asynchronous execution (crucial for FastAPI)
        result = await agent_executor.ainvoke({"input": request.query})
        
        return AgentResponse(
            answer=result["output"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="localhost", port=8000)