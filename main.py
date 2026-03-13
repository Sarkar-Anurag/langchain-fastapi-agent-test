from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_classic.agents import AgentExecutor
from agent import create_agent
from tools import tools
from typing import Any

app = FastAPI(
    title="LangChain Agent with FastAPI",
    description=" A FastAPI server exposing a LangChain agent powered by Ollama",
    version="1.0.0"
    )

agent = create_agent()

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    answer: str
    intermediate_steps: list[Any] | None = None

@app.post("/api/chat", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        result = await agent_executor.ainvoke(
            {"input": request.query}
        )
        return AgentResponse(
            answer=result["output"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

# @app.get("/api/chat/response", response_model=AgentResponse)
# def get_response():
#     return AgentResponse()

# 6. Create the GET API Endpoint
# @app.get("/api/v1/chat", response_model=AgentResponse)
# async def agent_get(query: str):
#     """
#     GET endpoint for the agent. 
#     The input is passed directly in the URL as a query parameter.
#     """
#     try:
#         # The prompt now comes from the 'query' variable extracted from the URL
#         result = await agent_executor.ainvoke({"input": query})
        
#         return AgentResponse(
#             answer=result["output"]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="localhost", port=8000)

# @app.post("/agent", response_model=AgentResponse)
# def run_agent(request: AgentRequest):
#     result = agent_executor.invoke(
#         {"input": request.query}
#     )
#     return AgentResponse(
#         answer=result["output"]
#     )
