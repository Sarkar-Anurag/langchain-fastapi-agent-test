from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import tools

def create_agent():
    llm = ChatOllama(
        model="llama3.1",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful local AI assistant. Use the provided tools when necessary to answer the user's questions."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt
    )

    return agent
