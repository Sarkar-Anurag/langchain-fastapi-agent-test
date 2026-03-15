# Agent with langchain and FastAPI

## Description
User can use this agent to calculate and ask any questions from LLM

## Installation
1. python version 3.10.0 (Recommended use venv)
2. Run `pip install -r requirements.txt`
3. Run `ollama pull llama3.1`

## Usage
1. Activate the venv in cmd 
2. Start the server with `uvicorn main:app --reload` 
3. It will start running in `http://localhost:8000` by default if port 8000 is ready to use.
4. Visit `http://localhost:8000/docs` in your browser to use the application.
5. Try it out the POST operation and give your query in Request Schema and execute
6. Agent will give you the response in the Response Schema 

## Contact
Email: sarkaranurag007@gmail.com