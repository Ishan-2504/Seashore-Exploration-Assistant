import argparse
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

file = open("vectorstore.pkl", 'rb')
vector_store = pickle.load(file)
file.close()

retriever = vector_store.as_retriever()

llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 128,
        "top_k": 5,
        "temperature": 0.2,
        "repetition_penalty": 1.03,
        "stream": True,
    },
    huggingfacehub_api_token="!!!Replace with hugging face api key!!!",
)

template = """Answer the question based only on the following context and if you do not know the answer just answer 'I do not have this information':

    {context}

    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

def extract_answer(response):
    if "Answer:" in response:
        return response.split("Answer:")[1].strip() 
    return response.strip() 

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_message = request.message
    try:
        bot_message = chain.invoke(user_message)
        answer = extract_answer(bot_message)
        return ChatResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
