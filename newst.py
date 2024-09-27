import argparse
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import pickle
from langchain.chains import ConversationalRetrievalChain

# pip install -U langchain-community
from langchain_community.vectorstores import FAISS

# pip install -U langchain-huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

file = open("vectorstore.pkl",'rb')
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
        "stream" : True,
    },
    huggingfacehub_api_token= "[hugging face api key]",
)

template = """Answer the question based only on the following context and if you do not know the answer just answer 'I do not have this information':

    {context}

    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)
model = llm
chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
def extract_answer(response):
    if "Answer:" in response:
        return response.split("Answer:")[1].strip() 
    return response.strip() 
user_message = "Do you know where NSUT in Delhi"
bot_message = chain.invoke(user_message)

print(extract_answer(bot_message))

