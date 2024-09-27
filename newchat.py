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

# Set the API token as an environment variable within the script
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HF token"

# Initialize HuggingFaceHub with the token already set in the environment
llm =  HuggingFaceHub(repo_id="gpt2", huggingfacehub_api_token="Hf token", verbose = False)


# Define a prompt template to guide the model
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template='''You are a helpful assistant. Use the following context to answer the question. Keep the answer short, about 20 words.\n\n
             Context:\n{context}\n\n
             Question: {question}\n\n
             Answer concisely and accurately. If you do not know the answer just say "I do not know the answer."'''
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


file = open("vectorstore.pkl",'rb')
vector_store = pickle.load(file)
file.close()

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},  # Pass the prompt correctly in the chain configuration
)

def chatbot(query):
    # Use the RAG chain to generate a response with the custom prompt
    
    # method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use invoke instead.
    response = rag_chain.invoke({"question": query})
    return response

# Example interaction
user_query = "Can you tell me about Candolim beach"
response = chatbot(user_query)
print("ANSWEWR: \n\n")
print(response['answer'])
