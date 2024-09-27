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



file = open("vectorstore.pkl",'rb')
vector_store = pickle.load(file)
file.close()


retriever = vector_store.as_retriever()


prompt_search_query = ChatPromptTemplate.from_messages([
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}"),
("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

prompt_get_answer = ChatPromptTemplate.from_messages([
("system", "If you do not know the answer just say 'I do not know'. Do not make up your own answer.Answer the user's questions based on the below context:\\n\\n{context}"),
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}"),
])

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain=create_stuff_documents_chain(llm,prompt_get_answer)
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="Am i going to fail my exams"), AIMessage(content="Yes")]
response = retrieval_chain.invoke({
"chat_history":chat_history,
"input":"how",
})
print (response['answer'])