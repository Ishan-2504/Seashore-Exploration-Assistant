import argparse
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import pyttsx3

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
engine = pyttsx3.init()

# Load vector store
with open("vectorstore.pkl", 'rb') as file:
    vector_store = pickle.load(file)

# Create retriever from the vector store
retriever = vector_store.as_retriever()

# Define Hugging Face model with HuggingFaceHub
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-1B",  # Replace with your model if needed
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 128,
        "top_k": 5,
        "temperature": 0.2,
        "repetition_penalty": 1.03,
        "stream": False,  # Set stream to False for synchronous output
    },
    huggingfacehub_api_token="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",  # Replace with your token
)



# Define prompt template for the retrieval
template = """
You are a conversational and engaging assistant. Engage in a thoughtful dialogue with the user, responding naturally to maintain a flowing conversation.  However, answer factual questions strictly based on the information provided in the 'context.' If the answer isn't in the context, respond with: 'I don’t have that information right now, but feel free to ask something else!'

    {context}

    Question: {question}
"""

# Create a chat prompt from the template
prompt = ChatPromptTemplate.from_template(template)

# Create the chain that processes the question
chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Define function to extract the answer from the response
def extract_answer(response):
    if "Answer:" in response:
        return response.split("Answer:")[1].strip() 
    return response.strip()

# Example user message
user_message = "Do you know where NSUT in Delhi is?"
bot_message = chain.invoke(user_message)
print(bot_message)

print(extract_answer(bot_message))
# engine.say(extract_answer(bot_message))
# engine.runAndWait()

