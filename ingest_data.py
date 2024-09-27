import argparse
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Build vector store for a LLM')
parser.add_argument('text', nargs='*')  # list of text files to be passed as input
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--output', default='vectorstore.pkl', help='Output vector store filename')
parser.add_argument('--chunk_size', default=500, type=int, help='Chunk size')
parser.add_argument('--chunk_overlap', default=20, type=int, help='Chunk overlap')
args = parser.parse_args()

# Load Data
docs = args.text
all_docs = []

# Loop through the documents passed as input
for doc in docs:
    if args.verbose:
        print(f"Loading document {doc}")
    
    # Specify the 'utf-8' encoding to handle non-ASCII characters
    loader = TextLoader(doc, encoding='utf-8')  # Ensure doc is defined here
    all_docs.extend(loader.load())  # Load and append document contents

# Split text into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=args.chunk_size, 
    separator="\n\n", 
    is_separator_regex=False, 
    length_function=len, 
    chunk_overlap=args.chunk_overlap
)
documents = text_splitter.split_documents(all_docs)
print(documents[0])
# Create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if args.verbose:
    print("Embeddings created")

# Load Data to vectorstore
vectorstore = FAISS.from_documents(documents, embeddings)

# Save vectorstore
if args.verbose:
    print(f"Saving pickle file as {args.output}")
with open(args.output, "wb") as f:
    pickle.dump(vectorstore, f)
