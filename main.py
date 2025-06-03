from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Load documents from the 'documents' folder
def load_documents(directory="documents"):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    return documents


# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Small for CPU efficiency
        # chunk_size=500 using LangChain’s RecursiveCharacterTextSplitter, meaning each chunk is up to 500 characters long. */
        chunk_overlap=50,  # Overlap for context
        # Chunk overlap is the number of characters (or tokens) shared between consecutive chunks. In your script, we set chunk_overlap=50, meaning each chunk overlaps with the next by 50 characters.
    )

    return text_splitter.split_documents(documents)


# Create FAISS vector store
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")  # Save to disk for reuse
    return vector_store


# Set up Groq LLM
def setup_llm():
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,  # For consistent answers
    )
    return llm


# Create RAG chain
def create_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Get top 3 chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Combines retrieved chunks into prompt
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


# Update main to include interactive loop
if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No documents found in 'documents' directory. Add .txt files and rerun.")
    else:
        chunks = split_documents(docs)
        print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks.")
        print("Sample chunk:", chunks[0].page_content[:100])
        vector_store = create_vector_store(chunks)
        print("FAISS vector store created with", len(chunks), "chunks.")
        llm = setup_llm()
        rag_chain = create_rag_chain(vector_store, llm)
        # Interactive loop
        print("RAG Chatbot ready! Type 'exit' to quit.")
        while True:
            query = input("Enter your question: ")
            if query.lower() == "exit":
                break
            result = rag_chain({"query": query})
            print("Answer:", result["result"])
            print("Sources:", [doc.metadata for doc in result["source_documents"]])
            print()
