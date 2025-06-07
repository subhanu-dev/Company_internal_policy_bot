from dotenv import load_dotenv

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Load documents from the 'documents' folder
def load_documents(directory="documents"):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") or filename.endswith(".md"):
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
        temperature=0.4,
    )
    return llm


# Custom prompt template with history
custom_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "You are an assistant on company internal policies."
        "If the user's question is not about company policy, or is just a greeting or thanks, respond politely and briefly without referencing policy documents without referencing policy documents or repeating previous answers. "
        " As An internal knowledge bot. Use the provided context and recent conversation history to answer the user's question concisely. "
        "Conversation History:\n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
)


# Create Conversational RAG chain
def create_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # Initialize memory with a rolling window (e.g., last 5 exchanges)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=False,
    )
    return qa_chain


# Load or create vector store
def load_or_create_vector_store(chunks, embeddings, index_path="faiss_index"):
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
        return vector_store


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No documents found in 'documents' directory. Add .txt files and rerun.")
    else:
        chunks = split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = load_or_create_vector_store(chunks, embeddings)
        llm = setup_llm()
        rag_chain = create_rag_chain(vector_store, llm)

        # Interactive loop
        print("RAG Chatbot ready! Type 'exit' to quit.")
        while True:
            query = input("Enter your question: ")
            if query.lower() == "exit":
                break
            result = rag_chain.invoke({"question": query})
            print("Answer:", result["answer"])


def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index_path = "faiss_index"
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        docs = load_documents()
        if not docs:
            raise RuntimeError(
                "No documents found in 'documents' directory. Add .txt or .md files and rerun."
            )
        chunks = split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
    llm = setup_llm()
    rag_chain = create_rag_chain(vector_store, llm)
    return rag_chain
