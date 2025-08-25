from groq import Groq
import os
from dotenv import load_dotenv
from search import search_similar_chunks  # Import your search function
import re

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api)


# greeting detection pattern
greeting_pattern = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|kohomada|greetings|hola)(\s|$|[!?.])",
    re.IGNORECASE,
)

# Pre-defined greeting responses
greeting_responses = [
    "Hello! I'm your HR assistant. How can I help you today?",
    "Hi there! Feel free to ask me any questions about company policies.",
    "Greetings! I'm here to assist with HR-related inquiries. What would you like to know?",
    "Hello! I can help you with information about our company policies. What would you like to know?",
]


def rag_chat(user_query):
    # Check if the input is a simple greeting
    if greeting_pattern.match(user_query.strip()):
        # Return a random greeting response
        import random

        return random.choice(greeting_responses)

    # Step 1: Retrieve relevant chunks from vector database
    search_results = search_similar_chunks(user_query)

    # Step 2: Format retrieved context
    if search_results:
        context_chunks = [result["chunk"] for result in search_results]
        context = "\n\n".join(context_chunks)
        print("context")

        # Add similarity scores for debugging
        debug_info = "\n".join(
            [f"[Score: {result['score']:.4f}]" for result in search_results]
        )
        print(f"Found {len(search_results)} relevant chunks. {debug_info}")

    else:
        context = ""
        print("No relevant information found in the knowledge base.")

    # Step 3: Create RAG prompt
    if context:
        system_prompt = """You are a helpful assistant that is specialized in Company HR policy which answers questions based on the provided context.
Base your answers only on the context provided. If the context doesn't contain relevant information,
say "I don't have enough information to answer this question." Do not make up information."""

        user_prompt = f"""Context:{context}

        Question: {user_query}

Please answer the question based only on the context provided above."""

    else:
        system_prompt = """You are a polite HR assistant. 
If the query is a greeting, respond in a friendly and concise manner. 
If the query asks about HR policies or information not found in the knowledge base, 
respond with: "I don't have enough information in the knowledge base to answer that, 
but you can reach out to HR for more details."""

        user_prompt = f"Question: {user_query}\nPlease note that I don't have specific information about this in my knowledge base."

    # Step 4: Generate response with Groq
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="openai/gpt-oss-20b",
        temperature=0.6,
        max_completion_tokens=2000,
    )

    return response.choices[0].message.content


while True:
    query = input("\nYour question: ")
    if query.lower() in ("exit", "quit", "bye"):
        break

    answer = rag_chat(query)
    print("\nAnswer:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
