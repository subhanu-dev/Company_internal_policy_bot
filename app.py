import gradio as gr
from main import get_rag_chain

rag_chain = get_rag_chain()


def chat_interface(message, history):
    # Format history for the prompt
    chat_history = ""
    if history:
        for turn in history:
            if turn["role"] == "user":
                chat_history += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                chat_history += f"Assistant: {turn['content']}\n"
    # Call the RAG chain with chat history
    result = rag_chain.invoke({"question": message, "chat_history": chat_history})
    answer = result["answer"] if "answer" in result else result["result"]
    # Append the new user and assistant messages in OpenAI format
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, history


with gr.Blocks() as demo:
    gr.Markdown(
        "# RAG Chatbot\nAsk questions about your documents. Chat history is preserved below."
    )
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Your question")
    clear = gr.Button("Clear chat")

    def clear_fn():
        return [], []

    msg.submit(chat_interface, [msg, chatbot], [chatbot, chatbot])
    clear.click(clear_fn, [], [chatbot, chatbot])

demo.launch()
