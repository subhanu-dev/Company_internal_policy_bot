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

    # Append the new user and assistant messages
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, history, ""


with gr.Blocks(
    theme="soft",
    css="""
    body, .gradio-container {
        background: linear-gradient(rgba(0,0,0.9), rgba(0,255,255, 0.6)), url('https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=1500&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    """,
) as demo:
    gr.Markdown("<div style='height: 20px'></div>")

    gr.Markdown(
        "# Internal Company Policy bot (RAG) 💬 \n Ask questions about the company policies, procedures and get answers based on internal docs.\n"
        "\n This bot can answer question about policies regarding HR, training and development, codes of conduct, IT support,tooling guides and more."
    )

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Your question")
    with gr.Row():
        with gr.Column(scale=1):
            clear = gr.Button("Clear chat", variant="secondary")
        with gr.Column(scale=9):
            gr.Markdown("")  # Empty column to push the button to the left (or right)

    def clear_fn():
        return [], [], ""

    msg.submit(chat_interface, [msg, chatbot], [chatbot, chatbot, msg])
    clear.click(clear_fn, [], [chatbot, chatbot, msg])

demo.launch()
