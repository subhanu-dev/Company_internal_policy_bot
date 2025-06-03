import gradio as gr
from main import get_rag_chain

rag_chain = get_rag_chain()


def chat_interface(query):
    result = rag_chain({"query": query})
    answer = result["result"]
    return answer


gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs="text",
    title="RAG Chatbot",
    description="Ask questions about your documents!",
).launch()
