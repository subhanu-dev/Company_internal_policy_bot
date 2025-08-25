import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from fastapi import FastAPI

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_key)
app = FastAPI()

# In-memory conversation store: session_id -> list of messages
conversations = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id
    user_message = {"role": "user", "content": req.message}

    # Initialize conversation if first message
    if session_id not in conversations:
        conversations[session_id] = []

    # Append user's message to history
    conversations[session_id].append(user_message)

    # Call Groq API with full message history
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=conversations[session_id],
        temperature=2,
        max_completion_tokens=1000,
        top_p=1,
        stream=False,
    )

    # Extract assistant's reply
    assistant_reply = completion.choices[0].message.content

    # Append assistant's reply to history
    conversations[session_id].append({"role": "assistant", "content": assistant_reply})
    print(len(conversations["fvrwg"]))

    return ChatResponse(reply=assistant_reply)
