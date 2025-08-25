import os
import dotenv
from openai import OpenAI
from pydantic import BaseModel
from fastapi import FastAPI

dotenv.load_dotenv()  # Loads .env variables into environment

openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_key)


class ChatResponse(BaseModel):
    response: str


class ChatRequest(BaseModel):
    session_id: str
    input: str


app = FastAPI()


@app.post("/openai_response", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = client.responses.create(
        model="gpt-4.1-nano",
        instructions="talk like a loving woman",
        input=request.input,
        max_output_tokens=200,
        stream=False,
    )
    # message_text = response["output"][0]["content"][0]["text"]
    # message_text = response.output_text  # can also extract the response this way

    message_text = response.output_text
    return ChatResponse(response=message_text)
