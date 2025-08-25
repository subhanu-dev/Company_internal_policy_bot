import os
import dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel

dotenv.load_dotenv()  # Loads .env variables into environment

openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_key)


app = FastAPI()


class name_age(BaseModel):
    name: str
    age: int


@app.post("/openai_response")
def chat(request):
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions="you are an assistant",
        input=request,
        max_output_tokens=200,
        stream=False,
        response_format=name_age,
    )
    message_text = response["output"][0]["content"][0]["text"]
    # message_text = response.output_text # can also extract the response this way

    return {"response": message_text}
