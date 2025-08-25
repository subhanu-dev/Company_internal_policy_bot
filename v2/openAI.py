import os
import dotenv
from openai import OpenAI
from fastapi import FastAPI

dotenv.load_dotenv()  # Loads .env variables into environment

openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_key)


app = FastAPI()


@app.post("/openai_response")
def chat(request):
    response = client.responses.create(
        model="gpt-4.1-nano",
        instructions="talk like a loving woman",
        input=request,
        max_output_tokens=200,
        stream=False,
    )
    message_text = response["output"][0]["content"][0]["text"]
    # message_text = response.output_text # can also extract the response this way

    return {"response": message_text}
