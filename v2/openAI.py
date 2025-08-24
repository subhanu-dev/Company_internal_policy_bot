import os
import dotenv
from openai import OpenAI
from pydantic import BaseModel


dotenv.load_dotenv()  # Loads .env variables into environment

openai_key = os.getenv("OPENAI_KEY")


client = OpenAI(api_key=openai_key)

response = client.responses.create(
    model="gpt-4.1-nano",
    instructions="talk like a loving woman",
    input="Write a short bedtime story about a unicorn.",
    max_output_tokens=200,
    stream=False,
)

print(response.output_text)
