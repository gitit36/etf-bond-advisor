import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use 'gpt-3.5-turbo' if you don't have access to 'gpt-4'
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
    )
    print(response.choices[0].message['content'].strip())
except Exception as e:
    print("Error during OpenAI API call:")
    print(e)
