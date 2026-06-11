from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")

client = OpenAI(base_url="https://openai.vocareum.com/v1")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Very concisely, which genes are associated with increased breast cancer risk?",
        },
    ],
)

print(response.choices[0].message.content)