from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=1.5,max_completion_tokens=50)

result = model.invoke("give me 5 lines of a poem about the cricket game")

print(result.content)