from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=64,
)
documents = [
    "Delhi is the capital of India.",
    "The capital of France is Paris.",
    "Tokyo is the capital of Japan.",
]

result = embeddings.embed_documents(documents)

print(str(result))