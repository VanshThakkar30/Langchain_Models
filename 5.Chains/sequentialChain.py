from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

template1 = PromptTemplate(
    template="Write a detailed Report on {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a a5 line summary on the following text. {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

# print(result)
chain.get_graph().print_ascii()