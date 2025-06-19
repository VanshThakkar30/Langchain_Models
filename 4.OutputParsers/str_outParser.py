from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed Report on {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a a5 line summary on the following text. {text}",
    input_variables=['text']
)
# in place of this all we will use output parsers
# prompt1 = template1.invoke({'topic':'black hole'})

# result = model.invoke(prompt1)

# prompt2 = template2.invoke({'text': result.content})

# result2 = model.invoke(prompt2)

# print(result.content)
# print(result2.content)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)