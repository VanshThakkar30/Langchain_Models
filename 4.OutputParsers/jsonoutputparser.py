from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

parser = JsonOutputParser()
model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="give me name age and city of a marvel fiction character \n {format_instructions} ",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})
print(result)