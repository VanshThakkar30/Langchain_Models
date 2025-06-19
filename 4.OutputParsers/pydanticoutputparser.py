from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(description="age of the person")
    city: str = Field(description="name of the city where the person belongs to")

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template="give me name age and city of a marvel fiction {place}character \n {format_instructions} ",
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'place': 'indian'})

print(result)