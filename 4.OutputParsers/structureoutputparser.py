from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1", description="fact 1 about the topic"),
    ResponseSchema(name="fact2", description="fact 2 about the topic"),
    ResponseSchema(name="fact3", description="fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template="Write 3 facts about {topic} \n {format_instructions} ",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'Marvels Doomsday'})
print(result)