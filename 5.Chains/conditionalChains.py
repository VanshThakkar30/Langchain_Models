from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="sentiment of the feedback")

pyparser = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template="classify the sentiment of the following feedback: {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': pyparser.get_format_instructions()}
)

classifierChain = prompt1 | model | pyparser

prompt2 = PromptTemplate(
    template="write a simple & appropriate response to this positive feedback: {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="write a simple & appropriate response to this negative feedback: {feedback}",
    input_variables=['feedback']
)

branchChain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: f"Sentiment not recognized:")
)

chain = classifierChain | branchChain


# print(chian.invoke({'feedback': 'this product is neutral'}))
chain.get_graph().print_ascii()
