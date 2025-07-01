from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence  # | operator
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()  

prompt = PromptTemplate(
    template="generate sort and simple note on following topic \n {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt,model,parser)

result = chain.invoke({
    'text': """1.4. Support Vector Machines"""
})
print(result)