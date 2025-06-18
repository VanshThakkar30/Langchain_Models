from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

templat = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

with open('Prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = templat.invoke({
    'chat_history': chat_history,
    'query': 'What is the status of my order?'
})

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

result = model.invoke(prompt)
print(result.content)