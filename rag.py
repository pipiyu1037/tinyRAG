from openai import AzureOpenAI
from database import Database

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Context: {context}
Question: {question}
Answer:
"""

class RAG:
    def __init__(self, resource, api_key, api_version, azure_endpoint):
        self.database = Database(resource)
        self.client = AzureOpenAI(api_key = api_key, api_version = api_version, azure_endpoint = azure_endpoint)
        self.retriever = self.database.vectorstore.as_retriever(search_kwargs={"k": 1})
        
    
    def retrieve(self, query):
        retrieved_docs = self.retriever.invoke(query)
        return retrieved_docs
    
    def get_response(client, prompt):
        pass
    
    def generate(self, query):
        context = self.retrieve(query)[0].page_content
        prompt = template.format(context=context, question=query)
        print(prompt)
        response = self.client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content
        return answer
    
    def query(self, query):
        pass