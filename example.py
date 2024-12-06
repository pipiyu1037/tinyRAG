from rag import RAG
import requests

# Initialize the RAG model with your documents
api_key = '' # Your HKUST Azure OpenAI API key
api_version = '2024-06-01'
azure_endpoint = 'https://hkust.azure-api.net'

r = RAG('./Hamlet.txt', api_key, api_version, azure_endpoint)
oo = r.generate("who is HAMLET?")
print(oo)
