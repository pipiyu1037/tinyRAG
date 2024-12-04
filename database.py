from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
#from weaviate.classes.init import Auth
import chromadb

class Database:
    def __init__(self, resource):
        loader = TextLoader(resource)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])
        texts = text_splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(model="llama3")
        persistent_client = chromadb.PersistentClient()
        # collection = persistent_client.get_or_create_collection(name="rag", embedding_function=embeddings)
        # ids = [str(i) for i in range(len(texts))]
        # collection.add(ids=ids, documents=texts)

        self.vectorstore = Chroma.from_documents(
            documents=texts, 
            embedding=embeddings, 
            client=persistent_client
        )