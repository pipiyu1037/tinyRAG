import openai
import faiss
import chromadb
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # 修正导入
import numpy as np
import re

# OpenAI Configuration
openai.api_type = "azure"
openai.api_base = "https://hkust.azure-api.net"
openai.api_version = "2023-05-15"
openai.api_key = "1b09b2d1b9754ca5b0910328f6fea9a0"


class AzureEmbeddingFunction:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model

    def embed_documents(self, texts):
        response = openai.Embedding.create(engine=self.model, input=texts)
        return [item["embedding"] for item in response["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class HuggingFaceEmbeddingFunction:
    def __init__(self, model_name="shibing624/text2vec-base-chinese"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts):
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)


class RAGWithHybrid:
    def __init__(self, resource, azure_model="text-embedding-ada-002", chinese_model="shibing624/text2vec-base-chinese"):
        # Load documents
        loader = TextLoader(resource)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(documents)

        # BM25 initialization
        tokenized_corpus = [doc.page_content.split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Azure embedding + Chroma
        azure_embedding_function = AzureEmbeddingFunction(azure_model)
        self.azure_chroma_client = chromadb.Client()
        self.azure_vectorstore = Chroma.from_documents(
            documents=self.texts,
            embedding=azure_embedding_function,
            client=self.azure_chroma_client
        )

        # Azure embedding + HNSW
        azure_embeddings = azure_embedding_function.embed_documents(
            [doc.page_content for doc in self.texts]
        )
        self.azure_hnsw_index = faiss.IndexHNSWFlat(len(azure_embeddings[0]), 32)
        self.azure_hnsw_index.hnsw.efConstruction = 200
        self.azure_hnsw_index.add(np.array(azure_embeddings, dtype="float32"))

        # Chinese embedding + HNSW
        chinese_embedding_function = HuggingFaceEmbeddingFunction(model_name=chinese_model)
        chinese_embeddings = chinese_embedding_function.embed_documents(
            [doc.page_content for doc in self.texts]
        )
        self.chinese_hnsw_index = faiss.IndexHNSWFlat(len(chinese_embeddings[0]), 32)
        self.chinese_hnsw_index.hnsw.efConstruction = 200
        self.chinese_hnsw_index.add(np.array(chinese_embeddings, dtype="float32"))
        self.chinese_docs = [doc.page_content for doc in self.texts]

    def retrieve_with_bm25(self, query, k=3):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.texts[i].page_content for i in top_k_indices]
        
    def retrieve_with_chroma(self, query, k=3, embedding_type="azure"):
        try:
            if embedding_type == "azure":
                retriever = self.azure_vectorstore.as_retriever(search_kwargs={"k": k})
            else:
                retriever = self.azure_vectorstore.as_retriever(search_kwargs={"k": k})
            results = retriever.invoke(query)
            if not results:
                print(f"[Warning] Chroma retrieval returned empty results for query: {query}")
            return results or ["No relevant results found."]
        except Exception as e:
            print(f"[Error] Error during Chroma retrieval: {e}")
            return ["An error occurred during Chroma retrieval."]
        
    def retrieve_with_hnsw(self, query, k=3, embedding_type="azure"):
        if embedding_type == "azure":
            query_vector = np.array(AzureEmbeddingFunction().embed_query(query), dtype="float32").reshape(1, -1)
            distances, indices = self.azure_hnsw_index.search(query_vector, k)
            return [self.texts[i].page_content for i in indices[0] if i < len(self.texts)]
        else:
            query_vector = np.array(HuggingFaceEmbeddingFunction().embed_query(query), dtype="float32").reshape(1, -1)
            distances, indices = self.chinese_hnsw_index.search(query_vector, k)
            return [self.chinese_docs[i] for i in indices[0] if i < len(self.chinese_docs)]
    
    def hybrid_retrieve(self, query, k=3, embedding_type="chroma"):
        bm25_results = self.retrieve_with_bm25(query, k=k * 2)

        if embedding_type == "chroma":
            embedding_results = self.retrieve_with_chroma(query, k)
        elif embedding_type == "hnsw":
            embedding_results = self.retrieve_with_hnsw(query, k, embedding_type)
        else:
            raise ValueError(f"Unknown hybrid method: {embedding_type}")

        # 提取文档内容（确保为字符串）
        bm25_results = [doc if isinstance(doc, str) else doc.page_content for doc in bm25_results]
        embedding_results = [doc if isinstance(doc, str) else doc.page_content for doc in embedding_results]

        # 合并并去重
        combined_results = list(dict.fromkeys(bm25_results + embedding_results))
        return combined_results[:k]

    def rerank_with_openai(self, query, documents):
        if not documents:
            print(f"[Warning] No documents to rerank for query: {query}")
            return []

        documents_text = "".join([f"Document {i+1}: {doc}\n" for i, doc in enumerate(documents)])
        prompt = (
            f"Query: {query}\n"
            "Documents:\n"
            f"{documents_text}\n"
            "Please rank the documents by relevance."
        )

        messages = [
            {"role": "system", "content": "You are a document ranking assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.0,
            )
            ranking_output = response['choices'][0]['message']['content']

            # 提取排名中的数字索引
            ranked_indices = re.findall(r'\d+', ranking_output)
            ranked_indices = [int(idx) - 1 for idx in ranked_indices if idx.isdigit()]
            ranked_indices = [i for i in ranked_indices if i < len(documents)]

            # 如果解析失败，返回原顺序
            if not ranked_indices:
                print(f"[Warning] No valid ranking indices found. Returning documents in original order.")
                return documents

            return [documents[i] for i in ranked_indices]
        except Exception as e:
            print(f"[Error] Error during reranking: {e}")
            return documents
        
    def generate(self, query, method="bm25", rank=False, k=3, embedding_type="azure"):
        if method == "bm25":
            retrieved_docs = self.retrieve_with_bm25(query, k)
        elif method == "chroma":
            retrieved_docs = self.retrieve_with_chroma(query, k, embedding_type)
        elif method == "hnsw":
            retrieved_docs = self.retrieve_with_hnsw(query, k, embedding_type)
        elif method == "chroma_bm25":
            retrieved_docs = self.hybrid_retrieve(query, k, embedding_type="chroma")
        elif method == "hnsw_bm25":
            retrieved_docs = self.hybrid_retrieve(query, k, embedding_type="hnsw")
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        if not retrieved_docs:
            print(f"[Warning] No documents retrieved for query: {query}, method: {method}")
            return "No relevant documents were found for this query."

        if rank:
            retrieved_docs = self.rerank_with_openai(query, retrieved_docs)
            if not retrieved_docs:
                print(f"[Warning] Reranking returned no results for query: {query}")
                return "Reranking did not yield any results."

        # 提取文档内容
        if isinstance(retrieved_docs[0], str):
            context = " ".join(retrieved_docs)
        else:
            context = " ".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        You are an assistant for question-answering tasks.
        Use the following retrieved context to answer the question:
        Context: {context}
        Question: {query}
        Answer:
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.9,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"[Error] Error during generation: {e}")
            return "An error occurred during generation."

if __name__ == "__main__":
    resource_path = "./leiyu.txt"
    methods = [
        "bm25",
        "chroma",
        "hnsw",
        "chroma_bm25",
        "hnsw_bm25"
    ]
    ranks = [False, True]
    embedding_types = ["azure", "chinese"]

    print("Starting experiments...")

    for method in methods:
        for rank in ranks:
            if method == "bm25":  # BM25 与 embedding_type 无关
                print(f"\nMethod: {method.upper()}, LLM Rank: {rank}")
                try:
                    rag = RAGWithHybrid(resource_path)
                    query = "请评价周朴园的性格。"
                    answer = rag.generate(query, method=method, rank=rank, k=3)
                    print(f"Answer:\n{answer}")
                except Exception as e:
                    print(f"[Error] An error occurred for method: {method}, rank: {rank}. Error: {e}")
            else:  # 其他方法需要区分 embedding_type
                for embedding_type in embedding_types:
                    print(f"\nEmbedding Type: {embedding_type.upper()}, Method: {method.upper()}, LLM Rank: {rank}")
                    try:
                        rag = RAGWithHybrid(resource_path)
                        query = "请评价周朴园的性格。"
                        answer = rag.generate(query, method=method, rank=rank, k=3, embedding_type=embedding_type)
                        print(f"Answer:\n{answer}")
                    except Exception as e:
                        print(f"[Error] An error occurred for embedding type: {embedding_type}, method: {method}, rank: {rank}. Error: {e}")