# 探索不同组合的RAG方法在小说人物性格总结任务上的效果
```
!pip install openai
!pip install faiss-gpu  # 或 faiss-cpu
!pip install chromadb
!pip install rank-bm25
!pip install langchain
!pip install text2vec
!pip install numpy pandas tqdm
!pip install openai==0.28.1

! pip install langchain_community
! pip install langchain_chroma

!pip install langchain_huggingface
```

| **实验编号** | **嵌入模型**        | **检索方式**               | **是否使用 LLM Rank** |
|--------------|---------------------|----------------------------|-----------------------|
| 1            | 无                 | BM25                       | 否                    |
| 2            | 无                 | BM25                       | 是                    |
| 3            | Azure Embedding    | Embedding (Chroma)         | 否                    |
| 4            | Azure Embedding    | Embedding (Chroma)         | 是                    |
| 5            | Azure Embedding    | Embedding (HNSW)           | 否                    |
| 6            | Azure Embedding    | Embedding (HNSW)           | 是                    |
| 7            | Azure Embedding    | Hybrid (BM25 + Chroma)     | 否                    |
| 8            | Azure Embedding    | Hybrid (BM25 + Chroma)     | 是                    |
| 9            | Azure Embedding    | Hybrid (BM25 + HNSW)       | 否                    |
| 10           | Azure Embedding    | Hybrid (BM25 + HNSW)       | 是                    |
| 11           | Chinese Embedding  | Embedding (Chroma)         | 否                    |
| 12           | Chinese Embedding  | Embedding (Chroma)         | 是                    |
| 13           | Chinese Embedding  | Embedding (HNSW)           | 否                    |
| 14           | Chinese Embedding  | Embedding (HNSW)           | 是                    |
| 15           | Chinese Embedding  | Hybrid (BM25 + Chroma)     | 否                    |
| 16           | Chinese Embedding  | Hybrid (BM25 + Chroma)     | 是                    |
| 17           | Chinese Embedding  | Hybrid (BM25 + HNSW)       | 否                    |
| 18           | Chinese Embedding  | Hybrid (BM25 + HNSW)       | 是                    |