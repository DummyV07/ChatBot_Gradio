from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import json

 
        
# 定义查询文本
query_text = "购买20万的办公用品，需要走什么流程"

# 加载预训练的 SentenceTransformer 模型
embedding_fn = SentenceTransformer('/home/cdjzxy/dev_llm/ChatBot_Gradio/bce-embedding-base_v1')

# 生成查询向量
query_vector = embedding_fn.encode(query_text, normalize_embeddings=True)

# 搜索 top-5 最相似的向量
try:
    results = client.search(
        collection_name="demo_collection",
        data=[query_vector],  # 确保传递的是一个包含向量的列表
        limit=1,
        output_fields=["id", "text"],  # 输出更多的字段以便查看结果
    )
except Exception as e:
    print(f"Search error: {e}")

# 输出结果
for result in results:
    for item in result:
        print(item)

"""
description:
    利用本地向量数据库搜索相似文本

input: query
output: query_result
"""
class searcher():
    def __init__(self):
        
        # 加载预训练的 SentenceTransformer 模型
        self.embedding_fn = SentenceTransformer('/home/cdjzxy/dev_llm/ChatBot_Gradio/bce-embedding-base_v1')
        # 初始化 Milvus 客户端
        self.client = MilvusClient("milvus_demo.db")

        # 查询特定条件的数据

        self.res = self.client.query(
            collection_name="demo_collection",
            filter="subject == 'history'",
            output_fields=["text", "subject"],
        )  

    def search(self,quary_text:str,top_k=5):
        # 生成查询向量
        query_vector = embedding_fn.encode(query_text, normalize_embeddings=True)