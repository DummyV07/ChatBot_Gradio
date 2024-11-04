"""
description: 
    制作向量数据库
input:txt
output:db
"""
# 设置矢量数据库
# 要创建本地 Milvus 矢量数据库，只需MilvusClient通过指定一个文件名来实例化一个来存储所有数据，例如“milvus_demo.db”。

from pymilvus import MilvusClient,model

client = MilvusClient("milvus_demo.db")
from sentence_transformers import SentenceTransformer
from utils import save_chunks_to_files,read_text_file,split_text_by_semantic,get_sentence_embedding

class Database_create():
    def __init__(self,txt_path:str):
        self.txt_path = txt_path


    def cut_chunks(self):
        output_dir = "./data/chunks"
        # 读取长文本
        long_text = read_text_file(self.txt_path)
        # 设置每个文本块儿的最大分词数量和相似度阈值
        max_length = 1024  # 可根据需要调整
        similarity_threshold = 0.5  # 可根据需要调整
        # 分割长文本
        text_chunks = split_text_by_semantic(long_text, max_length, similarity_threshold)
        # 保存分割后的文本块到指定目录
        save_chunks_to_files(text_chunks, output_dir)
        print("保存路径{output_dir}")
        return text_chunks # 分割后的文本块儿列表
    
    def create_database(self):
        docs = self.cut_chunks()
        
        embedding_fn = SentenceTransformer('/home/cdjzxy/dev_llm/ChatBot_Gradio/bce-embedding-base_v1')
        # embedding_fn = model.DefaultEmbeddingFunction() # 默认的embeding模型 效果待评估 之前用过bge-4
        ectors = embedding_fn.encode_documents(docs)
        vectors = embedding_fn.encode(docs, normalize_embeddings=True)
        
        data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
        for i in range(len(vectors))
        ]
        client.insert(collection_name="milvus_demo", data=data)
        print("数据插入成功")

if __name__ == "__main__":
    creater = Database_create('/home/cdjzxy/dev_llm/ChatBot_Gradio/txt_files/成都交子新业科技发展有限公司服务和物资采购管理办法（试行）.txt')
    creater.create_database()