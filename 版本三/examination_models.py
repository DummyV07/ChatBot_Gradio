

"""
desription:
    输入query返回top_k个相似度最高的chunk
input:
    query: str
output:
    embedding: list
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class examination_models():
    def __init__(self,model_path:str,test_doc_path:str):
        self.model = SentenceTransformer(model_path)
        self.docments = [line for line in open(test_doc_path,'r')]
    def get_embedding(self,input):
        return self.model.encode(input,normalize_embeddings=True)
    
    def get_similarity(self,query:str):
        query_embedding = self.get_embedding(query)
        return [self.get_embedding(doc).dot(query_embedding) for doc in self.docments]
    def get_similar_doc(self,query:str,top_k=5):
        similarity_scores = self.get_similarity(query)
        return [self.docments[i] for i in np.argsort(similarity_scores)[::-1][:top_k]]
if __name__ == '__main__':
    model_path = '/home/cdjzxy/dev_llm/ChatBot_Gradio/bce-embedding-base_v1'
    test_doc_path = '/home/cdjzxy/dev_llm/ChatBot_Gradio/data/chapter4.txt'
    exam_model = examination_models(model_path,test_doc_path)
    for sentence in exam_model.get_similar_doc('我想采买20万的法律服务，应该走什么流程',3):
        print(sentence)
        print('\n')