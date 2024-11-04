
import re
import os
import torch
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel


def get_sentence_embedding(sentence, model, tokenizer):
    """
    获取句子的嵌入表示

    参数:
    sentence (str): 输入句子
    model (BertModel): 预训练的BERT模型
    tokenizer (BertTokenizer): BERT分词器

    返回:
    numpy.ndarray: 句子的嵌入向量
    """
    # 使用分词器处理输入句子，并转换为模型输入格式
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # 使用模型获取输出，不计算梯度
    with torch.no_grad():
        outputs = model(**inputs)
    # 返回最后一层隐藏状态的平均值作为句子嵌入
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def split_text_by_semantic(text, max_length, similarity_threshold=0.5):
    """
    基于语义相似度对文本进行分块

    参数:
    text (str): 输入的长文本
    max_length (int): 每个文本块的最大长度（以BERT分词器的token为单位）
    similarity_threshold (float): 语义相似度阈值，默认为0.5

    返回:
    list: 分割后的文本块列表
    """
    # 加载BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    model = BertModel.from_pretrained('./bert-base-chinese')
    model.eval()  # 设置模型为评估模式

    # 按句子分割文本（使用常见的中文标点符号）
    sentences = re.split(r'(。|！|？|；)', text)
    # 重新组合句子和标点
    sentences = [s + p for s, p in zip(sentences[::2], sentences[1::2]) if s]

    chunks = []
    current_chunk = sentences[0]
    # 获取当前chunk的嵌入表示
    current_embedding = get_sentence_embedding(current_chunk, model, tokenizer)

    for sentence in sentences[1:]:
        # 获取当前句子的嵌入表示
        sentence_embedding = get_sentence_embedding(sentence, model, tokenizer)
        # 计算当前chunk和当前句子的余弦相似度
        similarity = 1 - cosine(current_embedding, sentence_embedding)

        # 如果相似度高于阈值且合并后不超过最大长度，则合并
        if similarity > similarity_threshold and len(tokenizer.tokenize(current_chunk + sentence)) <= max_length:
            current_chunk += sentence
            # 更新当前chunk的嵌入表示
            current_embedding = (current_embedding + sentence_embedding) / 2
        else:
            # 否则，保存当前chunk并开始新的chunk
            chunks.append(current_chunk)
            current_chunk = sentence
            current_embedding = sentence_embedding

    # 添加最后一个chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def read_text_file(file_path):
    """
    读取文本文件

    参数:
    file_path (str): 文件路径

    返回:
    str: 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def save_chunks_to_files(chunks, output_dir):
    """
    将分割后的文本块保存到文件

    参数:
    chunks (list): 文本块列表
    output_dir (str): 输出目录路径
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将每个文本块保存为单独的文件
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(output_dir, f"chunk_{i + 1}.txt")
        with open(chunk_file_path, 'w', encoding='utf-8') as file:
            file.write(chunk)
        print(f"已保存第 {i + 1} 个文本块到 {chunk_file_path}")

if __name__ == "__main__":
    # 主程序(带遍历)
    import os

    file_path = './data'

    # 遍历指定目录及其子目录下所有文件的绝对路径。
    for root, dirs, files in os.walk(file_path):
        for file in files:
            # 设置输入和输出路径
            input_file_path = os.path.join(root, file)  # 替换为你的长文本文件路径
            # output_dir = './saveChunk/'  # 替换为你希望保存文本块的目录路径
            output_dir = './saveChunk/'+str(file)[:-4] + '/'  # 替换为你希望保存文本块的目录路径
            # 读取长文本
            long_text = read_text_file(input_file_path)

            # 设置每个文本块的最大分词数量和相似度阈值
            max_length = 1024  # 可根据需要调整
            similarity_threshold = 0.5  # 可根据需要调整

            # 分割长文本
            text_chunks = split_text_by_semantic(long_text, max_length, similarity_threshold)

            # 保存分割后的文本块到指定目录
            save_chunks_to_files(text_chunks, output_dir)
