"""
description: 
    为了保持章节内容的完整性，选择了按章节切分的方法但是
    由于embedding模型对文本长度有要求，chunk_size过大进不去embedding模型
    所以要针对超长章节再进行切分，这时切分的每一个small_chunk中需要标注章节内容
input:txt
output:chunk_txts
"""

class chunk_cutter():   
    def __init__(self,txt_path:str,max_chunk_len=512):
        self.txt_path = txt_path
        self.max_chunk_len = max_chunk_len

    def cut_chunk(self):
        with open(self.txt_path,'r') as f:
            txt = f.read()
        
