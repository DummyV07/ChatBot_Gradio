# 启动ollama 服务 使用端口访问

#ollama本身也可以的
from config import * # 导入配置文件
from openai import OpenAI
from loguru import logger

class MyQwen():
    def __init__(self):
        
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1', # ollama 默认端口
            api_key='---', # openai需要填的，但填什么都可以，不会进行验证
        )
 
    def fet_completion(self,
            messages,
            model=DEFAULT_MODEL,
            max_tokens=200,
            temperature=0,
            stream=False,):
        """
        :param messages: 到目前为止，构成对话的消息列表。
        :param model: 要使用的模型的 ID。
        :param max_tokens: 聊天完成时可以生成的最大令牌数。
        :param temperature: 使用什么采样温度，介于 0 和 2 之间。
        较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更具集中性和确定性。
        :param stream: 是否流式输出。
        :return: chat completion object（聊天完成对象），
        如果请求是流式处理的，则返回chat completion chunk（聊天完成区块对象）的流序列。
        """
        # 输入验证
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            return "无效的 'messages' 类型。它应该是一个字符串或消息列表。"

        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
        )

        if stream:
            # 流式输出
            return response

        # 非流式输出
        logger.debug(response.choices[0].message.content)
        logger.info(f"总token数: {response.usage.total_tokens}")
        
        return response.choices[0].message.content

    def get_completion(
            self,
            messages,
            model=DEFAULT_MODEL,
            max_tokens=200,
            temperature=0,
            stream=False,
    ):
        """
        Creates a model response for the given chat conversation.
        为给定的聊天对话创建模型响应。

        API官方文档：https://platform.openai.com/docs/api-reference/chat/create

        :param messages: 到目前为止，构成对话的消息列表。
        :param model: 要使用的模型的 ID。
        :param max_tokens: 聊天完成时可以生成的最大令牌数。
        :param temperature: 使用什么采样温度，介于 0 和 2 之间。
        较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更具集中性和确定性。
        :param stream: 是否流式输出。
        :return: chat completion object（聊天完成对象），
        如果请求是流式处理的，则返回chat completion chunk（聊天完成区块对象）的流序列。
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            return "无效的 'messages' 类型。它应该是一个字符串或消息列表。"

        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
        )

        if stream:
            # 流式输出
            return response

        # 非流式输出
        logger.debug(response.choices[0].message.content)
        logger.info(f"总token数: {response.usage.total_tokens}")
        return response.choices[0].message.content

    def get_embedding(self, input):
        """
        Creates an embedding vector representing the input text.
        创建表示输入文本的嵌入向量。

        API官方文档：https://platform.openai.com/docs/api-reference/embeddings/create

        :param input: 输入要嵌入的文本，编码为字符串或标记数组。若要在单个请求中嵌入多个输入，请传递字符串数组或令牌数组数组。
        输入不得超过模型的最大输入标记数（8192 text-embedding-ada-002 个标记），不能为空字符串，任何数组的维数必须小于或等于 2048。
        :return: 嵌入对象的列表。
        """
        response = self.client.embeddings.create(
            input=input,
            model='text-embedding-ada-002',
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

if __name__ == "__main__":
    # 测试
    llm = MyQwen()

    # prompt
    prompt = '你好'
    response = llm.get_completion(prompt, temperature=1)
    print(response)