# 58xueke.com
import gradio as gr
from loguru import logger
from qwen import MyQwen
from config import MODELS, DEFAULT_MODEL, MODEL_TO_MAX_TOKENS


myqwen = MyQwen()


def fn_prehandle_user_input(user_input, chat_history):
    # 检查输入
    if not user_input:
        gr.Warning("请输入您的问题")
        logger.warning("请输入您的问题")
        return chat_history

    # 用户消息在前端对话框展示
    chat_history.append([user_input, None])

    logger.info(f"\n用户输入: {user_input}, \n"
                f"历史记录: {chat_history}")
    return chat_history


def fn_predict(
        user_input,
        chat_history,
        model,
        max_tokens,
        temperature,
        stream):

    # 如果用户输入为空，则返回当前的聊天历史
    if not user_input:
        return chat_history

    # 打印日志，记录输入参数信息
    logger.info(f"\n用户输入: {user_input}, \n"
                f"历史记录: {chat_history}, \n"
                f"使用模型: {model}, \n"
                f"要生成的最大token数: {max_tokens}\n"
                f"温度: {temperature}\n"
                f"是否流式输出: {stream}")

    # 构建 messages 参数
    messages = user_input  # or [{"role": "user", "content": user_input}]
    if len(chat_history) > 1:
        messages = []
        for chat in chat_history:
            if chat[0] is not None:
                messages.append({"role": "user", "content": chat[0]})
            if chat[1] is not None:
                messages.append({"role": "assistant", "content": chat[1]})
    print(messages)

    # 生成回复
    bot_response = myqwen.get_completion(
        messages, model, max_tokens, temperature, stream)

    if stream:
        # 流式输出
        chat_history[-1][1] = ""
        for character in bot_response:
            character_content = character.choices[0].delta.content
            if character_content is not None:
                chat_history[-1][1] += character_content
                yield chat_history
    else:
        # 非流式输出
        chat_history[-1][1] = bot_response
        logger.info(f"历史记录: {chat_history}")
        yield chat_history

    # return chat_history

    


def fn_update_max_tokens(model, origin_set_tokens):
    """
    更新最大令牌数的函数。

    :param model: 要更新最大令牌数的模型。
    :param origin_set_tokens: 原始滑块组件设置的令牌数。
    :return: 包含新最大令牌数的滑块组件。
    """
    # 获取模型对应的新最大令牌数，如果没有设置则使用传入的最大令牌数
    new_max_tokens = MODEL_TO_MAX_TOKENS.get(model)
    new_max_tokens = new_max_tokens if new_max_tokens else origin_set_tokens

    # 如果原始设置的令牌数超过了新的最大令牌数，将其调整为默认值（这里设置为500，你可以根据需要调整）
    new_set_tokens = origin_set_tokens if origin_set_tokens <= new_max_tokens else 500

    # 创建新的最大令牌数滑块组件
    new_max_tokens_component = gr.Slider(
        minimum=0,
        maximum=new_max_tokens,
        value=new_set_tokens,
        step=1.0,
        label="max_tokens",
        interactive=True,
    )

    return new_max_tokens_component


with gr.Blocks() as demo:
    # 标题
    gr.Markdown("# MyQwen")
    with gr.Row(equal_height=True):
        # 左侧对话栏
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="聊天机器人")
            user_input_textbox = gr.Textbox(label="用户输入框", value="你好")
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear", elem_id="btn")
        # 右侧工具箱
        with gr.Column(scale=1):
            with gr.Row():
                input_file = gr.File(label="上传文件")
            # 创建一个包含三个滑块的选项卡，用于调整模型的温度、最大长度和Top P参数
            with gr.Tab(label="LLM参数"):
                # 选择模型
                model_dropdown = gr.Dropdown(
                    label="model",
                    choices=MODELS,
                    value=DEFAULT_MODEL,
                    multiselect=False,
                    interactive=True,
                )
                max_tokens_slider = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=500,
                    step=1.0,
                    label="max_tokens",
                    interactive=True)
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label="temperature",
                    interactive=True)
                stream_radio = gr.Radio(
                    choices=[
                        True,
                        False],
                    label="stream",
                    value=True,
                    interactive=True)
            with gr.Tab("数据库参数"): # 
                # 选择检索模式
                model_dropdown = gr.Dropdown(
                    label="search",
                    choices=['稀疏', '密集', '混合'],
                    multiselect=False,
                    interactive=True,
                )

    # 模型有改动时，对应的 max_tokens_slider 滑块组件的最大值随之改动。
    # https://www.gradio.app/docs/dropdown
    model_dropdown.change(
        fn=fn_update_max_tokens,
        inputs=[model_dropdown, max_tokens_slider],
        outputs=max_tokens_slider
    )

    # 当用户在文本框处于焦点状态时按 Enter 键时，将触发此侦听器。
    # https://www.gradio.app/docs/textbox
    user_input_textbox.submit(
        fn=fn_prehandle_user_input,
        inputs=[
            user_input_textbox,
            chatbot],
        outputs=[chatbot]
    ).then(
        fn=fn_predict,
        inputs=[
            user_input_textbox,
            chatbot,
            model_dropdown,
            max_tokens_slider,
            temperature_slider,
            stream_radio],
        outputs=[chatbot]
    )

    # 单击按钮时触发。
    # https://www.gradio.app/docs/button
    submit_btn.click(
        fn=fn_prehandle_user_input,
        inputs=[
            user_input_textbox,
            chatbot],
        outputs=[chatbot]
    ).then(
        fn=fn_predict,
        inputs=[
            user_input_textbox,
            chatbot,
            model_dropdown,
            max_tokens_slider,
            temperature_slider,
            stream_radio],
        outputs=[chatbot]
    )

    clear_btn.click(lambda: None, None, chatbot, queue=False)


demo.queue().launch(share=True)
