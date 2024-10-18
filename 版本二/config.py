MODELS = [
    # 使用ollama本地部署的0.5b模型，为什么是0.5b呢？ 
    "qwen2:0.5b"
]

DEFAULT_MODEL = MODELS[0]

MODEL_TO_MAX_TOKENS = {
    "qwen2:0.5b": 4096
}