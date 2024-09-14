import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def get_hf_llm(model_name: str = "mattshumer/Reflection-Llama-3.1-70B",
            max_new_token = 1024,
            **kwargs):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization = True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_token = max_new_token,
        pad_token_id = tokenizer.eos_token_id,
        device_map = "auto"
    )
    llm = HuggingFacePipeline(
        pipeline = model_pipeline,
        model_kwargs = kwargs
    )
    return llm