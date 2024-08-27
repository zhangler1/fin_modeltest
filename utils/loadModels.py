import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from modelscope import Model
from modelscope.models.nlp.llama2 import Llama2Tokenizer


def load_models_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path,
                                              use_fast=False,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(checkpoint_path)

    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    return model, tokenizer


def chat_with_model_hf (model, tokenizer,prompt):
    def remove_prefix(a, b):
        if a.startswith(b):
            return a[len(b)+1:]
        else:
            return a

    # 示例




    # 对输入文本进行编码
    input_id= tokenizer(prompt, return_tensors="pt",padding=True, truncation=True).to('cuda')

    input_ids=input_id['input_ids'].to('cuda')
    attention_mask=input_id['attention_mask'].to('cuda')
    # 使用模型生成文本
    output_ids = model.generate(input_ids, attention_mask=attention_mask,num_return_sequences=1,pad_token_id=model.config.eos_token_id)
    # 生成 attention_mask

    # 解码生成的文本
    result = tokenizer.decode(output_ids[0],skip_special_tokens=True)
    # result :str= remove_prefix(output_text, prompt)
    if len(result)==0:
        result=result+"抱歉我不知道"
    print(result)
    return  result

# 创建对象


if __name__ == '__main__':
    checkpoint_path= "/LLMs/Qwen1.5-1.8B"
    prompt=("告诉我什么不是物理学")
    model, tokenizer=load_models_tokenizer(checkpoint_path)
    output_text=chat_with_model_hf( model, tokenizer,prompt)
    # 打印生成的文本
