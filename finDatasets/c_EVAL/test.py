import os
os.environ['HF_DATASETS_CACHE'] = './downloads'
import asyncio
import requests
from datasets import load_dataset
from datasets import get_dataset_config_names
import requests
import pandas as pd
# 获取数据集的所有配置名称

from chatWithModels.chat import chatWithModel



def process_example(example,prompt)->dict:
    # 假设你有一个函数或API可以调用大模型并返回结果

    # 假设数据集中有一个字段 'text'，作为模型输入
    model_input =f""""假设你是一位金融行业专家，请回答下列问题。
注意：题目是单选题，只需要返回一个最合适的选项，若有多个合适的答案，只返回最准确的即可。
注意：结果只输出两行，第一行只需要返回答案的英文选项(注意只需要返回一个最合适的答案)，第二行进行简要的解析，输出格式限制为：“答案：”，“解析：”。

{example['question']}
 选项A:{example['A']} 选项B:{example['B']} 选项C:{example['C']} 选项D:{example['D']}"""

    # 调用大模型进行推理
    result = chatWithModel("spark_lite",prompt,model_input)

    # 将结果添加到 example 中，例如 'prediction' 字段
    example['answerPredict'] = result

    return example

def main():
    config_names = ['accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 'business_administration', 'chinese_language_and_literature', 'civil_servant', 'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics', 'college_programming', 'computer_architecture', 'computer_network', 'discrete_mathematics', 'education_science', 'electrical_engineer', 'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology', 'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 'high_school_history', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'ideological_and_moral_cultivation', 'law', 'legal_professional', 'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer', 'middle_school_biology', 'middle_school_chemistry', 'middle_school_geography', 'middle_school_history', 'middle_school_mathematics', 'middle_school_physics', 'middle_school_politics', 'modern_chinese_history', 'operating_system', 'physician', 'plant_protection', 'probability_and_statistics', 'professional_tour_guide', 'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine']
    for config_name in config_names:
        prompt = f"你是一个中文问题专家.你在回答{config_name}领域的问题,给出推理步骤，并输出答案。"
        dataset = load_dataset(r"ceval/ceval-exam", name=f"{config_name}")["val"]
        result = dataset.map(lambda example: process_example(example, prompt))
        dataFrame=result.to_pandas()
        # 假设 dataFrame 是你的 DataFrame 对象
        dataFrame.to_excel(f'./answerVal/output_{config_name}.xlsx', index=False)
        print(f"Response JSON: {result}")

main()