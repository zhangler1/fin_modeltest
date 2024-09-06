import os
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
os.environ['HF_DATASETS_CACHE'] = './downloads'
# config_names = get_dataset_config_names('ceval/ceval-exam')
config_names = ['accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 'business_administration',
                'chinese_language_and_literature', 'civil_servant', 'clinical_medicine', 'college_chemistry',
                'college_economics', 'college_physics', 'college_programming', 'computer_architecture',
                'computer_network', 'discrete_mathematics', 'education_science', 'electrical_engineer',
                'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology',
                'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 'high_school_history',
                'high_school_mathematics', 'high_school_physics', 'high_school_politics',
                'ideological_and_moral_cultivation', 'law', 'legal_professional', 'logic', 'mao_zedong_thought',
                'marxism', 'metrology_engineer', 'middle_school_biology', 'middle_school_chemistry',
                'middle_school_geography', 'middle_school_history', 'middle_school_mathematics',
                'middle_school_physics', 'middle_school_politics', 'modern_chinese_history', 'operating_system',
                'physician', 'plant_protection', 'probability_and_statistics', 'professional_tour_guide',
                'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner',
                'veterinary_medicine']
def load_c_evalDataset(path):
    if "c_EVAL.xlsx" in os.listdir(path):
        df=pd.read_excel(os.path.join(path,"c_EVAL.xlsx"))
        return df
    else:
        dfs=[]
        for config_name in config_names:

            prompt = f"你是一个中文问题专家.你在回答{config_name}领域的问题,给出推理步骤，并输出答案。"
            dataset = load_dataset(r"ceval/ceval-exam", name=f"{config_name}")["val"]

            dataFrame = dataset.to_pandas()
            dataFrame['task'] = "单项选择题"
            dataFrame["domain"]=config_name
            print(config_name)
            # 假设 dataFrame 是你的 DataFrame 对象
            # dataFrame.to_excel(f'./answerVal/output_{config_name}.xlsx', index=False)
            # print(f"Response JSON: {result}")
            dfs.append(dataFrame)
        df = pd.concat(dfs, ignore_index=True)
        df['answerPredict'] = ""


        #更新choices行
        def update_choices(example):
            choices = {"A":f"{example['A']}" ,
                       "B":f"{example['B']}" ,
                       "C":f"{example['C']}",
                        "D":f"{example['D']}"
                       }

            return str(choices)

        # 使用 apply 方法遍历每一行，并为新列 'C' 赋值
        # df['answerPredict'] = df.apply(update_column, axis=1)

        df['choices'] = df.apply(update_choices, axis=1)
        df.to_excel(os.path.join(path,"c_EVAL.xlsx"))
    return df



if __name__ == '__main__':
    os.chdir("/home/llm/data")
    a=load_c_evalDataset("finDatasets/c_EVAL")
    pass