import os
import pandas as pd



# 定义存放 CSV 文件的文件夹路径
def load_finEvalDataset(folder_path):
    # 初始化一个空的列表，用于存放所有的 DataFrame
    dataframes = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为 CSV
        if filename.endswith('.csv'):
            # 构造文件的完整路径
            file_path = os.path.join(folder_path, filename)
            # 读取 CSV 文件为 DataFrame
            df = pd.read_csv(file_path, index_col=0, skip_blank_lines=True)
            # 将 DataFrame 添加到列表中
            if df.iloc[:, -1].isnull().all():
                df = df.iloc[:, :-1]

            # 将 DataFrame 添加到列表中
            dataframes.append(df)

    # 打印读取的 DataFrame 列表
    for i, df in enumerate(dataframes):
        print(f"DataFrame {i + 1}:")

    # 如果需要，可以将所有的 DataFrame 合并为一个 DataFrame
    # 例如，可以将它们按行合并
    df = pd.concat(dataframes, ignore_index=True)
    df['answerPredict'] = ""
    df['task']="单项选择题"
    # def update_column(example):
    #     model_input = f""""假设你是一位金融行业专家，请回答下列问题。
    #     注意：题目是单选题，只需要返回一个最合适的选项，若有多个合适的答案，只返回最准确的即可。
    #     注意：结果只输出两行，第一行只需要返回答案的英文选项(注意只需要返回一个最合适的答案)，第二行进行简要的解析，输出格式限制为：“答案：”，“解析：”。
    #
    #      {example['question']}
    #      选项A:{example['A']} 选项B:{example['B']} 选项C:{example['C']} 选项D:{example['D']}"""
    #     result = chat_with_spark("", model_input)
    #
    #     return result

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
    df.to_excel("finEvalDataset.xlsx")
    return df
if __name__ == '__main__':
    os.chdir("/home/llm/data")
    load_finEvalDataset("finDatasets/finEval/FinEval/val")