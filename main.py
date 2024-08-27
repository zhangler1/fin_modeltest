import argparse
import ast
import json

from torch.utils.hipify.hipify_python import meta_data


from exec_application_eval import eval_application
from exec_fin_eval import eval_fin_ability
from finDatasets.instruction_following_eval.ifmain import instruction_following_eval
parser = argparse.ArgumentParser()
from utils.InferStatistics import MeanVarianceDicts

from datetime import datetime

# 获取当前日期


# 假设你有不同的模型数据集名称

# parser.add_argument('--model_name', required=True, type=str)
# parser.add_argument('--save_result_dir', required=True, type=str)
# parser.add_argument('--checkpoint_path', required=False, type=str)
# # parser.add_argument('--eval_type', required=True, type=str)
args = parser.parse_args()


if __name__ == '__main__':
    # availble models are here "['spark13b', 'glm', 'spark_lite']"
    args.models="['spark_lite']"
    args.eval_type="application"
    # availble datasets are "['ceval', 'cflue', 'fineval']"
    args.datasets="['cflue']"
    args.datasetName=""
    args.request_type = "http"
    args.checkpoint_path="/home/llm/LLMs/Qwen1.5-1.8B"
    args.save_result_dir="modelResults"
    args.eval_times=5
    args.start_time = datetime.now().strftime('%Y-%m-%d %H.%M')
    args.sub_task = "英中翻译"


if args.eval_type == 'qa':
    datasetNames= ast.literal_eval(args.datasets)
    models= ast.literal_eval(args.models)
    for model in models:
        args.model_name=model
        print(f"using model : {model}")
        for dataset in datasetNames:
            args.datasetName=dataset
            print(f"using dataset : {dataset}")
            results=[]
            for time in range(args.eval_times):
                data,metaData=eval_fin_ability(args)
                results.append(data)

            result=MeanVarianceDicts(results)
            result["metaData"]=metaData
            file_name = f"{args.model_name}_{args.datasetName}_history.jsonl"
            with open(file_name,"a")as f:
                f.write(json.dumps(result,ensure_ascii=False) + '\n')

elif args.eval_type == 'if_eval':
    models= ast.literal_eval(args.models)
    for model in models:
        args.model_name=model
        print(f"using model : {model}")
        instruction_following_eval(model)

elif args.eval_type == 'application':
    datasetNames= ast.literal_eval(args.datasets)
    models= ast.literal_eval(args.models)
    for model in models:
        args.model_name=model
        print(f"using model : {model}")
        for dataset in datasetNames:
            args.datasetName=dataset
            print(f"using dataset : {dataset}")
            results=[]
            for time in range(args.eval_times):
                data,metaData=eval_application(args)
                results.append(data)

            result=MeanVarianceDicts(results)
            result["metaData"]=metaData
            file_name = f"{args.model_name}_{args.datasetName}_history.jsonl"
            with open(file_name,"a")as f:
                f.write(json.dumps(result,ensure_ascii=False) + '\n')

