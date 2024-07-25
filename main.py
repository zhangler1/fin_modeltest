import argparse
import ast

from exec_application_eval import eval_application
from exec_fin_eval import eval_fin_ability

parser = argparse.ArgumentParser()

# parser.add_argument('--model_name', required=True, type=str)
# parser.add_argument('--save_result_dir', required=True, type=str)
# parser.add_argument('--checkpoint_path', required=False, type=str)
# # parser.add_argument('--eval_type', required=True, type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # availble models are here "['spark13b', 'glm', 'spark_lite']"
    args.models="['spark_lite']"
    args.model_name=""
    args.eval_type="qa"
    # availble datasets are "['ceval', 'cflue', 'fineval']"
    args.datasets="['ceval']"
    args.datasetName=""
    args.request_type = "local"
    args.checkpoint_path="/home/llm/LLMs/Qwen1.5-1.8B"
    args.save_result_dir="modelResults"

if args.eval_type == 'qa':
    datasetNames= ast.literal_eval(args.datasets)
    models= ast.literal_eval(args.models)
    for model in models:
        args.model_name=model
        print(f"using model : {model}")
        for name in datasetNames:
            args.datasetName=name
            print(f"using dataset : {name}")
            eval_fin_ability(args)
elif args.eval_type == 'application':
    eval_application(args)
