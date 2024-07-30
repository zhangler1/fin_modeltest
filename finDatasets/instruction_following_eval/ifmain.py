import subprocess
import sys
import os

# 定义命令和参数

#被导入使用所以都得用main函数的路径
def instruction_following_eval(model_name:str):
    command_rundata= [
        "/home/llm/mambaforge/envs/py310-torch23-cu121/bin/python",
        "-m", "finDatasets.instruction_following_eval.data.runData",
        f"--model_name={model_name}"
    ]
    data_path=f"finDatasets/instruction_following_eval/data/input_response_data_{model_name}.jsonl"
    command_eval = [
        "/home/llm/mambaforge/envs/py310-torch23-cu121/bin/python",
        "-m", "finDatasets.instruction_following_eval.evaluation_main",
        "--input_data=finDatasets/instruction_following_eval/data/input_data.jsonl",
        f"--input_response_data={data_path}",
        "--output_dir=finDatasets/instruction_following_eval/data"
    ]

    # 使用 subprocess 运行命令
    try:
        if not os.path.exists(data_path):
            print("please run runData.py before generate llm ")
            return
        result = subprocess.run(command_eval, capture_output=True, text=True, check=True)
        # 输出结果
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        # 如果命令返回非零退出码，将抛出 CalledProcessError
        print("An error occurred while executing the command.")
        print("Exit Code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(e.returncode)

if __name__ == '__main__':
    os.chdir("/home/llm/data")
    instruction_following_eval("spark_lite")