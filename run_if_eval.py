import subprocess
import sys

# 定义命令和参数
command = [
    "/home/llm/mambaforge/envs/py310-torch23-cu121/bin/python",
    "-m", "finDatasets.instruction_following_eval.evaluation_main",
    "--input_data=finDatasets/instruction_following_eval/data/input_data.jsonl",
    "--input_response_data=finDatasets/instruction_following_eval/data/input_response_data_spark_lite.jsonl",
    "--output_dir=finDatasets/instruction_following_eval/data"
]

# 使用 subprocess 运行命令
try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
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