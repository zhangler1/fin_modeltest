import json
import time
import os
from datetime import datetime
formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from  tqdm import tqdm



from absl import app
from absl import flags #flags需要和app结合使用哦嗯
from absl import logging
FLAGS = flags.FLAGS
FLAGS.DEFINE_string(
    "model_name", "spark_lite", "path to input data"
)
def main(argv):
    # 读取并打印命令行参数
    logging.info(f"Model Name: {FLAGS.model_name}")
    print(f"Model Name: {FLAGS.model_name}")

    print("格式化日期和时间：", formatted_datetime)
    from chatWithModels.chat import chatWithModel


    jsonls=[]
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")
    with open('finDatasets/instruction_following_eval/data/input_data.jsonl', 'r') as file:

        for line in tqdm(file,desc="load lines :"):
            data = json.loads(line.strip())  # 解析JSON
            jsonls.append(data)


    jsonlsResponse=[]
    for res in tqdm(iter(jsonls),desc="generating response ..."):
        prompt=res["prompt"]
        response=chatWithModel(FLAGS.model_name.value, 'you are a helpful assistant', prompt)
        json_res={"prompt": f"{prompt}",
              "response": f"{response}"}

        jsonlsResponse.append(json_res)

    with open(f"finDatasets/instruction_following_eval/data/input_response_data_{FLAGS.model_name.value}.jsonl","a")as f:
        for j in tqdm(jsonlsResponse,desc="writing response ...:"):
            json_line = json.dumps(j)  # 将字典转换为JSON字符串
            f.write(json_line+"\n")
        print("if response completed")

if __name__ == '__main__':
    os.chdir("/home/llm/data")
    app.run(main)



