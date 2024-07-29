import json
import time
from datetime import datetime
formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from  tqdm import tqdm



from absl import app
from absl import flags
from absl import logging
model_name=flags.DEFINE_string(
    "input_data", "spark_lite", "path to input data", required=True
)
print("格式化日期和时间：", formatted_datetime)
from chatWithModels.chat import chatWithModel




jsonls=[]
with open('input_data.jsonl', 'r') as file:
    for line in tqdm(file,desc="load lines :"):
        data = json.loads(line.strip())  # 解析JSON
        jsonls.append(data)


jsonlsResponse=[]
for res in tqdm(iter(jsonls),desc="generating response ..."):
    prompt=res["prompt"]
    response=chatWithModel(model_name.value, 'you are a helpful assistant', prompt)
    json_res={"prompt": f"{prompt}",
          "response": f"{response}"}

    jsonlsResponse.append(json_res)

with open(f"input_response_data_{model_name.value}.jsonl","a")as f:
    for j in tqdm(jsonlsResponse,desc="writing response ...:"):
        json_line = json.dumps(j)  # 将字典转换为JSON字符串
        f.write(json_line+"\n")
    print("if测评结束")


