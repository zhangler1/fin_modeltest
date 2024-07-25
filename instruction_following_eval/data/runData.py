import json
import time
from datetime import datetime
formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from  tqdm import tqdm

model_name="spark_lite"

print("格式化日期和时间：", formatted_datetime)
from chatWithModels.chat import chatWithModel




jsonls=[]
with open('input_data.jsonl', 'r') as file:
    for line in tqdm(file,desc="load lines :"):
        data = json.loads(line)  # 解析JSON
        print(data)
        jsonls.append(data)

jsonlsResponse=[]
for json in tqdm(iter(jsonls),gui=True,desc="generating response ...:"):
    prompt=json["prompt"]
    response=chatWithModel(model_name, '你是一个问答助手', prompt)
    json_res={"prompt": f"{prompt}",
          "response": f"{response}"}

    jsonlsResponse.append(json_res)

with open(f"input_response_data_f{model_name}_f{formatted_datetime}.jsonl","w")as f:
    for jsonlsResponse in tqdm(iter(jsonlsResponse),desc="writing response ...:"):
        json_line = json.dumps(jsonlsResponse)  # 将字典转换为JSON字符串
        file.write(json_line)
    f.close()
    print("if测评结束")


