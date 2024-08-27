import requests
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import json

@retry(stop=stop_after_attempt(3))
def chat_with_GLM(prompt:str,inputText:str):

    # 定义重试策略：重试3次后停止

    def make_request():
        url = "http://182.215.238.14:8088/v1/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": "chatglm3-130b-v0.2",
            "messages": [
                {
                    "role": "system",
                    "content": f"you are helpful assistant"
                },
                {
                    "role": "user",
                    "content": f"{inputText}"
                }
            ],
            "do_sample": True,
            "max_tokens": 1024,
            "temperature": 0.75,
            "stream": False
        }

        response = requests.post(url, headers=headers, json=data, verify=False)
        response.raise_for_status()  # 如果响应状态码不是200，抛出HTTPError

        result=response.json()
        print(result['choices'][0]['message']['content'])
        return result['choices'][0]['message']['content']

    try:
        result = make_request()
        print("请求成功:", result)
        return result
    except Exception as e:
        print(e)
        raise





if __name__ == '__main__':
    chat_with_GLM("","你是谁")