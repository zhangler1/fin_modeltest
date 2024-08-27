from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

# 星火认知大模型Spark3.5 Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'
# 星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = 'd93945b6'
SPARKAI_API_SECRET = 'Nzk0ZDlhMDdmNDJiYTBjNjE1NDI4MGNm'
SPARKAI_API_KEY = '342d834338a3186b8f276afc26649364'
# 星火认知大模型Spark3.5 Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'general'




@retry(stop=stop_after_attempt(3))
def chat_with_spark_lite(prompt:str,inputText:str):

    def chat_spark(prompt: str, inputText: str):

        spark = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=SPARKAI_APP_ID,
            spark_api_key=SPARKAI_API_KEY,
            spark_api_secret=SPARKAI_API_SECRET,
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
        )
        messages = [ChatMessage(
            role="system",
            content=f'{prompt}'
        ), ChatMessage(
            role="user",
            content=f'{inputText}'
        )]
        handler = ChunkPrintHandler()
        a = spark.generate([messages], callbacks=[handler])
        print(a.generations[0][0].text)
        return a.generations[0][0].text

    try:
        return chat_spark(prompt,inputText)
    except Exception as e :
        print(f"{e}")
        raise



if __name__ == '__main__':

        chat_with_spark_lite("","你是谁")
