import json

from future.backports.datetime import datetime
from sentry_sdk.utils import json_dumps

rw = []
rr = []
wr = []
ww = []
data = {"rw": rw, "wr": wr, "ww": ww}


def compare_fields_in_jsonl(file_path_A, file_path_B):
    with open(file_path_A, 'r', encoding='utf-8') as fileA:
        with open(file_path_B, 'r', encoding='utf-8') as fileB:

            for text1, text2 in zip(fileA, fileB):  # 对应的数据只有一行
                line1 = json.loads(text1)
                line2 = json.loads(text2)
                index = 0
                for index, (data1, data2) in enumerate(zip(line1, line2)):
                    if 'answer' in data1 and 'pred' in data1 and 'answer' in data2 and 'pred' in data2:
                        if data1['answer'] == data1['pred']:
                            if data2['answer'] == data2['pred']:
                                rr.append(index)
                            else:
                                rw.append(index)
                                with open(f"compare_rw.jsonl", 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(data1, ensure_ascii=False))
                                    f.write("\n")
                                    f.write(json.dumps(data2, ensure_ascii=False))
                                    f.write("\n")
                        else:
                            if data2['answer'] == data2['pred']:
                                wr.append(index)
                                with open(f"compare_wr.jsonl", 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(data1["response"], ensure_ascii=False))
                                    f.write("\n")
                                    f.write(json.dumps(data2["response"], ensure_ascii=False))
                                    f.write("\n")
                            else:
                                ww.append(index)
                    else:
                        print(f"Line {index}: Missing 'a' or 'b' field.")

    with open(f"compare_total.jsonl", 'w+', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write("\n")

    # 示例使用

if __name__ == '__main__':
    compare_fields_in_jsonl('spark_lite_ceval_result.json', 'spark_lite_ceval_resultb.json')