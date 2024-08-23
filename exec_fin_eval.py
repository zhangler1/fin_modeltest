
from utils.evaluator import load_models_tokenizer
from utils.dataset import load_dataset
from utils.format_example import format_one_example, format_multi_example
from utils.extract_choice import extract_one_choice, extract_multi_choice
from utils.compute_score import *
from tqdm import tqdm
import pandas as pd
import jieba
from utils.loadModels import load_models_tokenizer, chat_with_model_hf
from chatWithModels.chat import chatWithModel


def eval_fin_ability(args)->dict:
    global model_response
    if args.request_type == "http":
        model = None

    elif args.request_type == "local":

        model, tokenizer = load_models_tokenizer(args.checkpoint_path)

    # 载入评测集
    dataset = load_dataset(args.datasetName)
    print(len(dataset))

    # 大模型推理回答&记录答案
    responses = []
    extractAns=[]
    y_true, y_pred = [], []
    references, candidates = [], []
    for _, record in tqdm(dataset.iterrows(),desc="Processing"):
        if record['task'] == '多项选择题':
            prompt = format_multi_example(record)
            # model_response, _ = model.chat(
            #     tokenizer,
            #     prompt,
            #     history=None,
            # )
            # model_response=chat_with_spark("",prompt)

            if args.request_type == "http":
                model_response = chatWithModel(args.model_name, "", prompt)
            elif args.request_type == "local":

                model_response = chat_with_model_hf(model, tokenizer, prompt)
            else:
                raise ValueError("Invalid request type")

            if len(model_response.split('\n')) == 2:
                pred = extract_multi_choice(model_response.split('\n')[0], [choice for choice in record["choices"]])
            else:
                pred = extract_multi_choice(model_response, [choice for choice in record["choices"]])
        else:
            prompt = format_one_example(record)
            # model_response, _ = model.chat(
            #     tokenizer,
            #     prompt,
            #     history=None,
            # )
            if args.request_type == "http":
                model_response = chatWithModel(args.model_name, "", prompt)
            elif args.request_type == "local":
                model_response = chat_with_model_hf(model, tokenizer, prompt)
            if len(model_response.split('\n')) == 2:
                pred = extract_one_choice(model_response.split('\n')[0], [choice for choice in record["choices"]])
            else:
                pred = extract_one_choice(model_response, [choice for choice in record["choices"]])

        responses.append(model_response)
        y_pred.append(pred)
        y_true.append(record['answer'])
        if 'analysis' in record and not pd.isna(record['analysis']):
            references.append(" ".join(jieba.lcut(record['analysis'])))
            if len(model_response.strip().split('\n')) == 2:
                candidates.append(" ".join(jieba.lcut(model_response.strip().split('\n')[1].replace("解析：", ""))))
            else:
                candidates.append(" ".join(jieba.lcut(model_response)))

    # 计算分数
    score_acc = acc_score(y_true, y_pred)
    score_weighted_f1 = f1_score(y_true, y_pred)
    if "analysis" in dataset.columns:
        bleu_1, bleu_4 = bleu_score(references, candidates)
        rouge_1, rouge_2, rouge_l = rouge_score(references, candidates)

    result_path = os.path.join(args.save_result_dir, f"{args.model_name}_{args.datasetName}_{args.start_time}.json")
    metaData={}
    metaData["time"]=args.start_time
    metaData["model_name"]=args.model_name
    metaData["datasetName"]=args.datasetName
    metaData["task"]=args.eval_type

    metrics = {}
    dataset["response"] = responses
    dataset["pred"] = y_pred
    metrics["acc"] = score_acc
    metrics["weighted_f1"] = score_weighted_f1
    if "analysis" in dataset.columns:
        # 如果数据集合中有标准答案，比如总结的标准答案
        metrics["bleu_1"] = bleu_1
        metrics["bleu_4"] = bleu_4
        metrics["rouge_1"] = rouge_1
        metrics["rouge_2"] = rouge_2
        metrics["rouge_L"] = rouge_l
    if args.save_result_dir:
        os.makedirs(args.save_result_dir, exist_ok=True)
        dataset.to_json(result_path, orient='records', force_ascii=False)
    dataset["metrics"] = metrics
    return metrics,metaData

