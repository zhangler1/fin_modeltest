#from main import dataset
from utils.evaluator import load_models_tokenizer, load_llama_models_tokenizer
from utils.dataset import load_dataset
from utils.compute_score import *
from tqdm import tqdm
import argparse
from chatWithModels.chat import chatWithModel
from utils.loadModels import load_models_tokenizer, chat_with_model_hf

def eval_application(args):
    # load model & tokenizer
    if args.request_type == "http":
        model = None

    elif args.request_type == "local":

        model, tokenizer = load_models_tokenizer(args.checkpoint_path)

    # 载入评测集
    dataset = load_dataset(args.eval_type)
    #dataset = dataset[dataset["sub_task"]==args.sub_task]
    # 大模型推理回答&记录答案
    responses = []

    for _, record in tqdm(dataset.iterrows()):
        if record['sub_task'] == "金融英中翻译":
            prompt = record['instruction'].split('\n')[0]

            input = record['instruction'].split('\n')[2]
            if  record['sub_task'] == "金融英中翻译":
                input = record['instruction'].split('\n')[2]
            try:
                if args.request_type == "http":
                    #model_response = chatWithModel(args.model_name, "", prompt)
                    model_response = chatWithModel(args.model_name, prompt, input)
                elif args.request_type == "local":

                    model_response = chat_with_model_hf(model, tokenizer, prompt)
                else:
                    raise ValueError("Invalid request type")
            except Exception:
                model_response = Exception
            # model_response, _ = model.chat(
            #     tokenizer,
            #     prompt,
            #     history=None,
            # )
            responses.append(model_response)

    os.makedirs(os.path.join(args.save_result_dir, args.model_name), exist_ok=True)
    result_path = os.path.join(args.save_result_dir, args.model_name,
                               f"{args.model_name}_{args.datasetName}_{args.start_time}.json")
    #result_path = os.path.join(args.save_result_dir, args.model_name,"{args.model_name}_{args.datasetName}_{args.start_time}.json")

    if not args.save_result_dir:
        os.makedirs(args.save_result_dir, exist_ok=True)
    dataset = dataset[dataset["sub_task"] == "金融英中翻译"]
    dataset["model_response"] = responses

    dataset.to_json(result_path, orient='records', force_ascii=False)

    # 计算应用评分
    #get_application_score(args)

    bleu1,bleu4 = get_application_score(result_path)


    metaData={}
    metaData["time"]=args.start_time
    metaData["model_name"]=args.model_name
    metaData["datasetName"]=args.datasetName
    metaData["task"]=args.eval_type
    metaData["sub_task"] = args.sub_task

    metrics = {}
    dataset["response"] = responses
    metrics["blue_1"] = bleu1
    metrics["blue_4"] = bleu4

    if args.save_result_dir:
        os.makedirs(args.save_result_dir, exist_ok=True)
        dataset.to_json(result_path, orient='records', force_ascii=False)
    dataset["metrics"] = metrics
    return metrics,metaData


def get_application_score(path):
    # _path = args.save_result_dir
    # file_path = f'{_path}/{args.datasetName}/{args.model_name}_application_result.json'

    result = {}
    #print('Model: %s' % args.models[0])
    # QA
    # rouge_l = compute_finqa(file_path)
    # result['QA'] = {'rouge-L': rouge_l
    #                 }

    # # TG
    # rouge_l_tg, _ = compute_text_generation(file_path)
    # result['TG'] = {'rouge-L': rouge_l_tg
    #                 }
    # # MT-e2zh
    bleu1,bleu4 = compute_nmt_en2zh(path)
    result['MT-e2zh'] = {'BLEU': bleu1}
    return bleu1,bleu4
    # # MT-zh2e
    # bleu = compute_nmt_zh2en(file_path)
    # result['MT-zh2e'] = {'BLEU': bleu
    #                      }
    #TC
    # acc = compute_text_classification(file_path)
    # result['TC'] = {'ACC': acc}

    # RE
    # f1, _ = compute_extraction(file_path)
    # result['RE'] = {'F1-score': f1}
    print(result)
