import importlib
import inspect
import os

import pandas as pd



def __load_cflue():
    dataset = pd.read_json('finDatasets/cflue/data/knowledge/knowledge.json')
    return dataset


def __load_application():
    dataset = pd.read_json('finDatasets/cflue/data/application/application.json')
    # dataset = pd.read_json('finDatasets/cflue/data/application/ap3.json')
    return dataset


def __load_fineval():
    from finDatasets.finEval.load_finEvalDataset import load_finEvalDataset
    dataset = load_finEvalDataset('finDatasets/finEval/FinEval/val')
    return dataset



def __load_ceval():
    from finDatasets.c_EVAL.load_C_eval import load_c_evalDataset
    dataset = load_c_evalDataset('finDatasets/c_EVAL')
    return dataset


def get_functions(moduleName="utils.dataset"):
    """
    获取指定模块中的所有函数。
    该函数导入指定的模块，并返回一个字典，其中包含该模块中所有函数的名称和函数对象。
    参数：
    moduleName (str): 要导入的模块的名称。默认为 "utils.dataset"。
    返回：
    dict: 一个字典，其中键是函数名称，值是函数对象。
    示例：
     functions = get_functions("math")
     functions["sqrt"]
    该函数对于动态加载模块并检索其函数特别有用，例如在需要根据配置或用户输入导入模块时。

    注意事项：
    - 确保传递的模块名称是有效的，并且模块能够被Python解释器找到。
    - 此函数只检索指定模块中的顶级函数，不包括类方法或嵌套函数。
    """
    functions = {}

    module_name = f"{moduleName}"
    module = importlib.import_module(module_name)
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            functions[name] = obj
    return functions




def load_dataset(dataset,*args,**kargs):
    functions = get_functions("utils.dataset")
    functionNames=[file[6:].strip("_").lower() for file in functions if file.startswith("__load")]
    func = functions.get(f"__load_{dataset}")
    print(f"{functionNames} are available , you chose {dataset}")
    if func and callable(func):
        return func(*args, **kargs)
    else:
        raise Exception(f"datasetName  '{dataset}' not found in :{functionNames} ")
