
import importlib
import os
import inspect

from tenacity import RetryError


#todo 需要改成按需要加載

def load_functions_from_package(package_name):
    functions = {}
    package = importlib.import_module(package_name)
    package_path = package.__path__[0]

    for filename in os.listdir(package_path):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = f"{package_name}.{filename[:-3]}"
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    functions[name] = obj
    return functions
def chatWithModel(model_name:str,*args,moduleName="chatWithModels",**kargs)->str:

    functions=load_functions_from_package(moduleName)
    func_name=f"chat_with_{model_name.lower()}"
    func = functions.get(func_name)
    if func and callable(func):
        try:
            return func(*args,**kargs)
        except RetryError as e:
            return "max_tries"
    else:
        print(f"Function '{func_name}' not found in moduleName:{moduleName}")

if __name__ == '__main__':


    # 示例：调用函数
    chatWithModel("spark_lite", '你是一个问答助手',"你是谁？")

