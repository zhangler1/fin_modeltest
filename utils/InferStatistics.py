import numpy as np
import  re
# 假设有一个样本数据集
data = [5.1, 7.3, 6.4, 7.8, 5.9, 6.3, 7.0]

class MeanVariance:


    def __init__(self,*args):
        if len(args) == 1:
            data = args[0]
            if isinstance(data, list):
                self.mean = np.mean(data)
                # 计算总体标准差 (有偏估计)
                self.std = np.std(data, ddof=1)
            if isinstance(data, str):
                match = re.match(r"([-+]?\d*\.?\d*)\s*±\s*([-+]?\d*\.?\d*)", data)

                if match:
                    self.mean = float(match.group(1))  # 均值
                    self.variance = float(match.group(2))  # 方差
                else:
                    raise ValueError("无法解析表达式")
            if isinstance(data,tuple) and len(data) == 2:
                self.mean = data[0]
                self.std = data[1]
            if isinstance(data, (int, float))  :
                self.mean = data
                self.std = 0
        elif len(args) == 2:
            self.mean = args[0]
            # 计算总体标准差 (有偏估计)
            self.std = args[1]
        else:
            print("非法的参数数量 ")

    def __str__(self):
        # 返回格式化字符串
        return f"{self.mean:.2f}±{self.std:.2f}"
    def __repr__(self):
        # 返回格式化字符串
        return f"{self.mean:.2f}±{self.std:.2f}"
# 示例用法


import numpy as np
def MeanVarianceDicts(dicts:list[dict])->dict[str, MeanVariance]:
    if len(dicts) == 0:
        return {}
    # 创建一个新的字典用于存储方差
    variance_dict = {}

    # 遍历字典的键
    for key in dicts[0].keys():
        # 提取当前键的所有值
        values = [d[key] for d in dicts]
        if any([ not isinstance(value,(float,int))  for value in values]):
            raise ValueError(" only recept value type belongs to int or float !")
        # 计算方差
        mv = MeanVariance(values)  # 使用ddof=1来计算样本方差

        # 将方差存入新的字典
        variance_dict[key] = str(mv)

    return variance_dict

if __name__ == '__main__':
    dict1 = {'a': 1.5, 'b': 2, 'c': 3}
    dict2 = {'a': 2, 'b': 3.8, 'c': 4}
    dict3 = {'a': 3, 'b': 4, 'c': 5}
    dicts = [dict1, dict2, dict3]
    print(MeanVarianceDicts(dicts))  # 输出 2±0.05
