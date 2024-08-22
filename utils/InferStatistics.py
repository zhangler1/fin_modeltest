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
        return f"{self.mean}±{self.std:.4f}"

# 示例用法

if __name__ == '__main__':
    mv = MeanVariance([2,23,5,4,2,45])
    print(mv)  # 输出 2±0.05
