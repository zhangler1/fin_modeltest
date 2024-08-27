from datasets import load_dataset
from datasets import get_dataset_config_names
import requests
# 获取数据集的所有配置名称
config_names = get_dataset_config_names('ceval/ceval-exam')

# 打印配置名称
print(config_names)
dataset=load_dataset(r"ceval/ceval-exam",name="computer_network")

print(dataset['val'][0])
# {'id': 0, 'question': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____', 'A': '1', 'B': '2', 'C': '3', 'D': '4', 'answer': 'C', 'explanation': ''}

