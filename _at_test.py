from tsc_taskdb import TaskDB
import ast

# 清理实验结果数据库
connect = ast.literal_eval(open('connect.txt', 'r', encoding='utf8').read().strip())
obj_train = TaskDB('am_all_train', mongo_url=connect['mongo_url'])
obj_train.close()
