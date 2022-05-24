import pymongo
import ast
from pprint import pprint

mongo_url = ast.literal_eval(open('connect.txt', 'r', encoding='utf8').read().strip())['mongo_url']
dbc = pymongo.MongoClient(mongo_url)['aa_hyperbolic']['aa_train']

stat = [
    # 1 雷达图/显著性检验
    {'des': '4损失/3方法/3流形/8维度/2数据', 'sum': 4*3*3*8*2, '$and': [
        {"paras.mark.0": {"$in": ["Classification", "LinkPred", "GraphDistor", "HypernymyRel"]}},
        {"paras.mark.1": {"$in": ['mlp', 'gcn', 'gat']}},
        {"$or": [
            {"paras.mark": {"$all": ['E0', 'A0', 'D0']}},
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt32']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$in": ['o2', 'o3']}},
    ]},
    # 2 精度展示
    {'des': 'comb/5精度/2数据', 'sum': 5*2, '$and': [
        {"paras.mark.0": {"$in": ["LinkPred"]}},
        {"paras.mark.1": {"$in": ['comb']}},
        {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        {"paras.mark.5": {"$in": ['d2']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt32', 'dt64', 'dt128', 'dt512', 'dt3000']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$in": ['o2', 'o3']}},
    ]},
    # 3 三维透视图
    {'des': '3损失/2方法/3流形/8维度/36数据', 'sum': 3*2*3*8*36, '$and': [
        {"paras.mark.0": {"$in": ["LinkPred", "GraphDistor", "HypernymyRel"]}},
        {"paras.mark.1": {"$in": ['gcn', 'gat']}},
        {"$or": [
            {"paras.mark": {"$all": ['E0', 'A0', 'D0']}},
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt32']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$regex": '^t6-c'}},
    ]},
    # 4 典型层次结构
    {'des': '3流形/8维度/8数据', 'sum': 3*8*8, '$and': [
        {"paras.mark.0": {"$in": ["LinkPred"]}},
        {"paras.mark.1": {"$in": ['gcn']}},
        {"$or": [
            {"paras.mark": {"$all": ['E0', 'A0', 'D0']}},
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt32']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$regex": '^(t[1-4]|g[1-4])$'}},
    ]},
    # 5 不同层次结构的精度
    {'des': 'comb/精度3000/12数据', 'sum': 12, '$and': [
        {"paras.mark.0": {"$in": ["LinkPred"]}},
        {"paras.mark.1": {"$in": ['comb']}},
        {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        {"paras.mark.5": {"$in": ['d2']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt3000']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$regex": '^(t[1-4]|t5\.[12]\.[1-4]-m1)$'}},
    ]},
    # 6 只在最后一层做双曲是否合适
    {'des': '3损失/27流形/8维度/2数据', 'sum': 3*27*8*2, '$and': [
        {"paras.mark.0": {"$in": ["LinkPred", "GraphDistor", "HypernymyRel"]}},
        {"paras.mark.1": {"$in": ['gcn']}},
        {"paras.mark.2": {"$in": ['E0', 'E1', 'E2']}},
        {"paras.mark.3": {"$in": ['A0', 'A1', 'A2']}},
        {"paras.mark.4": {"$in": ['D0', 'D1', 'D2']}},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt32']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$in": ['o2', 'o3']}},
    ]},
    # 7 联合任务预训练
    {'des': '4损失/3流形/8维度/5混合/2数据', 'sum': 4*3*8*5*2, '$and': [
        {"paras.mark.0": {"$in": ["Classification", "LinkPred", "GraphDistor", "HypernymyRel"]}},
        {"paras.mark.1": {"$in": ['gcn']}},
        {"$or": [
            {"paras.mark": {"$all": ['E0', 'A0', 'D0']}},
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},
        {"paras.mark.6": {"$in": ['mt0', 'mt1', 'mt2', 'mt3', 'mt9']}},
        {"paras.mark.7": {"$in": ['dt32']}},
        {"paras.mark.8": {"$in": ['tw0.9', 'tw1']}},
        {"paras.mark.9": {"$in": ['o2', 'o3']}},
    ]},
    # 8 不同层次结构
    {'des': '3损失/3流形/8维度/9数据', 'sum': 3*3*8*9, '$and': [
        {"paras.mark.0": {"$in": ["LinkPred", "GraphDistor", "HypernymyRel"]}},
        {"paras.mark.1": {"$in": ['gcn']}},
        {"$or": [
            {"paras.mark": {"$all": ['E0', 'A0', 'D0']}},
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt32']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$regex": '^(t5|t5\.[12]\.[1-4]-m1)$'}},
    ]},
    # 9 不同层次结构 - 其他方法
    {'des': '2损失/3流形/8维度/9数据(不全)', 'sum': 258, '$and': [
        {"paras.mark.0": {"$in": ["LinkPred", "GraphDistor"]}},
        {"paras.mark.1": {"$in": ['gat']}},
        {"$or": [
            {"paras.mark": {"$all": ['E0', 'A0', 'D0']}},
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},
        {"paras.mark.6": {"$in": ['mt9']}},
        {"paras.mark.7": {"$in": ['dt32']}},
        {"paras.mark.8": {"$in": ['tw1']}},
        {"paras.mark.9": {"$regex": '^(t5|t5\.[12]\.[1-4]-m1)$'}},
    ]},
    # 未执行任务
    {'des': '未执行任务', 'sum': 0, '$and': [
        {"executed": False},
    ]},
]

or_L = []  # stat 中所有
cumulative_sum = 0  # 含重复累计数量
each_id_L = []  # [{id,..},..]; 与 stat 一一对应, 用于求交集数量
# 输出每个部分的统计数量
for i, s in enumerate(stat):
    # 加入执行过滤
    have_executed = False
    for j in s['$and']:
        if 'executed' in j:
            have_executed = True
            break
    if not have_executed:
        s['$and'].append({"executed": True})
    print(f"{i+1} - {s['des']}, 预测数量: {s['sum']}")
    agg = [{"$match": {"$and": s['$and']}}, {'$project': {'_id': '$_id'}}]
    all_id = {j['_id'] for j in dbc.aggregate(agg)}
    real_sum = len(all_id)
    if real_sum != s['sum']:
        print('以上实际数量(与预测不符):', real_sum)
        pprint(agg)
    or_L.append({'$and': s['$and']})
    cumulative_sum += real_sum
    each_id_L.append(all_id)

# 统计每个部分之间的重合数量
print('每个部分之间的重合数量[i,j,in_sum]:')
for i in range(len(each_id_L)):
    for j in range(i + 1, len(each_id_L)):
        in_sum = len(each_id_L[i] & each_id_L[j])
        if in_sum > 0:
            print([i + 1, j + 1, in_sum])

# 总统计数量
all_sum = list(dbc.aggregate([{'$count': 'count'}]))[0]['count']
agg = [{"$match": {"$or": or_L}}, {'$count': 'count'}]
or_sum = list(dbc.aggregate(agg))[0]['count']
print(f'总统计数量: {or_sum}, 累计数量: {cumulative_sum}, 总重复数量: {cumulative_sum-or_sum}')
print(f'数据库总数: {all_sum}, 剩余数量: {all_sum-or_sum}')
if all_sum != or_sum:
    print('非统计结果的例子:')
    agg = [{"$match": {"$nor": or_L}}, {'$limit': 10}, {'$project': {'mark': '$paras.mark'}}]
    for i, j in enumerate(dbc.aggregate(agg)):
        print(i+1, j)
