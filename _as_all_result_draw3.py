from tsc_draw import Draw
from tsc_taskdb import TaskDB
import ast
import sys
from _am_create_all_train import DataHelper
from tsc_base import get
from copy import deepcopy

connect = ast.literal_eval(open('connect.txt', 'r', encoding='utf8').read().strip())
obj_train = TaskDB('am_all_train', mongo_url=connect['mongo_url'])
metric_tex_D = {i[1]: i[2] for i in DataHelper.get_default_metrics()['.shorthand']}  # {'指标简写':'指标tex',..}


def hierarchical_performance_ftest_task(best_result=('metric', 'dev')):
    """
    显著性检验图: 9损失-辅助损失/2方法/2流形/8维度/8混合/2数据
    以任务为子图
    === 220612 每行子图需要的数据量
        2*2*8*2*3*8+2*2*8*2*1 = 1600  # 分类行
        2*2*8*2*2*8+2*2*8*2*1 = 1088  # 其他任务
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    global no
    print('\n', no, sys._getframe().f_code.co_name, '...')
    def best_epoch_f(t): return str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
    
    # 多个子图的行列信息
    行s = [
        {'mark': 'Classification', 'des': 'LR'},
        {'mark': 'LinkPred', 'des': 'FD'},
        {'mark': 'GraphDistor', 'des': 'GD'},
        {'mark': 'HypernymyRel', 'des': 'HR'},
    ]
    列s = [
        {
            'des':'HRC',
            'metrics': [lambda 结果, 流形, **k: 结果['to_m'][str(流形)][i][0] for i in ['M1', 'M2', 'M3', 'M4']]
        }, {
            'des':'performace',
            'metrics': [lambda 结果, 流形, 分割, 指标, 任务, **k: -结果['to_m'][str(流形)]['M7'][0] if 任务 in {'GraphDistor'} else 结果['dataset'][分割][指标]]
        },
    ]
    draw = Draw(length=len(列s) * 10, width=len(行s) * 5, r=len(行s), c=len(列s))
    结合方式s = [
        {'mark': 'tw0.5', 'des': 'L0.5'},
        {'mark': 'tw0.6', 'des': 'L0.6'},
        {'mark': 'tw0.7', 'des': 'L0.7'},
        {'mark': 'tw0.8', 'des': 'L0.8'},
        {'mark': 'tw0.9', 'des': 'L0.9'},
        {'mark': 'mt1', 'des': 'ED'},
        {'mark': 'mt2', 'des': 'EfED'},
        {'mark': 'mt3', 'des': 'EfD'},
        {'mark': 'mt9', 'des': 'Normal'},
    ]
    # 初始数据库提取语句
    agg = [{'$match': {'$and': [
        {'paras.mark.1': {'$in': ['gcn', 'gat']}},  # 方法
        {'$or': [  # 流形
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},  # 维度
        {"paras.mark": {"$in": [i['mark'] for i in 结合方式s]}},  # 结合方式
        {"paras.mark.7": {"$in": ['dt32']}},  # 精度
        {"paras.mark.9": {"$in": ['o2', 'o3']}},  # 数据集
        {"paras.mark.10": {"$in": ['re1']}},  # 重复
        {"paras.trainParas.dh_L": {"$in": [
            ['Classification', 'LinkPred'],  #
            ['Classification', 'HypernymyRel'],
            ['Classification', 'GraphDistor'],
            # ['LinkPred', 'HypernymyRel'],
            ['LinkPred', 'GraphDistor'],  #
            # ['HypernymyRel', 'LinkPred'],
            ['HypernymyRel', 'GraphDistor'],  #
            ['GraphDistor', 'LinkPred'],  #
            ['GraphDistor', 'HypernymyRel'],
            ['Classification'], ['LinkPred'], ['GraphDistor'], ['HypernymyRel'],  # mt9
        ]}},
    ]}}]
    
    # 数据处理与绘图
    for i1, dh_L_0 in enumerate(行s):  # 行
        # 获取每个任务的所有结合方式数据
        agg_ = deepcopy(agg)
        agg_[0]['$match']['$and'].append({'paras.trainParas.dh_L.0': {'$in': [dh_L_0['mark']]}})  # 任务
        tasks = obj_train.que_tasks(agg_)
        print(f'{i1+1}行数据量: {len(tasks)}')
        # 排序
        tasks = sorted(tasks, key=lambda t: (
            get([int(i) if i.isdigit() else i for i in 'paras.trainParas.dh_L.1'.split('.')], t, ''),  # 辅助任务, 必须开头才能对应, 因为mt9没有这个要扩展
            get([int(i) if i.isdigit() else i for i in 'paras.mark.1'.split('.')], t, ''),  # 方法
            get([int(i) if i.isdigit() else i for i in 'paras.mark.2'.split('.')], t, ''),  # 流形
            get([int(i) if i.isdigit() else i for i in 'paras.mark.5'.split('.')], t, ''),  # 维度
            get([int(i) if i.isdigit() else i for i in 'paras.mark.9'.split('.')], t, ''),  # 数据集
        ))
        # 分别构建性能和层次能力图
        for i2, HRC_performance in enumerate(列s):  # 列
            # 提取每种结合方式的数据进行处理
            x = []
            yticks = []
            x_i3_len_max = x_i3_len_min_index = 0  # 用于没有辅助任务的mt9进行扩展
            for i3, mt_tw in enumerate(结合方式s):
                # 提取
                x_i3 = []  # 每种结合方式的结果
                for t in tasks:
                    if mt_tw['mark'] not in get('paras.mark'.split('.'), t):
                        continue
                    # 处理
                    流形 = get(['paras', 'encoderParas', 'manifold'], t)  # 此时使用的encoder流形
                    结果 = t['result_all']['epoch'][best_epoch_f(t)]  # 此时的指标结果
                    x_i3 += [f(
                        结果=结果,
                        流形=流形,
                        分割=best_result[1],
                        指标=best_result[0],
                        任务=dh_L_0['mark'],
                    ) for f in HRC_performance['metrics']]
                x.append(x_i3)
                yticks.append(mt_tw['des'])
                # 扩展mt9
                x_i3_len_max = len(x_i3) if len(x_i3) > x_i3_len_max else x_i3_len_max
                if mt_tw['mark'] in {'mt9'}:
                    x_i3_len_min_index = i3
            x[x_i3_len_min_index] = x[x_i3_len_min_index] * int(x_i3_len_max / len(x[x_i3_len_min_index]))
            # 整合所有的数据绘制显著性检验图
            draw.add_Ftest(
                x=x,
                yticks=yticks,
                xlabel='Rank',
                alpha=0.05,
                sub_title=f"({chr(i1 * len(列s) + i2 + 97)}) {dh_L_0['des']}, {HRC_performance['des']}",
                sub_title_fs=13,
                legend_fs=13,
                xlabel_fs=13,
                xtick_fs=12,
                ytick_fs=12,
                colors=draw.ncolors(len(x)),
            )
    draw.draw(f'as_{no}_{sys._getframe().f_code.co_name}.pdf')
    no += 1


def hierarchical_performance_ftest_data(best_result=('metric', 'dev')):
    """
    显著性检验图: 9损失-辅助损失/2方法/2流形/8维度/8混合/2数据
    以数据为子图
    === 220612 每行子图需要的数据量
        2*2*8*9*8+2*2*8*4*1 = 2432
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    global no
    print('\n', no, sys._getframe().f_code.co_name, '...')
    def best_epoch_f(t): return str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
    
    # 多个子图的行列信息
    行s = [
        {'mark': 'o2', 'des': 'Disease'},
        {'mark': 'o3', 'des': 'Animal'},
    ]
    列s = [
        {
            'des':'HRC',
            'metrics': [lambda 结果, 流形, **k: 结果['to_m'][str(流形)][i][0] for i in ['M1', 'M2', 'M3', 'M4']]
        }, {
            'des':'performace',
            'metrics': [lambda 结果, 流形, 分割, 指标, 任务, **k: -结果['to_m'][str(流形)]['M7'][0] if 任务 in {'GraphDistor'} else 结果['dataset'][分割][指标]]
        },
    ]
    draw = Draw(length=len(列s) * 10, width=len(行s) * 5, r=len(行s), c=len(列s))
    结合方式s = [
        {'mark': 'tw0.5', 'des': 'L0.5'},
        {'mark': 'tw0.6', 'des': 'L0.6'},
        {'mark': 'tw0.7', 'des': 'L0.7'},
        {'mark': 'tw0.8', 'des': 'L0.8'},
        {'mark': 'tw0.9', 'des': 'L0.9'},
        {'mark': 'mt1', 'des': 'ED'},
        {'mark': 'mt2', 'des': 'EfED'},
        {'mark': 'mt3', 'des': 'EfD'},
        {'mark': 'mt9', 'des': 'Normal'},
    ]
    # 初始数据库提取语句
    NC = [
        ['Classification', 'LinkPred'],
        ['Classification', 'HypernymyRel'],
        ['Classification', 'GraphDistor'],
    ]
    LP = [
        ['LinkPred', 'HypernymyRel'],
        ['LinkPred', 'GraphDistor'],
    ]
    HR = [
        ['HypernymyRel', 'LinkPred'],
        ['HypernymyRel', 'GraphDistor'],
    ]
    GD = [
        ['GraphDistor', 'LinkPred'],
        ['GraphDistor', 'HypernymyRel'],
    ]
    all_dh_L_L = [NC, LP, HR, GD]  # 分离任务用于4等分扩展
    agg = [{'$match': {'$and': [
        {'paras.mark.1': {'$in': ['gcn', 'gat']}},  # 方法
        {'$or': [  # 流形
            {"paras.mark": {"$all": ['E1', 'A1', 'D1']}},
            {"paras.mark": {"$all": ['E2', 'A2', 'D2']}},
        ]},
        {"paras.mark.5": {"$in": ['d2', 'd4', 'd6', 'd8', 'd10', 'd12', 'd14', 'd16']}},  # 维度
        {"paras.mark": {"$in": [i['mark'] for i in 结合方式s]}},  # 结合方式
        {"paras.mark.7": {"$in": ['dt32']}},  # 精度
        {"paras.mark.10": {"$in": ['re1']}},  # 重复
        {"paras.trainParas.dh_L": {"$in": sum(all_dh_L_L, []) + [[j] for j in set(i[0] for i in sum(all_dh_L_L, []))]}},
    ]}}]
    
    # 数据处理与绘图
    for i1, dh_L_0 in enumerate(行s):  # 行
        # 获取每个任务的所有结合方式数据
        agg_ = deepcopy(agg)
        agg_[0]['$match']['$and'].append({'paras.mark.9': {'$in': [dh_L_0['mark']]}})  # 任务
        tasks = obj_train.que_tasks(agg_)
        print(f'{i1+1}行数据量: {len(tasks)}')
        # 排序
        tasks = sorted(tasks, key=lambda t: (
            get([int(i) if i.isdigit() else i for i in 'paras.trainParas.dh_L.0'.split('.')], t, ''),  # 便于辅助任务4等分扩展
            get([int(i) if i.isdigit() else i for i in 'paras.trainParas.dh_L.1'.split('.')], t, ''),  # 辅助任务, 必须开头才能对应, 因为mt9没有这个要扩展
            get([int(i) if i.isdigit() else i for i in 'paras.mark.1'.split('.')], t, ''),  # 方法
            get([int(i) if i.isdigit() else i for i in 'paras.mark.2'.split('.')], t, ''),  # 流形
            get([int(i) if i.isdigit() else i for i in 'paras.mark.5'.split('.')], t, ''),  # 维度
            get([int(i) if i.isdigit() else i for i in 'paras.mark.9'.split('.')], t, ''),  # 数据集
        ))
        # 分别构建性能和层次能力图
        for i2, HRC_performance in enumerate(列s):  # 列
            # 提取每种结合方式的数据进行处理
            x = []
            yticks = []
            for i3, mt_tw in enumerate(结合方式s):
                # 提取
                x_i3 = []  # 每种结合方式的结果
                for t in tasks:
                    if mt_tw['mark'] not in get('paras.mark'.split('.'), t):
                        continue
                    # 处理
                    流形 = get(['paras', 'encoderParas', 'manifold'], t)  # 此时使用的encoder流形
                    结果 = t['result_all']['epoch'][best_epoch_f(t)]  # 此时的指标结果
                    x_i3 += [f(
                        结果=结果,
                        流形=流形,
                        分割=best_result[1],
                        指标=best_result[0],
                        任务=dh_L_0['mark'],
                    ) for f in HRC_performance['metrics']]
                # 4等分扩展
                if mt_tw['mark'] in {'mt9'}:
                    x_i3_ = []
                    split_num = [len(i) for i in sorted(all_dh_L_L, key=lambda t:get([0,0],t,'')) if i]
                    split_unit = int(len(x_i3) / len(split_num))
                    for i, j in enumerate(split_num):
                        x_i3_ += x_i3[i * split_unit: (i+1) * split_unit] * j
                    x_i3 = x_i3_
                x.append(x_i3)
                yticks.append(mt_tw['des'])
            print([len(i) for i in x])
            # 整合所有的数据绘制显著性检验图
            draw.add_Ftest(
                x=x,
                yticks=yticks,
                xlabel='Rank',
                alpha=0.05,
                sub_title=f"({chr(i1 * len(列s) + i2 + 97)}) task={dh_L_0['des']}, {HRC_performance['des']}",
                sub_title_fs=13,
                legend_fs=13,
                xlabel_fs=13,
                xtick_fs=12,
                ytick_fs=12,
                colors=draw.ncolors(len(x)),
            )
    draw.draw(f'as_{no}_{sys._getframe().f_code.co_name}.pdf')
    no += 1


if __name__ == '__main__':
    no = 1
    hierarchical_performance_ftest_task(('metric', 'dev'))
    # hierarchical_performance_ftest_data(('metric', 'dev'))
    