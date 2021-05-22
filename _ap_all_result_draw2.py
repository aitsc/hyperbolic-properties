from _am_create_all_train import *
from tanshicheng import Draw
from scipy import stats

obj_data = 数据生成任务('al_all_data')
obj_train = 训练生成任务('am_all_train',
                   mongo_url=ast.literal_eval(open('connect.txt', 'r', encoding='utf8').read().strip())['mongo_url'])
no = 1
metric_tex_D = {i[1]: i[2] for i in DataHelper.get_default_metrics()['.shorthand']}  # {'指标简写':'指标tex',..}


def get_obj_marks(query, dels=None):
    """
    获取任务的mark标记用于获取对应任务结果, 同时删除一些mark的列用于同时获取多个结果
    :param query: 穷举构建简单任务方法.初始参数D
    :param dels: list(mark_index_D.keys())
    :return:
    """
    if dels is None:
        dels = []
    paras_L = 穷举构建简单任务方法(query, obj_data, {}, obj_train)[0]
    mark_L = []
    for paras in paras_L:
        mark = paras['mark']
        for d in dels:
            del mark[mark_index_D[d]]  # 删除某个信息以查询这个信息的多个结果
        mark_L.append(mark)
    return mark_L


def get_dim_task(tasks, dim_S=None):  # 筛选具有某些维度的任务
    if dim_S is None:
        dim_S = {2, 4, 6, 8, 10, 12, 14, 16}
        dim_S = {f'd{i}' for i in dim_S}
    tasks_ = []
    dim_S_ = set()
    for task in tasks:
        dim = task['paras']['mark'][mark_index_D['d']]
        if dim in dim_S:
            tasks_.append(task)
            dim_S_.add(dim)
    tasks_ = sorted(tasks_, key=lambda t: t['paras']['mark'][mark_index_D['d']])  # 排序
    # print([t['paras']['mark'][mark_index_D['d']] for t in tasks_])
    assert len(tasks_) == len(tasks), f'这组任务缺少一些维度: {dim_S - dim_S_}'
    return tasks_, \
           tasks_[0]['paras']['encoderParas']['manifold']  # encoder 输出的流形


def chunks(l, n):
    """
    将一个列表划分为长度相等的子列表, 不够等分会报错
    :param l: list; 原始列表
    :param n: int; 每个子列表的长度
    :return: [[],..]
    """
    assert len(l) % int(n) == 0, f'{len(l)} % {int(n)} != 0'
    n = int(n)
    ll = []
    for i in range(0, len(l), n):
        ll.append(l[i:i + n])
    return ll


def friedman_test(paras_L, file_end='friedman'):
    """
    绘制 f test 图
    :param paras_L: [{'x':[[],..], 'yticks':[], 'title_end':''},..]; 一个dict一个子图
    :param file_end: str; 文件名后缀
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, file_end, '...')
    r = len(paras_L)
    draw = Draw(length=10, width=5 * r, r=r, c=1)
    alpha = 0.05
    for paras in paras_L:
        x = paras['x']
        yticks = paras['yticks']
        title_end = paras['title_end']
        r = draw.add_Ftest(
            x=x, yticks=yticks, xlabel='Rank', alpha=alpha,
            sub_title='Friedman test: $p$-value{p}, Nemenyi post-hoc test: $\\alpha$=' +
                      f'{alpha}{title_end}',
            p_vaule_f=lambda t, p: t.replace('{p}', ('=%.3e' % p) if p >= 10 ** -15 else '<1e-15')
        )
        del r['x_no']
        print(r)
    global no
    draw.draw(f'ap_{no}_{file_end}.pdf')
    no += 1


def decoder_radar(best_result=('metric', 'dev')):
    """
    6个雷达图: (4任务*5指标)*3E流形*GCN*(Animal+Disease)
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, '...')
    r = 2
    c = 3
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    x_labels = ['M1', 'M2', 'M3', 'M4', 'M7']
    line_labels = ['NC', 'LP', 'GD', 'HR']
    datasets = ['Animal', 'Disease']  # 与 ds 对应
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
    method_result_D = {}  # {方法名:[结果1,..],..}
    metrics_F = ['M1', 'M2', 'M3', 'M4']  # 用于 friedman test 的指标

    for i, ds in enumerate([['o3'], ['o2']]):
        for j, m in enumerate([0, 1, 2]):
            line_data = []
            for k, dh_L in enumerate([['Classification'], ['LinkPred'], ['GraphDistor'], ['HypernymyRel']]):
                tasks = obj_train.que_tasks([{'$match': {'$and': [
                    {'paras.mark': {
                        '$all': [dh_L[0], f'E{m}', f'A{m}', f'D{m}', 'mt9', 'dt32', 'tw1', ds[0], 're1'],
                        '$in': [f'd{i}' for i in [2, 4, 6, 8, 10, 12, 14, 16]],
                    }},
                    {'paras.mark': {'$in': ['mlp', 'gcn', 'gat']}},
                    # {'paras.mark': {'$in': ['gcn']}},
                ]}}])
                tasks = sorted(tasks, key=lambda t: tuple(t['paras']['mark']))  # 排序
                指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(m)] for t in tasks]
                line_data.append([])
                for x_label in x_labels:  # 每个指标的结果
                    result_L = [指标D[x_label][0] for 指标D in 指标D_L]  # 所有维度平均
                    line_data[-1].append(sum(result_L) / len(result_L))
                # friedman test
                method = f'{Manifold.s_to_tex(m)}-{line_labels[k]}'
                for metric in metrics_F:
                    result_L = [指标D[metric][0] for 指标D in 指标D_L]
                    if method in method_result_D:
                        method_result_D[method] += result_L
                    else:
                        method_result_D[method] = result_L
            sub_title = f'({len(draw.already_drawn) + 1}): {Manifold.s_to_tex(m)}, {datasets[i]}'
            draw.add_radar([metric_tex_D[metric] for metric in x_labels],
                           line_labels, line_data, sub_title, fill_alpha=0.1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend='best')
    global no
    stop_strategy = f'{best_result[1]}-{best_result[0]}'
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}_{stop_strategy}.pdf')
    method_result_L = sorted(method_result_D.items())
    no += 1
    return {  # 用于 friedman_test
        'x': [i[1] for i in method_result_L],
        'yticks': [i[0] for i in method_result_L],
        'title_end': f', Stop strategy: {stop_strategy}'
    }


def encoder_radar(best_result=('metric', 'dev')):
    """
    6个雷达图: (3方法*5指标)*3E流形*LP*(Animal+Disease)
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, '...')
    r = 2
    c = 3
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    x_labels = ['M1', 'M2', 'M3', 'M4', 'M7']
    line_labels = ['MLP', 'GCN', 'GAT', 'Comb']
    datasets = ['Animal', 'Disease']  # 与 ds 对应
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
    method_result_D = {}  # {方法名:[结果1,..],..}
    metrics_F = ['M1', 'M2', 'M3', 'M4']  # 用于 friedman test 的指标
    tasks_num = None
    colors = draw.ncolors(len(line_labels))[::-1]

    for i, ds in enumerate([['o3'], ['o2']]):
        for j, m in enumerate([0, 1, 2]):
            line_data = []
            for k, layer in enumerate(['mlp', 'gcn', 'gat', 'comb']):  # comb 放最后
                if layer != 'comb':
                    tasks = obj_train.que_tasks([{'$match': {'$and': [
                        {'paras.mark': {
                            '$all': [layer, f'E{m}', f'A{m}', f'D{m}', 'mt9', 'dt32', 'tw1', ds[0], 're1'],
                            '$in': [f'd{i}' for i in [2, 4, 6, 8, 10, 12, 14, 16]],
                        }},
                        {'paras.mark': {'$in': ['LinkPred', 'GraphDistor', 'HypernymyRel']}},
                        # {'paras.mark': {'$in': ['LinkPred']}},
                    ]}}])
                    tasks_num = len(tasks)
                    tasks = sorted(tasks, key=lambda t: tuple(t['paras']['mark']))  # 排序
                    指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(m)] for t in tasks]
                    method = f'{Manifold.s_to_tex(m)}-{line_labels[k]}'
                elif m == 2:
                    tasks = obj_train.que_tasks([{'$match': {'$and': [
                        {'paras.mark': {
                            '$all': [layer, f'E2', f'A2', f'D2', 'd2', 'mt9', 'dt3000', 'tw1', ds[0], 're1'],
                            '$in': ['LinkPred'],
                        }},
                    ]}}]) * tasks_num  # 一个任务结果
                    指标D_L = [t['result_all']['epoch']['0']['to_m'][str(m)] for t in tasks]
                    method = f'{line_labels[k]}'
                else:  # 其他流形的 comb 不需要
                    continue
                # friedman test
                for metric in metrics_F:
                    result_L = [指标D[metric][0] for 指标D in 指标D_L]
                    if method in method_result_D:
                        method_result_D[method] += result_L
                    else:
                        method_result_D[method] = result_L
                # radar, 如果需要comb可以放在 friedman test 后面
                line_data.append([])
                for x_label in x_labels:  # 每个指标的结果
                    result_L = [指标D[x_label][0] for 指标D in 指标D_L]  # 所有维度平均
                    line_data[-1].append(sum(result_L) / len(result_L))
            sub_title = f'({len(draw.already_drawn) + 1}): {Manifold.s_to_tex(m)}, {datasets[i]}'
            draw.add_radar([metric_tex_D[metric] for metric in x_labels],
                           line_labels[:len(line_data)], line_data, sub_title, fill_alpha=0.1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend='best', colors=colors)
    global no
    stop_strategy = f'{best_result[1]}-{best_result[0]}'
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}_{stop_strategy}.pdf')
    method_result_L = sorted(method_result_D.items())
    no += 1
    return {  # 用于 friedman_test
        'x': [i[1] for i in method_result_L],
        'yticks': [i[0] for i in method_result_L],
        'title_end': f', Stop strategy: {stop_strategy}'
    }


def hierarchical_structure_3d(best_result=('metric', 'dev')):
    """
    5个三维透视图: 36双指标可变树*5指标*二维*GCN*Poincare*LP
    1个雷达图: (4可变树+2固定树+4可变图)*(comb结果+3E流形)*GCN*LP*M5
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, '...')
    r = 1
    c = 5
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    get_指标D_L = lambda tasks: [  # 从tasks中提取所有指标结果, 使用encoder输出流形结果
        t['result_all']['epoch'][
            str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
        ]['to_m'][str(t['paras']['encoderParas']['manifold'])] for t in tasks]
    x_labels = ['Tree1', 'Tree2', 'Tree3  ', 'Tree4', 'Disease', 'Animal', ' Graph1', '   Graph2', ' Graph3', 'Graph4']
    spaces = ' ' * 40
    # 绘制3d需要的所有数据
    marks = get_obj_marks({'data_result': [['t6']]})
    tasks = obj_train.que_tasks([{'$match': {'$and': [
        {'paras.mark': {
            '$all': ['gcn', 'mt9', 'dt32', 'tw1', 're1'],
            '$in': [f'd{i}' for i in [2, 4, 6, 8, 10, 12, 14, 16]],
        }},
        {'paras.mark': {'$in': [j[mark_index_D['ds']] for j in marks]}},
        {'paras.mark': {'$in': ['LinkPred']}},
        {'$or': [
            # {'paras.mark': {'$all': ['E0', 'A0', 'D0']}},
            {'paras.mark': {'$all': ['E1', 'A1', 'D1']}},
            {'paras.mark': {'$all': ['E2', 'A2', 'D2']}},
        ]},
    ]}}])
    tasks = sorted(tasks, key=lambda t: t['paras']['mark'][mark_index_D['ds']])  # 按数据排序
    tasks_L = chunks(tasks, len(tasks) / len(marks))
    print('任务数量:', len(tasks), ', 数据点数量:', len(tasks_L))

    for i, 指标 in enumerate(['M1', 'M2', 'M3', 'M4', 'M7']):
        if 指标 == 'data':  # 绘制IB-ID的平面散点图
            tasks = obj_data.que_tasks({'paras': {'mark': ['t6']}})
            xyt_scatter = []  # [(IB,ID,'(IB;ID;H)'),..]
            for task in tasks:
                if len(task['graph_info']['每棵树层次度分布的偏移度']) == len(task['graph_info']['每棵树不平衡程度']) == 1:
                    ID = task['graph_info']['每棵树层次度分布的偏移度'][0]
                    IB = task['graph_info']['每棵树不平衡程度'][0]
                    H = task['graph_info']['每棵树的层数'][0]
                    xyt_scatter.append((IB, ID, f'({round(IB, 3)};{round(ID, 3)};{H})'))
            sub_title = f'({i + 1}): ($I_B$;$I_D$;$H$) for each dataset'
            draw.add_scatter([xyt_scatter], scatter_labels=['Tree'], scatter_c='black', n=i + 1, sub_title=sub_title,
                             sub_title_fs=None, xlabel='$I_B$', ylabel='$I_D$')
        elif 指标 == 'M5':  # 层次指标使用雷达图
            line_data = [[], [], [], []]
            line_labels = []
            for j, lm in enumerate([0, 1, 2, 'comb']):
                if lm == 'comb':
                    layer = lm
                    lm = 2
                    dtype = 3000
                    line_labels.append('Comb')
                else:
                    layer = 'gcn'
                    dtype = 32
                    line_labels.append(Manifold.s_to_tex(lm))
                # for k, data in enumerate(['t1', 't2', 't3', 't4', 'o2', 'o3', 'g1', 'g2', 'g3', 'g4']):
                for k, data in enumerate(['t1', 't2', 't3', 't4', 'o2', 'o3']):
                    if layer == 'comb' and k > 5:  # comb 方法没有图
                        line_data[j].append(0)
                        continue
                    mark = get_obj_marks(
                        {'dh_L': [['LinkPred']], 'layer': [layer], 'layerManifold': [lm],
                         'data_result': [[data]], 'dtype': [dtype]}, ['d'])[0]
                    tasks, manifold = get_dim_task(obj_train.que_tasks({'paras': {'mark': mark}}))[:2]  # 获得所有维度任务
                    try:
                        指标D_L = get_指标D_L(tasks)
                    except:
                        指标D_L = [t['result_all']['epoch']['0']['to_m']['2'] for t in tasks]
                    result_L = [指标D[指标][0] for 指标D in 指标D_L]
                    line_data[j].append(sum(result_L) / len(result_L))
            sub_title = f'({i + 1}): {metric_tex_D[指标]}, LP' + spaces * 2
            draw.add_radar(x_labels[:len(line_data[0])], line_labels, line_data, sub_title, fill_alpha=0.1, n=i + 1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend=(.9, .87), title_pad=-15)
        else:
            xyz_L = []
            for j, tasks in enumerate(tasks_L):
                指标D_L = get_指标D_L(tasks)
                ID = tasks[0]['graph_info']['每棵树层次度分布的偏移度'][0]  # 每组任务只是维度不同, ID一样
                IB = tasks[0]['graph_info']['每棵树不平衡程度'][0]
                result_L = [指标D[指标][0] for 指标D in 指标D_L]
                xyz_L.append((IB, ID, sum(result_L) / len(result_L)))
            sub_title = f'({i + 1}): Z axis is {metric_tex_D[指标]}'
            if 指标 == 'M5':
                azim = 45
            else:
                azim = None
            draw.add_3d(xyz_L, xyz_scatter=[xyz_L], x_multiple=5, y_multiple=5,
                        scatter_labels=['Tree'], interp_kind='linear', azim=azim,
                        xlabel='$I_B$', ylabel='$I_D$', zlabel='', sub_title=sub_title, n=i + 1)
    global no
    stop_strategy = f'{best_result[1]}-{best_result[0]}'
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}_{stop_strategy}.pdf')
    no += 1


def multi_hierarchical_structure_radar(best_result=('metric', 'dev')):
    """
    4个雷达图: (comb单树+Poincare*(单树+子树))*8子树*4指标*LP*GCN
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, '...')
    r = 1
    c = 5
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    # 该顺序需要和多树混合图中的子树顺序一致, 该顺序从生成数据的图片或IB/ID或['mixed_tree_order']中看出
    sub_trees = ['t5.1.1', 't5.1.2', 't5.1.3', 't5.1.4', 't5.2.1', 't5.2.2', 't5.2.3', 't5.2.4']
    sub_trees = [i + '-m1' for i in sub_trees]
    x_labels = ['$T_1$', '$T_2$', '$T_3$', '$T_4$', '$T_5$', '$T_6$', '$T_7$', '$T_8$']
    line_labels = ['Comb', 'Mix-tree', 'Sub-tree']  # 每个线的名称, 注意顺序与 line_data 一致
    # 用于 friedman test
    sub_tree_ftg_D = {  # 一个 sub_tree 对应于 friedman检验图的第几个, 和 paras_L 一致
        sub_trees[0]: 0, sub_trees[4]: 0,
        sub_trees[1]: 1, sub_trees[5]: 1,
        sub_trees[2]: 2, sub_trees[6]: 2,
        sub_trees[3]: 3, sub_trees[7]: 3,
    }
    paras_L = [{  # friedman检验图 的参数
        'x': [[] for i in range(len(x_labels))],
        # 算法 line_labels[1] 和 line_labels[2] 交替
        'yticks': [f'{j}.{i}' for i in [f"{x_labels[k]}{x_labels[k + 4]}" for k in range(4)] for j in line_labels[1:]],
        'title_end': ''
    }]
    print(paras_L[0]['yticks'])
    metrics_F = ['M1', 'M2', 'M3', 'M4']  # 用于 friedman test 的指标
    # 从tasks中提取所有指标结果, 使用encoder输出流形结果
    get_指标D_L = lambda tasks: [
        t['result_all']['epoch'][
            str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
        ]['to_m'][str(t['paras']['encoderParas']['manifold'])] for t in tasks]
    # 用于查询合并多少结果
    query_paras_L = [
        {'paras.mark': {'$in': ['LinkPred']}},
        {'$or': [
            # {'paras.mark': {'$all': ['E0', 'A0', 'D0']}},
            {'paras.mark': {'$all': ['E1', 'A1', 'D1']}},
            {'paras.mark': {'$all': ['E2', 'A2', 'D2']}},
        ]},
        {'paras.mark': {'$in': [f'd{i}' for i in [2, 4, 6, 8, 10, 12, 14, 16]]}},
    ]

    # 多树混合图流形结果
    tasks = obj_train.que_tasks(
        [{'$match': {'$and': [
                                 {'paras.mark': {'$all': ['gcn', 'mt9', 'dt32', 'tw1', 't5', 're1']}},
                             ] + query_paras_L}}])
    print('多树混合图流形结果数量:', len(tasks))
    task_mix = sorted(tasks, key=lambda t: tuple(t['paras']['mark']))  # 排序

    # 子树流形结果
    tasks = obj_train.que_tasks(
        [{'$match': {'$and': [
                                 {'paras.mark': {'$all': ['gcn', 'mt9', 'dt32', 'tw1', 're1']}},
                                 {'paras.mark': {'$in': sub_trees}},
                             ] + query_paras_L}}])
    tasks = sorted(tasks, key=lambda t: t['paras']['mark'][mark_index_D['ds']])  # 按数据排序, 排序后要能与sub_trees一致
    print('子树流形结果数量:', len(tasks))
    task_sub_L = chunks(tasks, len(tasks) / len(sub_trees))
    for i in range(len(task_sub_L)):  # 内部每个子树数据排序, 与 task_mix 顺序一致以便于 friedman test
        task_sub_L[i] = sorted(task_sub_L[i], key=lambda t: tuple(t['paras']['mark']))

    # 子树comb结果
    tasks = obj_train.que_tasks([{'$match': {'$and': [
        {'paras.mark': {
            '$all': ['comb', 'd2', 'mt9', 'dt3000', 'tw1', 're1'],
        }},
        {'paras.mark': {'$in': sub_trees}},
    ]}}])
    tasks = sorted(tasks, key=lambda t: t['paras']['mark'][mark_index_D['ds']])  # 按数据排序, 排序后要能与sub_trees一致
    print('子树comb结果数量:', len(tasks))  # 一个子数一个comb结果
    assert len(tasks) == len(sub_trees), f'{len(tasks)}!={len(sub_trees)} 可能影响 friedman test'
    task_sub_comb_L = chunks(tasks, len(tasks) / len(sub_trees))
    for i in range(len(task_sub_comb_L)):  # 扩展结果以便于 friedman test 保持数据一致
        task_sub_comb_L[i] = task_sub_comb_L[i] * len(task_mix)

    for i, 指标 in enumerate(['M1', 'M2', 'M3', 'M4', 'M7']):
        line_data = [[], [], []]  # 每个线的数据
        for k, (task_sub, task_sub_comb) in enumerate(zip(task_sub_L, task_sub_comb_L)):
            assert task_sub[0]['paras']['mark'][mark_index_D['ds']] == task_sub_comb[0]['paras']['mark'][
                mark_index_D['ds']] == sub_trees[k], '数据排序后与sub_trees顺序不一致!'
            # Comb
            指标D_L = [t['result_all']['epoch']['0']['to_m']['2'] for t in task_sub_comb]
            result_L = [指标D[指标][0] for 指标D in 指标D_L]
            line_data[0].append(sum(result_L) / len(result_L))
            # Mix-tree
            指标D_L = get_指标D_L(task_mix)
            result_L = [指标D[指标][k] for 指标D in 指标D_L]
            line_data[1].append(sum(result_L) / len(result_L))
            if 指标 in metrics_F:
                paras_L[0]['x'][sub_tree_ftg_D[sub_trees[k]] * 2 + 0] += result_L
            # Sub-tree
            指标D_L = get_指标D_L(task_sub)
            result_L = [指标D[指标][0] for 指标D in 指标D_L]
            line_data[2].append(sum(result_L) / len(result_L))
            if 指标 in metrics_F:
                paras_L[0]['x'][sub_tree_ftg_D[sub_trees[k]] * 2 + 1] += result_L
        sub_title = f'({i + 1}): {metric_tex_D[指标]}'
        draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i + 1,
                       radii=(0.2, 0.4, 0.6, 0.8), set_legend=(0.95, .9), title_pad=20)
    global no
    stop_strategy = f'{best_result[1]}-{best_result[0]}'
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}_{stop_strategy}.pdf')
    no += 1
    friedman_test(paras_L, 'multi_hierarchical_structure_friedman_' + stop_strategy)


def act_loss_heatmap(best_result=('metric', 'dev')):
    """
    4个数字热力图: (3D流形*2维度)*(3E流形*3A流形)*4指标*GCN*Animal*LP
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, '...')
    r = 2
    c = 2
    draw = Draw(length=c * 4, width=r * 5, r=r, c=c)
    data_result = ['o3']
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])

    for i0, 指标 in enumerate(['M1', 'M2', 'M3', 'M4']):
        mat = []
        yticks = []
        for i1, E in enumerate([0, 1, 2]):
            for i2, A in enumerate([0, 1, 2]):
                mat.append([])
                xticks = []
                for i3, D in enumerate([0, 1, 2]):
                    mark = get_obj_marks({'layerManifold': [E], 'actM_L': [A], 'manifold': [D],
                                          'data_result': [data_result]}, ['d'])[0]
                    tasks, m = get_dim_task(obj_train.que_tasks({'paras': {'mark': mark}}))[:2]  # 获得所有维度任务
                    指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(m)] for t in tasks]
                    result_L = [指标D[指标][0] for 指标D in 指标D_L]  # 所有维度平均
                    mat[-1].append(sum(result_L) / len(result_L))
                    xticks.append(f'D:{Manifold.s_to_tex(D)}')
                yticks.append(f'E:{Manifold.s_to_tex(E)},A:{Manifold.s_to_tex(A)}')
        sub_title = f'({i0 + 1}): {metric_tex_D[指标]}'
        draw.add_heatmap(mat, xticks, yticks, sub_title=sub_title, n=i0 + 1, x_rotation=90, mat_text=2)
    global no
    stop_strategy = f'{best_result[1]}-{best_result[0]}'
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}_{stop_strategy}.pdf')
    no += 1


def hierarchical_performance_line(best_result=('metric', 'dev')):
    """
    8个折线图: (4结合方式+不结合)*(4指标+1metric)*4任务*2公开树*Hyperboloid*GCN
    :param best_result: ('metrics/loss', 'test/dev/train')
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, '...')
    r = 4
    c = 2
    draw = Draw(length=c * 8, width=r * 4, r=r, c=c)
    layerManifold = 1
    query_f = lambda data_result, dh_L, mixedType: {  # 用于切换不同的树
        'layerManifold': [layerManifold], 'task_weight': [0.9], 'mixedType': [mixedType], 'dh_L': [dh_L],
        'data_result': [data_result]
    }
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
    tree_name_L = ['Disease', 'Animal']  # 2个数据集名字

    for i, data in enumerate(['o2', 'o3']):
        dh_name_L = ['NC', 'LP', 'GD', 'HR']  # 4个任务的名字
        yaxis_right_L = ['acc', 'acc', f'-{metric_tex_D["M6"]}', 'acc']  # 4个任务metric指标不同, 右侧1个line标签
        for j, dh_L in enumerate([['Classification', 'LinkPred'], ['LinkPred', 'GraphDistor'],
                                  ['GraphDistor', 'LinkPred'], ['HypernymyRel', 'GraphDistor']]):
            y_left = [[], [], [], []]  # 4个M指标
            y_right = []  # 1个metric指标
            ylabel_left = ['M1', 'M2', 'M3', 'M4']  # 左侧4个line的标签
            xticks = ['No', 'EfD', 'ED', 'EfED', 'L']  # x的坐标描述
            for i3, mixedType in enumerate([9, 3, 1, 2, 0]):  # x对应的y
                mark = get_obj_marks(query_f([data], dh_L, mixedType), ['d'])[0]
                tasks, m = get_dim_task(obj_train.que_tasks({'paras': {'mark': mark}}))[:2]  # 获得所有维度任务
                指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(m)] for t in tasks]
                for i4, 指标 in enumerate(ylabel_left):
                    result_L = [指标D[指标][0] for 指标D in 指标D_L]  # 所有维度平均
                    y_left[i4].append(sum(result_L) / len(result_L))
                best = [t['result_all']['best_result'][best_result[0]][best_result[1]]['test'][1] for t in tasks]
                y_right.append(sum(best) / len(best))
            sub_title = f'({j * c + i + 1}): {Manifold.s_to_tex(layerManifold)}, task={dh_name_L[j]}, dataset={tree_name_L[i]}'
            draw.add_line([1, 2, 3, 4, 5], xticks, xaxis='Hybrid training method', y_left=y_left, yaxis_left='M',
                          y_right=[y_right], yaxis_right=f'{yaxis_right_L[j]} (test set performance)',
                          ylabel_right=[yaxis_right_L[j]],
                          ylabel_left=[metric_tex_D[metric] for metric in ylabel_left],
                          title=sub_title, n=j * c + i + 1)
    global no
    stop_strategy = f'{best_result[1]}-{best_result[0]}'
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}_{stop_strategy}.pdf')
    no += 1


def hierarchical_performance_heatmap(best_epoch=600):
    """
    2个热力图: 7指标*(3metrics*4任务)*2公开树*Hyperboloid*GCN
    :param best_epoch: int; 取 best_epoch 之前的结果计算 spearmanr. 因为多个维度平均所以不能使用 best_result
    :return:
    """
    print('\n', sys._getframe().f_code.co_name, '...')
    r = 1
    c = 2
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    layerManifold = 1
    query_f = lambda data_result, dh_L: {  # 用于切换不同的树
        'layerManifold': [layerManifold], 'dh_L': [dh_L], 'data_result': [data_result]
    }
    avg_f = lambda x: sum(x) / len(x)
    tree_name_L = ['Disease', 'Animal']  # 2个数据集名字
    for i, data in enumerate(['o2', 'o3']):
        mat = []
        xticks = ['M1', 'M2', 'M3', 'M4', 'M6', 'M7']
        yticks = []
        dh_name_L = ['NC', 'LP', 'GD', 'HR']  # 4个任务的名字
        for j, dh_L in enumerate([['Classification'], ['LinkPred'], ['GraphDistor'], ['HypernymyRel']]):
            dataset_describe_L = ['train', 'dev', 'test']  # 数据集描述
            mark = get_obj_marks(query_f([data], dh_L), ['d'])[0]
            tasks, m = get_dim_task(obj_train.que_tasks({'paras': {'mark': mark}}))[:2]  # 获得所有维度任务
            epoch_L = [t['result_all']['epoch'] for t in tasks]
            epoch_result_D = {}  # {epoch:{'metrics':[M1,..],'performance':[train,dev,test]},..}
            for k in epoch_L[0].keys():  # 每个任务的 epochs 需要一样才能平均
                epoch_result_D[int(k)] = {
                    'metrics': [avg_f([e[k]['to_m'][str(m)][xtick][0] for e in epoch_L]) for xtick in xticks],
                    'performance': [avg_f([e[k]['dataset'][dd]['metric'] for e in epoch_L]) for dd in
                                    dataset_describe_L],
                }
            epoch_result_L = sorted(epoch_result_D.items(), key=lambda t: t[0])
            epoch_result_L = [er for er in epoch_result_L if er[0] <= best_epoch]
            # 整合所有epoch结果
            metrics = np.array([er[1]['metrics'] for er in epoch_result_L])  # epoch * len(xticks)
            performance = np.array([er[1]['performance'] for er in epoch_result_L])  # epoch * len(dataset_describe_L)
            # 计算 spearman
            for k, dataset_describe in enumerate(dataset_describe_L):
                yticks.append(f'{dh_name_L[j]}-{dataset_describe}')
                # len(xticks) 指标结果一组
                mat.append([stats.spearmanr(metrics[:, i1], performance[:, k])[0] for i1 in range(len(xticks))])
        sub_title = f'({i + 1}): {Manifold.s_to_tex(layerManifold)}, dataset={tree_name_L[i]}'
        draw.add_heatmap(mat, [metric_tex_D[metric] for metric in xticks],
                         yticks, sub_title=sub_title, n=i + 1, x_rotation=90, mat_text=2)
    global no
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}.pdf')
    no += 1


if __name__ == '__main__':
    friedman_paras_L = []
    friedman_paras_L.append(decoder_radar(('loss', 'dev')))
    friedman_paras_L.append(decoder_radar(('loss', 'train')))
    friedman_test(friedman_paras_L, 'decoder_friedman')

    friedman_paras_L = []
    friedman_paras_L.append(encoder_radar(('loss', 'dev')))
    friedman_paras_L.append(encoder_radar(('loss', 'train')))
    friedman_test(friedman_paras_L, 'encoder_friedman')

    hierarchical_structure_3d(('loss', 'train'))
    multi_hierarchical_structure_radar(('loss', 'train'))
    # act_loss_heatmap(('loss', 'train'))
    # hierarchical_performance_line(('loss', 'train'))
    # hierarchical_performance_heatmap(300)
