from _am_create_all_train import *
from tanshicheng import Draw

obj_data = 数据生成任务('al_all_data')
obj_train = 训练生成任务('am_all_train', mongo_url=mongo_url)


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
    assert len(tasks_) == len(tasks), f'这组任务缺少一些维度: {dim_S - dim_S_}'
    return tasks_, \
           tasks_[0]['paras']['encoderParas']['manifold']  # encoder 输出的流形


def decoder_radar(best_result=('metric', 'dev'), no=1):
    """
    6个雷达图: (4任务*5指标)*3E流形*GCN*(Animal+Disease)
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 2
    c = 3
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    x_labels = ['M1', 'M2', 'M3', 'M4', 'M7']
    line_labels = ['NC', 'LP', 'GD', 'HR']
    datasets = ['Animal', 'Disease']  # 与 ds 对应
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])

    for i, ds in enumerate([['o3'], ['o2']]):
        for j, layerManifold in enumerate([0, 1, 2]):
            line_data = []
            for k, dh_L in enumerate([['Classification'], ['LinkPred'], ['GraphDistor'], ['HypernymyRel']]):
                mark = get_obj_marks({'dh_L': [dh_L], 'layerManifold': [layerManifold], 'data_result': [ds]}, ['d'])[0]
                tasks, manifold = get_dim_task(obj_train.que_tasks({'paras': {'mark': mark}}))[:2]  # 获得所有维度任务
                指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(manifold)] for t in tasks]
                line_data.append([])
                for x_label in x_labels:  # 每个指标的结果
                    result_L = [指标D[x_label][0] for 指标D in 指标D_L]  # 所有维度平均
                    line_data[-1].append(sum(result_L) / len(result_L))
            sub_title = f'({i * c + j + 1}): {Manifold.s_to_tex(layerManifold)}, {datasets[i]}'
            draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i * c + j + 1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend='best')
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}.pdf')


def encoder_radar(best_result=('metric', 'dev'), no=1):
    """
    6个雷达图: (3方法*5指标)*3E流形*LP*(Animal+Disease)
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 2
    c = 3
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    x_labels = ['M1', 'M2', 'M3', 'M4', 'M7']
    line_labels = ['MLP', 'GCN', 'GAT']
    datasets = ['Animal', 'Disease']  # 与 ds 对应
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])

    for i, ds in enumerate([['o3'], ['o2']]):
        for j, layerManifold in enumerate([0, 1, 2]):
            line_data = []
            for k, layer in enumerate(['mlp', 'gcn', 'gat']):
                mark = get_obj_marks({'layer': [layer], 'layerManifold': [layerManifold], 'data_result': [ds]},
                                     ['d'])[0]
                tasks, manifold = get_dim_task(obj_train.que_tasks({'paras': {'mark': mark}}))[:2]  # 获得所有维度任务
                指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(manifold)] for t in tasks]
                line_data.append([])
                for x_label in x_labels:  # 每个指标的结果
                    result_L = [指标D[x_label][0] for 指标D in 指标D_L]  # 所有维度平均
                    line_data[-1].append(sum(result_L) / len(result_L))
            sub_title = f'({i * c + j + 1}): {Manifold.s_to_tex(layerManifold)}, {datasets[i]}'
            draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i * c + j + 1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend='best')
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}.pdf')


def hierarchical_structure_3d(best_result=('metric', 'dev'), no=1):
    """
    5个三维透视图: 36双指标可变树*5指标*二维*GCN*Poincare*LP
    1个雷达图: (4可变树+2固定树+4可变图)*(comb结果+3E流形)*GCN*LP*M5
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 2
    c = 3
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    layerManifold = 2
    query_f = lambda data_result: {  # 用于切换不同的树
        'dh_L': [['LinkPred']], 'layer': ['gcn'], 'layerManifold': [layerManifold], 'data_result': [data_result]
    }
    marks = get_obj_marks(query_f(['t6']), ['d'])
    tasks_L = [get_dim_task(obj_train.que_tasks({'paras': {'mark': m}}))[0] for m in marks]
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])
    x_labels = ['Tree1', 'Tree2', 'Tree3', 'Tree4', 'Disease', 'Animal', 'Graph1', 'Graph2', 'Graph3', 'Graph4']

    for i, 指标 in enumerate(['M1', 'M2', 'M3', 'M4', 'M5', 'M7']):
        if 指标 == 'M5':  # 层次指标使用雷达图
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
                for k, data in enumerate(['t1', 't2', 't3', 't4', 'o2', 'o3', 'g1', 'g2', 'g3', 'g4']):
                    if layer == 'comb' and k > 5:  # comb 方法没有图
                        line_data[j].append(0)
                        continue
                    mark = get_obj_marks(
                        {'dh_L': [['LinkPred']], 'layer': [layer], 'layerManifold': [lm],
                         'data_result': [[data]], 'dtype': [dtype]}, ['d'])[0]
                    tasks, manifold = get_dim_task(obj_train.que_tasks({'paras': {'mark': mark}}))[:2]  # 获得所有维度任务
                    try:
                        指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(lm)] for t in tasks]
                    except:
                        指标D_L = [t['result_all']['epoch']['0']['to_m']['2'] for t in tasks]
                    result_L = [指标D[指标][0] for 指标D in 指标D_L]
                    line_data[j].append(sum(result_L) / len(result_L))
            sub_title = f'({i + 1}): {指标}'
            draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i + 1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend=(0.95, .9), title_pad=20)
        else:
            xyz_L = []
            for j, tasks in enumerate(tasks_L):
                指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(layerManifold)] for t in tasks]
                ID = tasks[0]['graph_info']['每棵树层次度分布的偏移度'][0]  # 每组任务只是维度不同, ID一样
                IB = tasks[0]['graph_info']['每棵树不平衡程度'][0]
                result_L = [指标D[指标][0] for 指标D in 指标D_L]
                xyz_L.append((IB, ID, sum(result_L) / len(result_L)))
            m_name = Manifold.s_to_tex(layerManifold)
            sub_title = f'({i + 1}): {m_name}, Z={指标}'
            draw.add_3d(xyz_L, xyz_scatter=[xyz_L], x_multiple=5, y_multiple=5,
                        scatter_labels=['Trees'], interp_kind='linear',
                        xlabel='$I_B$', ylabel='$I_D$', zlabel='', sub_title=sub_title, n=i + 1)
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}.pdf')


def multi_hierarchical_structure_radar(best_result=('metric', 'dev'), no=1):
    """
    4个雷达图: (comb单树+Poincare*(单树+子树))*8子树*4指标*LP*GCN
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 2
    c = 2
    layerManifold = 2
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    # 该顺序需要和多树混合图中的子树顺序一致, 该顺序从生成数据的图片或IB/ID或['mixed_tree_order']中看出
    sub_trees = [['t5.1.1'], ['t5.1.2'], ['t5.1.3'], ['t5.1.4'], ['t5.2.1'], ['t5.2.2'], ['t5.2.3'], ['t5.2.4']]
    x_labels = ['$T_1$', '$T_2$', '$T_3$', '$T_4$', '$T_5$', '$T_6$', '$T_7$', '$T_8$']
    query_f = lambda data_result, layer='gcn', dtype=32: {  # 用于切换不同的树
        'dh_L': [['LinkPred']], 'layer': [layer], 'layerManifold': [layerManifold],
        'data_result': [data_result], 'dtype': [dtype]
    }
    # 多树混合图流形结果
    task_mix = get_dim_task(obj_train.que_tasks({'paras': {'mark': get_obj_marks(query_f(['t5']), ['d'])[0]}}))[0]
    # 子树流形结果
    task_sub_L = [get_dim_task(obj_train.que_tasks({'paras': {'mark': get_obj_marks(query_f(i1), ['d'])[0]}}))[0]
                  for i1 in sub_trees]
    # 子树comb结果
    task_sub_comb_L = [
        get_dim_task(obj_train.que_tasks({'paras': {'mark': get_obj_marks(query_f(i1, 'comb', 3000), ['d'])[0]}}))[0]
        for i1 in sub_trees]
    best_epoch_f = lambda t: str(t['result_all']['best_result'][best_result[0]][best_result[1]]['epoch'])

    for i, 指标 in enumerate(['M1', 'M2', 'M3', 'M4']):
        line_data = [[], [], []]  # 每个线的数据
        line_labels = ['Comb', 'Mix-tree', 'Sub-tree']  # 每个线的名称, 注意顺序
        for k, (task_sub, task_sub_comb) in enumerate(zip(task_sub_L, task_sub_comb_L)):
            # Comb
            指标D_L = [t['result_all']['epoch']['0']['to_m']['2'] for t in task_sub_comb]
            result_L = [指标D[指标][0] for 指标D in 指标D_L]
            line_data[0].append(sum(result_L) / len(result_L))
            # Mix-tree
            指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(layerManifold)] for t in task_mix]
            result_L = [指标D[指标][k] for 指标D in 指标D_L]
            line_data[1].append(sum(result_L) / len(result_L))
            # Sub-tree
            指标D_L = [t['result_all']['epoch'][best_epoch_f(t)]['to_m'][str(layerManifold)] for t in task_sub]
            result_L = [指标D[指标][0] for 指标D in 指标D_L]
            line_data[2].append(sum(result_L) / len(result_L))
        sub_title = f'({i + 1}): {Manifold.s_to_tex(layerManifold)}, {指标}'
        draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i + 1,
                       radii=(0.2, 0.4, 0.6, 0.8), set_legend=(0.95, .9), title_pad=20)
    draw.draw(f'ap_{no}_{sys._getframe().f_code.co_name}.pdf')


if __name__ == '__main__':
    no = 1
    decoder_radar(('loss', 'dev'), no)
    no += 1
    encoder_radar(('loss', 'train'), no)
    no += 1
    hierarchical_structure_3d(('loss', 'train'), no)
    no += 1
    multi_hierarchical_structure_radar(('loss', 'train'), no)
