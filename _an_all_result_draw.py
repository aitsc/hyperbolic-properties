from _am_create_all_train import *
from tanshicheng import Draw
from scipy import stats

数据生成任务_obj = 数据生成任务('al_all_data')
训练生成任务_obj = 训练生成任务('am_all_train')


def 获得训练任务结果(query, 训练生成任务_obj=训练生成任务_obj, 数据生成任务_obj=数据生成任务_obj):
    paras_L, 初始参数D = 穷举构建简单任务方法(query, 数据生成任务_obj, {}, 训练生成任务_obj)[0:2]
    if len(paras_L) > 1:
        print('发现参数数量 =', len(paras_L), '; query =', query)
        tasks = []
        for paras in paras_L:
            tasks += 训练生成任务_obj.que_tasks({'paras': {'mark': paras['mark']}})
        print('获得任务数量:', len(tasks))
    else:
        tasks = 训练生成任务_obj.que_tasks({'paras': {'mark': paras_L[0]['mark']}})
    return (
        tasks[0],  # 第一个任务
        tasks[0]['paras']['encoderParas']['manifold'],  # 第一个任务 encoder 输出的流形
        初始参数D,
        tasks,  # 所有任务
    )


def decoder_radar(best_result=('metric', 'dev'), no=1):
    """
    6个雷达图: (4任务*5指标)*3E流形*2维度*GCN*Animal
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
    for i, dim in enumerate([2, 16]):
        for j, layerManifold in enumerate([0, 1, 2]):
            line_data = []
            for k, dh_L in enumerate([['Classification'], ['LinkPred'], ['GraphDistor'], ['HypernymyRel']]):
                task, manifold = 获得训练任务结果({'dh_L': [dh_L], 'layerManifold': [layerManifold], 'dim': [dim]})[:2]
                best_epoch = task['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
                指标D = task['result_all']['epoch'][str(best_epoch)]['to_m'][str(manifold)]
                line_data.append([指标D[x_label][0] for x_label in x_labels])
            sub_title = f'({i * c + j + 1}): {Manifold.s_to_tex(layerManifold)}, dim={dim}'
            draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i * c + j + 1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend='best')
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


def encoder_radar(best_result=('metric', 'dev'), no=1):
    """
    6个雷达图: (3方法*5指标)*3E流形*2维度*LP*Animal + comb方法, 欧式二维绘图结果
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 2
    c = 3
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    x_labels = ['M1', 'M2', 'M3', 'M4', 'M7']
    for i, dim in enumerate([2, 16]):
        for j, layerManifold in enumerate([0, 1, 2]):
            line_data = []
            line_labels = ['MLP', 'GCN', 'GAT']
            for k, layer in enumerate(['mlp', 'gcn', 'gat']):
                task, manifold, 初始参数D = 获得训练任务结果(
                    {'layer': [layer], 'layerManifold': [layerManifold], 'dim': [dim]})[:3]
                best_epoch = task['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
                指标D = task['result_all']['epoch'][str(best_epoch)]['to_m'][str(manifold)]
                line_data.append([指标D[x_label][0] for x_label in x_labels])
            if dim == 2 and layerManifold == 0:  # 欧式绘图方法
                指标D = 初始参数D['data_result'][0]['metrics']  # 默认 animal
                line_data.append([指标D[x_label][0] for x_label in x_labels])
                line_labels.append('Draw')
            if layerManifold == 2:  # comb方法
                指标D = 获得训练任务结果({'layer': ['comb'], 'dim': [dim], 'dtype': [3000]}
                               )[0]['result_all']['epoch']['0']['to_m']['2']
                line_data.append([指标D[x_label][0] for x_label in x_labels])
                line_labels.append('Comb')
            sub_title = f'({i * c + j + 1}): {Manifold.s_to_tex(layerManifold)}, dim={dim}'
            draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i * c + j + 1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend='best')
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


def hierarchical_structure_3d(best_result=('metric', 'dev'), no=1):
    """
    6个三维透视图: 36双指标可变树*6指标*二维*GCN*Poincare*LP
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 2
    c = 3
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    dim = 2
    layerManifold = 2
    query_f = lambda data_result: {  # 用于切换不同的树
        'dh_L': [['LinkPred']], 'layer': ['gcn'], 'layerManifold': [layerManifold], 'dim': [dim],
        'data_result': [data_result]
    }
    tasks = 获得训练任务结果(query_f(['t6']))[3]
    tasks_no_surf = []
    # tasks_no_surf = [获得训练任务结果(query_f([[i]]))[0] for i in ['t1', 't2', 't3', 't4']]
    scatter_labels = ['Trees', 'Tree1', 'Tree2', 'Tree3', 'Tree4']
    for i, 指标 in enumerate(['M1', 'M2', 'M3', 'M4', 'M5', 'M7']):
        xyz_L = []
        xyz_scatter = [xyz_L, [], [], [], []]
        for j, task in enumerate(tasks_no_surf + tasks):
            best_epoch = task['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
            指标D = task['result_all']['epoch'][str(best_epoch)]['to_m'][f'{layerManifold}']
            ID = task['graph_info']['每棵树层次度分布的偏移度'][0]
            IB = task['graph_info']['每棵树不平衡程度'][0]
            if j < len(tasks_no_surf):
                xyz_scatter[j + 1].append((IB, ID, 指标D[指标][0]))
            else:
                xyz_L.append((IB, ID, 指标D[指标][0]))
        m_name = Manifold.s_to_tex(layerManifold)
        sub_title = f'({i + 1}): {m_name}, dim={dim}, Z={指标}'
        draw.add_3d(xyz_L, xyz_scatter=xyz_scatter[:len(tasks_no_surf) + 1], x_multiple=5, y_multiple=5,
                    scatter_labels=scatter_labels[:len(tasks_no_surf) + 1], interp_kind='linear',
                    xlabel='$I_B$', ylabel='$I_D$', zlabel='', sub_title=sub_title, n=i + 1)
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


def hierarchical_structure_radar(best_result=('metric', 'dev'), no=1):
    """
    2个雷达图: (4可变树+2固定树+4可变图)*(comb结果+3E流形)*2维度*GCN*LP*M5
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 1
    c = 2
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    x_labels = ['Tree1', 'Tree2', 'Tree3', 'Tree4', 'Disease', 'Animal', 'Graph1', 'Graph2', 'Graph3', 'Graph4']
    query_f = lambda data_result, layerManifold, dim=2, layer='gcn', dtype=32: {  # 用于切换不同的树
        'dh_L': [['LinkPred']], 'layer': [layer], 'layerManifold': [layerManifold], 'dim': [dim],
        'data_result': [data_result], 'dtype': [dtype]
    }
    for i, dim in enumerate([2, 16]):
        line_data = [[], [], [], []]
        line_labels = []
        for j, layerManifold in enumerate([0, 1, 2, 'comb']):
            if layerManifold == 'comb':
                layer = layerManifold
                layerManifold = 2
                dtype = 3000
                line_labels.append('Comb')
            else:
                layer = 'gcn'
                dtype = 32
                line_labels.append(Manifold.s_to_tex(layerManifold))
            for k, data in enumerate(['t1', 't2', 't3', 't4', 'o2', 'o3', 'g1', 'g2', 'g3', 'g4']):
                if layer == 'comb' and k > 5:  # comb 方法没有图
                    line_data[j].append(0)
                    continue
                task = 获得训练任务结果(query_f([data], layerManifold, dim, layer, dtype))[0]
                try:
                    best_epoch = task['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
                    指标D = task['result_all']['epoch'][str(best_epoch)]['to_m'][str(layerManifold)]
                except:
                    指标D = task['result_all']['epoch']['0']['to_m']['2']
                line_data[j].append(指标D['M5'][0])
        sub_title = f'({i + 1}): dim={dim}'
        draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i + 1,
                       radii=(0.2, 0.4, 0.6, 0.8), set_legend=(0.95, .9), title_pad=20)
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


def act_loss_heatmap(best_result=('metric', 'dev'), no=1):
    """
    4个数字热力图: (3D流形*2维度)*(3E流形*3A流形)*4指标*GCN*Animal*LP
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 2
    c = 2
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    query_f = lambda D, E, A, dim=2: {  # 用于切换不同的树
        'layerManifold': [E], 'dim': [dim], 'actM_L': [A], 'manifold': [D]
    }
    for i0, 指标 in enumerate(['M1', 'M2', 'M3', 'M4']):
        mat = []
        yticks = []
        for i1, E in enumerate([0, 1, 2]):
            for i2, A in enumerate([0, 1, 2]):
                mat.append([])
                xticks = []
                for i3, dim in enumerate([2, 16]):
                    for i4, D in enumerate([0, 1, 2]):
                        task, m = 获得训练任务结果(query_f(D, E, A, dim))[:2]
                        best_epoch = task['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
                        指标D = task['result_all']['epoch'][str(best_epoch)]['to_m'][str(m)]
                        mat[-1].append(指标D[指标][0])
                        xticks.append(f'dim:{dim},D:{Manifold.s_to_tex(D)}')
                yticks.append(f'E:{Manifold.s_to_tex(E)},A:{Manifold.s_to_tex(A)}')
        sub_title = f'({i0 + 1}): {指标}'
        draw.add_heatmap(mat, xticks, yticks, sub_title=sub_title, n=i0 + 1, x_rotation=90, mat_text=2)
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


def hierarchical_performance_line(best_result=('metric', 'dev'), no=1):
    """
    8个折线图: (4结合方式+不结合)*(4指标+1metric)*4任务*2公开树*Hyperboloid*GCN*二维
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 4
    c = 2
    draw = Draw(length=c * 8, width=r * 4, r=r, c=c)
    layerManifold = 1
    dim = 2
    query_f = lambda data_result, dh_L, mixedType: {  # 用于切换不同的树
        'layerManifold': [layerManifold], 'task_weight': [0.9], 'mixedType': [mixedType], 'dh_L': [dh_L],
        'data_result': [data_result], 'dim': [dim]
    }
    tree_name_L = ['Disease', 'Animal']  # 2个数据集名字
    for i, data in enumerate(['o2', 'o3']):
        dh_name_L = ['NC', 'LP', 'GD', 'HR']  # 4个任务的名字
        yaxis_right_L = ['acc', 'acc', '-M6', 'acc']  # 4个任务metric指标不同, 右侧1个line标签
        for j, dh_L in enumerate([['Classification', 'LinkPred'], ['LinkPred', 'GraphDistor'],
                                  ['GraphDistor', 'LinkPred'], ['HypernymyRel', 'GraphDistor']]):
            y_left = [[], [], [], []]  # 4个M指标
            y_right = []  # 1个metric指标
            ylabel_left = ['M1', 'M2', 'M3', 'M4']  # 左侧4个line的标签
            xticks = ['No', 'EfD', 'ED', 'EfED', 'L']  # x的坐标描述
            for i3, mixedType in enumerate([9, 3, 1, 2, 0]):  # x对应的y
                task, m = 获得训练任务结果(query_f([data], dh_L, mixedType))[:2]
                best = task['result_all']['best_result'][best_result[0]][best_result[1]]
                指标D = task['result_all']['epoch'][str(best['epoch'])]['to_m'][str(m)]
                for i4, 指标 in enumerate(ylabel_left):
                    y_left[i4].append(指标D[指标][0])
                y_right.append(best['test'][1])
            sub_title = f'({j * c + i + 1}): {Manifold.s_to_tex(layerManifold)}, task={dh_name_L[j]}, dataset={tree_name_L[i]}'
            draw.add_line([1, 2, 3, 4, 5], xticks, xaxis='Hybrid training method', y_left=y_left, yaxis_left='M',
                          y_right=[y_right], yaxis_right=f'{yaxis_right_L[j]} (test set performance)',
                          ylabel_right=[yaxis_right_L[j]], ylabel_left=ylabel_left, title=sub_title, n=j * c + i + 1)
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


def hierarchical_performance_heatmap(best_result=('metric', 'dev'), no=1):
    """
    2个热力图: 7指标*(3metrics*4任务)*2公开树*Hyperboloid*GCN*二维
    :param best_result: None or ('metrics/loss', 'test/dev/train'); None表示全取
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 1
    c = 2
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    layerManifold = 1
    dim = 2
    query_f = lambda data_result, dh_L: {  # 用于切换不同的树
        'layerManifold': [layerManifold], 'dh_L': [dh_L], 'dim': [dim], 'data_result': [data_result]
    }
    tree_name_L = ['Disease', 'Animal']  # 2个数据集名字
    for i, data in enumerate(['o2', 'o3']):
        mat = []
        xticks = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']
        yticks = []
        dh_name_L = ['NC', 'LP', 'GD', 'HR']  # 4个任务的名字
        for j, dh_L in enumerate([['Classification'], ['LinkPred'], ['GraphDistor'], ['HypernymyRel']]):
            dataset_describe_L = ['train', 'dev', 'test']  # 数据集描述
            task, m = 获得训练任务结果(query_f([data], dh_L))[:2]
            epoch_result_D = {int(k): {  # {epoch:{'metrics':[M1,..],'performance':[train,dev,test]},..}
                'metrics': [v['to_m'][str(m)][xtick][0] for xtick in xticks],
                'performance': [v['dataset'][dd]['metric'] for dd in dataset_describe_L],
            } for k, v in task['result_all']['epoch'].items()}
            epoch_result_L = sorted(epoch_result_D.items(), key=lambda t: t[0])
            # 提取最好结果epoch之前的结果
            if best_result:
                best_epoch = task['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
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
        draw.add_heatmap(mat, xticks, yticks, sub_title=sub_title, n=i + 1, x_rotation=90, mat_text=2)
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


def multi_hierarchical_structure_radar(best_result=('metric', 'dev'), no=1):
    """
    8个雷达图: (comb单树+Poincare*(单树+子树))*8子树*4指标*2维度*LP*GCN
    :param best_result: ('metrics/loss', 'test/dev/train')
    :param no: int; 序号
    :return:
    """
    print(sys._getframe().f_code.co_name, '...')
    r = 4
    c = 2
    layerManifold = 2
    draw = Draw(length=c * 5, width=r * 5, r=r, c=c)
    # 该顺序需要和多树混合图中的子树顺序一致, 该顺序从生成数据的图片或IB/ID或['mixed_tree_order']中看出
    sub_trees = [['t5.1.1'], ['t5.1.2'], ['t5.1.3'], ['t5.1.4'], ['t5.2.1'], ['t5.2.2'], ['t5.2.3'], ['t5.2.4']]
    x_labels = ['$T_1$', '$T_2$', '$T_3$', '$T_4$', '$T_5$', '$T_6$', '$T_7$', '$T_8$']
    query_f = lambda data_result, dim=2, layer='gcn', dtype=32: {  # 用于切换不同的树
        'dh_L': [['LinkPred']], 'layer': [layer], 'layerManifold': [layerManifold], 'dim': [dim],
        'data_result': [data_result], 'dtype': [dtype]
    }
    for j, dim in enumerate([2, 16]):
        task_mix = 获得训练任务结果(query_f(['t5'], dim))[0]  # 多树混合图流形结果
        task_sub_L = [获得训练任务结果(query_f(i1, dim))[0] for i1 in sub_trees]  # 子树流形结果
        task_sub_comb_L = [获得训练任务结果(query_f(i1, dim, 'comb', 3000))[0] for i1 in sub_trees]  # 子树comb结果
        for i, 指标 in enumerate(['M1', 'M2', 'M3', 'M4']):
            line_data = [[], [], []]  # 每个线的数据
            line_labels = ['Comb', 'Mix-tree', 'Sub-tree']  # 每个线的名称, 注意顺序
            for k, (task_sub, task_sub_comb) in enumerate(zip(task_sub_L, task_sub_comb_L)):
                # Comb
                指标D = task_sub_comb['result_all']['epoch']['0']['to_m']['2']
                line_data[0].append(指标D[指标][0])
                # Mix-tree
                best_epoch = task_mix['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
                指标D = task_mix['result_all']['epoch'][str(best_epoch)]['to_m'][str(layerManifold)]
                line_data[1].append(指标D[指标][k])
                # Sub-tree
                best_epoch = task_sub['result_all']['best_result'][best_result[0]][best_result[1]]['epoch']
                指标D = task_sub['result_all']['epoch'][str(best_epoch)]['to_m'][str(layerManifold)]
                line_data[2].append(指标D[指标][0])
            sub_title = f'({i * c + j + 1}): {Manifold.s_to_tex(layerManifold)}, {指标}, dim={dim}'
            draw.add_radar(x_labels, line_labels, line_data, sub_title, fill_alpha=0.1, n=i * c + j + 1,
                           radii=(0.2, 0.4, 0.6, 0.8), set_legend=(0.95, .9), title_pad=20)
    draw.draw(f'an_{no}_{sys._getframe().f_code.co_name}.pdf')


if __name__ == '__main__':
    best_result = ('loss', 'train')
    no = 1
    decoder_radar(best_result, no)
    no += 1
    encoder_radar(best_result, no)
    no += 1
    hierarchical_structure_3d(best_result, no)
    no += 1
    hierarchical_structure_radar(best_result, no)
    no += 1
    act_loss_heatmap(best_result, no)
    no += 1
    hierarchical_performance_line(best_result, no)
    no += 1
    hierarchical_performance_heatmap(best_result, no)
    no += 1
    multi_hierarchical_structure_radar(best_result, no)
