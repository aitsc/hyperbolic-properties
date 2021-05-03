from _al_create_all_data import 数据生成任务
from _af_train import *
from _ak_sala2018comb import *
from tanshicheng import TaskDB, get_logger
from tinydb import where
from pprint import pformat

logger = get_logger(f'log/{os.path.split(__file__)[1]}.log')


class 训练生成任务(TaskDB):
    @property
    def result(self):
        result = super().result
        result.update({
            # 待执行任务参数模版
            'paras': {
                'comb': {  # 这个参数方法优先考虑
                    'RG': '',  # 随机图存储路径, 运行时需要 dataset_db_path
                    'dh': '',  # dh存储路径, 运行时需要 dataset_db_path
                    'dtype': 32,  # 精度
                    'dim': 2,  # 维度
                    'julia': '/root/soft/julia/julia-0.7.0/bin/julia',  # julia executable 路径
                    'comb_path': '/root/git/paper/hyperbolics-julia-0.7',  # Combinatorial Constructions 算法文件夹路径
                },
                'trainParas': {
                    # 使用的任务
                    'dh_L': ['LinkPred'],
                    # dh_L: ['Classification'],
                    # dh_L: ['GraphDistor'],
                    # dh_L: ['HypernymyRel'],
                    # dh_L: ['Classification', 'LinkPred'],
                    # dh_L: ['Classification', 'LinkPred', 'GraphDistor', 'HypernymyRel'],

                    'ap': 'self._dir/am_graph/',  # 图片等等输出的主目录, 这个变量在run_task时自动生成
                    'dh_path': 'ac_test.pkl',  # DataHelper 的 load_file 数据路径, 运行时需要 dataset_db_path
                    'pictureFromat': 'pdf',  # 所有静态图片的格式, eps会导致gif不清晰但空间小, pdf会无法生成gif
                    'epochs': 600,  # 最多训练多少轮
                    'epoch_out': 10,  # 多少轮评估一次
                    'tiems_eval_draw': 10,  # 评估多少次进行一次可视化绘图
                    'stop_strategy': {
                        'devLossStop': 0,  # 至少多少轮dev数据集的loss不再降低就跳出训练
                        'devMetricStop': 0,  # epochs至少多少次后数据集的对应指标不再降低就跳出训练
                        'trainLossStop': 0,
                        'trainMetricStop': 0,
                        'testLossStop': 0,
                        'testMetricStop': 0,
                    },
                    'mixedType': 0,  # 多个dh时的混合方式: 0=损失加权求和, 1=E参数不固定, 2=E参数固定-新增E, 3=E参数固定
                },
                # train通过datahelper获取数据集的参数
                'dataParas': {
                    'devRate': 0.1,  # 验证集的比例
                    'testRate': 0.1,  # 测试集的比例
                    'shuffle': False,  # 是否每次都随机划分数据集
                    'dsMult': None,  # GraphDistor 边采样倍数
                    '等类别': True,  # Classification 是否保证 train/dev/test 中每个分类的节点数量相等
                    'negNum': 10,  # HypernymyRel 负例边是正例边的几倍
                    'shuffle_neg': True,  # 每轮训练是否打乱负例
                },
                # decoder模型其他参数
                'decoderParas': {
                    'task_weight': {},  # Decoder_mixedLoss 中不同损失函数的权重
                    'compute_all_dist': True,  # 是否在图失真模型中一次计算好所有点间距离
                    'learning_rate': 0.01,  # 学习率
                },
                # encoder参数
                'encoderParas': {
                    'manifold': 2,  # encoder 输出的流形
                    'encoderOutDim': (100, 2),  # encoder 每一层的输出维度
                    'layer': 'gat',  # encoder 层的神经网络, 参考 HyperLayer 中的方法名
                    'layerManifold': None,  # encoder 层的流形, None表示与manifold一致
                    'inM': 0,  # 输入encoder的初始特征流形
                    'layer_paras': {  # layer 的额外参数, 参考 HyperLayer 对应参数解释
                        'actM_L': 1,  # 每一层激活函数在什么流形空间上转换
                        'act_L': None,  # 每一层使用什么激活函数
                        'head_L': None,  # 每一层 gat 使用几个注意力头, 为None表示全部使用4个, 为int表示每层都是int
                    },
                    'compute_all_dist': True,  # 是否在encoder距离模型中一次计算好所有点间距离
                },
                'mark': ['111'],  # 分类标记, 用于过滤查询, 会放在文件夹名中
            },
            # 结果
            'time_start': 1617292720.1552498,  # 任务开始执行时间戳 time.time(), 会放在文件夹名中
            'graph_info': {'describe': 'wordnet.mammal', '树数量': 1, '每棵树节点数量': [1182], '每棵树的层数': [10],
                           '每棵树不平衡程度': [0.055837372325049164], '每棵树层次度分布的偏移度': [1.1622529654407538],
                           '树总节点数量': 1182, 'nx图节点数量': 0, 'nx图边数量': 0},
            'result_all': {  # 全部见 main_api
                'epoch': {
                    '0': {
                        'to_m': {
                            '2': {  # comb 方法单独一个
                                'M1': [1.0],
                                'M2': [1.0],
                                'M3': [1.0],
                                'M4': [1.0],
                                'M5': [1.0],
                                'M6': [1.0],
                                'M7': [1.0],
                                'd_mean': [1.0],
                                'd_std': [1.0],
                                'd_max': [1.0],
                            },
                            # ...
                        },
                        # ...
                    },
                    # ...
                },
                # ...
                'best_result': None,
                'dh_graph_info': {},  # dataHelper.data['图统计信息']
            },
        })
        return result

    def run_task(self, dataset_db_path):
        """
        运行任务
        :param dataset_db_path: 数据生成任务_obj 的位置, 用于寻找数据. 任务执行依赖其他 TaskDB
        :return:
        """
        tasks = self.get_uncomplete_tasks()
        完成任务 = 1
        while len(tasks) != 0:
            task = tasks.pop(0)
            paras = copy.deepcopy(task['paras'])  # 后续可能需要修改
            # 增加数据集db位置
            if 'comb' in paras and paras['comb']['RG'] and paras['comb']['dh']:
                paras['comb']['RG'] = f"{dataset_db_path}/" + paras['comb']['RG']
                paras['comb']['dh'] = f"{dataset_db_path}/" + paras['comb']['dh']
            if 'trainParas' in paras and paras['trainParas']['dh_path']:
                paras['trainParas']['dh_path'] = f"{dataset_db_path}/" + paras['trainParas']['dh_path']
            # 提示
            print('=' * 20, f'本次任务({完成任务}/{len(tasks)})参数({datetime.datetime.now()}):')
            pprint(paras)
            logger.critical(f'开始任务({完成任务}/{len(tasks)})参数:\n' + pformat(paras))
            # 结果过滤
            result = {}
            filter = task['filter']
            if not filter:  # 防止参数输入错误
                filter = {'test123': None}
            while self.filtrate_result(result, **filter):
                # 文件夹名
                time_start = time.time()
                folder_name_ = f"{'_'.join(paras['mark'])};{time_start}"
                folder_name = f"{self.db_dir}/{folder_name_}/"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                # 构建结果参数
                is_gpu = None
                if 'comb' in paras and os.path.exists(paras['comb']['RG']) and os.path.exists(paras['comb']['dh']):
                    RG = 随机图(paras['comb']['RG'])
                    dataHelper = DataHelper(load_file=paras['comb']['dh'])
                    metrics = 自动评估绘图(RG, dataHelper, f'{folder_name}/_.eps', **paras['comb'])[0]
                    result = {
                        'epoch': {'0': {'to_m': {'2': metrics_to_results(metrics)}}},
                        'dh_graph_info': dataHelper.data['图统计信息'],
                        'best_result': None,
                    }
                    is_gpu = False
                else:
                    paras['trainParas']['ap'] = folder_name
                    try:
                        result = main_api(**paras)
                    except:
                        logger.error('尝试使用cpu运行 main_api', exc_info=True)
                        with tf.device('/cpu:0'):
                            result = main_api(**paras)
                        logger.critical('使用cpu运行 main_api 成功!')
                        is_gpu = False
                result = {
                    'executed': True,
                    'main_path': folder_name_,
                    'machine': self.get_machine(is_gpu=is_gpu),
                    'graph_info': result['dh_graph_info'],
                    'time_start': time_start,
                    'result_all': result,
                }
            logger.critical(f'完成任务({完成任务}/{len(tasks)})结果:\ngraph_info:\n' +
                            pformat(result['graph_info']) + '\nbest_result:\n' +
                            pformat(result['result_all']['best_result']))
            # 更新任务
            self.update_one(result, task['no'])
            print('=' * 20, '本次任务结果:')
            pprint(result)
            print('=' * 20, f'已完成第{完成任务}个任务, 剩余{len(tasks)}个, 本次耗时{(time.time() - time_start) / 60}分钟.')
            完成任务 += 1
            print()

    def 统计结果(self):
        tasks = self._db_tasks.search(where('executed') == True)
        print('已完成任务:')
        pprint(tasks)
        print('已完成任务数:', len(tasks), '; 未完成任务数:', len(self._db_tasks.all()) - len(tasks))


def 穷举构建简单任务方法(初始参数D, obj: 数据生成任务, mark_re_D, 允许重复mark=True, 训练生成任务_obj: 训练生成任务 = None):
    """
    :param 初始参数D: dict;
    :param obj: 数据生成任务
    :param mark_re_D: {'mark去重复re后用_拼接':这个mark的重复数量,..}; 用于重复mark标记
    :param 允许重复mark: bool; 是否添加在mark_re_D中重复的mark
    :param 训练生成任务_obj: 训练生成任务, 用于获取参数模版
    :return:
    """
    初始参数D_ = {
        'dh_L': [['LinkPred']],  # decoder方法
        'layer': ['gcn'],  # encoder 使用神经网络类型, 也可以是 ['comb']
        'layerManifold': [None],  # encoder层流形, 3E
        'actM_L': [None],  # 激活函数流形, 3A
        'manifold': [None],  # encoder输出/decoder输入流形, 3D
        'dim': [2],  # encoder 输出维度
        'mixedType': [9],  # 结合方法, 默认非结合
        'dtype': [32],  # comb方法精度
        'task_weight': [1],  # 主任务占比权重
        'data_result': [['o3']],  # 默认 Animal
    }
    初始参数D_.update(初始参数D)
    paras_f = lambda: 训练生成任务_obj.result['paras']  # 获取参数模版的方法
    from_mark_get_result_f = lambda mark: sum([obj.que_task({'paras': {'mark': i}}) for i in mark], [])
    初始参数D_['data_result'] = from_mark_get_result_f(初始参数D_['data_result'])

    def 构建简单任务(dh_L, layer, layerManifold, actM_L, manifold, dim, mixedType, dtype, task_weight, data_result):
        # 混合encoder的处理
        if mixedType == 9:  # 不混合dh_L长度不能大于1, 否则可能因mixedType报错
            dh_L = dh_L[:1]
        if len(dh_L) == 1 or layer == 'comb':  # 非结合
            mixedType = 9
            task_weight = 1
        elif len(dh_L) == 2:
            if mixedType == 0:  # 损失函数结合才需要权重
                task_weight = 0.9 if (task_weight >= 1 or task_weight <= 0) else task_weight
            else:
                task_weight = 1
        else:
            raise NameError('dh_L 数量必须是1或2!', dh_L)
        # 默认流形
        if (manifold is layerManifold is actM_L is None) or layer == 'comb':
            manifold = layerManifold = actM_L = 2  # 默认 poincare
        else:
            dtype = 32  # 不是 comb 统一精度
        if manifold is None:
            manifold = layerManifold if (layerManifold is not None) else actM_L
        if layerManifold is None:
            layerManifold = manifold
        if actM_L is None:
            actM_L = layerManifold
        # 构建mark
        paras = paras_f()
        paras['mark'] = [dh_L[0], layer, f'E{layerManifold}', f'A{actM_L}', f'D{manifold}', f'd{dim}',
                         f'mt{mixedType}', f'dt{dtype}', f'tw{task_weight}',
                         '-'.join(data_result['paras']['mark'] + data_result['paras']['mixed_tree'])]
        mark = '_'.join(paras['mark'])
        if mark in mark_re_D:
            if not 允许重复mark:
                return None
            mark_re_D[mark] += 1
        else:
            mark_re_D[mark] = 1
        paras['mark'].append(f're{mark_re_D[mark]}')  # 重复编号, 第一个就是 re1
        # 构建参数
        if layer == 'comb':
            paras['comb']['RG'] = data_result['data_path']['RG']
            paras['comb']['dh'] = data_result['data_path']['dh']
            paras['comb']['dtype'] = dtype
            paras['comb']['dim'] = dim
        else:
            paras['trainParas']['dh_L'] = dh_L
            paras['encoderParas']['layer'] = layer
            paras['encoderParas']['layerManifold'] = layerManifold
            paras['encoderParas']['layer_paras']['actM_L'] = actM_L
            paras['encoderParas']['manifold'] = manifold
            paras['encoderParas']['encoderOutDim'] = (dim, dim)
            paras['trainParas']['mixedType'] = mixedType
            paras['decoderParas']['task_weight'] = {dh_L[0]: task_weight, dh_L[1]: (1 - task_weight)} \
                if len(dh_L) >= 2 else {}
            paras['trainParas']['dh_path'] = data_result['data_path']['dh']
        # 额外参数修改
        if dh_L[0] == 'Classification':  # NC任务数据集比例修改
            paras['dataParas']['devRate'] = 0.1
            paras['dataParas']['testRate'] = 0.6
        return paras

    paras_L = []
    for dh_L in 初始参数D_['dh_L']:
        for layer in 初始参数D_['layer']:
            for layerManifold in 初始参数D_['layerManifold']:
                for actM_L in 初始参数D_['actM_L']:
                    for manifold in 初始参数D_['manifold']:
                        for dim in 初始参数D_['dim']:
                            for mixedType in 初始参数D_['mixedType']:
                                for dtype in 初始参数D_['dtype']:
                                    for task_weight in 初始参数D_['task_weight']:
                                        for data_result in 初始参数D_['data_result']:
                                            paras = 构建简单任务(dh_L, layer, layerManifold, actM_L, manifold, dim,
                                                           mixedType, dtype, task_weight, data_result)
                                            if paras is not None:
                                                paras_L.append(paras)
    return paras_L, 初始参数D_


def 构训练任务(训练生成任务_obj: 训练生成任务, obj: 数据生成任务):
    允许重复mark = False

    paras_L_L = []
    mark_re_D = {}
    # 6个雷达图: (4任务*6指标)*3E流形*2维度*GCN*Animal
    paras_L_L.append(穷举构建简单任务方法({
        'dh_L': [['Classification'], ['LinkPred'], ['GraphDistor'], ['HypernymyRel']],
        'layerManifold': [0, 1, 2],
        'dim': [2, 16],
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])

    # 6个雷达图: (3方法*6指标)*3E流形*2维度*LP*Animal + 庞加莱comb方法, 欧式二维绘图结果
    # 8个散点图: poincare上的 HNN、GCN、GAT、Combinatorial*5精度*Animal
    paras_L_L.append(穷举构建简单任务方法({
        'layer': ['mlp', 'gcn', 'gat'],
        'layerManifold': [0, 1, 2],
        'dim': [2, 16],
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])
    paras_L_L.append(穷举构建简单任务方法({
        'layer': ['comb'],
        'dtype': [32, 64, 128, 512, 3000],
        'dim': [2, 16],
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])

    # 6个三维透视图: (双指标可变树+2固定树)*6指标*二维*GCN*Poincare*LP
    paras_L_L.append(穷举构建简单任务方法({
        'data_result': [['t6']],
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])

    # 2个雷达图: (4可变树+2固定树+4可变图)*(comb结果+3E流形)*2维度*GCN*LP
    mark = [['t1'], ['t2'], ['t3'], ['t4'], ['o2'], ['o3'], ['g1'], ['g2'], ['g3'], ['g4']]
    paras_L_L.append(穷举构建简单任务方法({
        'layerManifold': [0, 1, 2],
        'dim': [2, 16],
        'data_result': mark,
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])
    paras_L_L.append(穷举构建简单任务方法({
        'layer': ['comb'],
        'dtype': [3000],
        'dim': [2, 16],
        'data_result': mark[:-4],
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])

    # 4个数字热力图: (3E流形*3A流形*3D流形)*2维度*4指标*GCN*Animal*LP
    paras_L_L.append(穷举构建简单任务方法({
        'manifold': [0, 1, 2],
        'layerManifold': [0, 1, 2],
        'actM_L': [0, 1, 2],
        'dim': [2, 16],
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])

    # 8个折线图: (4结合方式+不结合)*5指标*4任务*2公开树*Hyperboloid*GCN*二维
    # 2个热力图: 7指标*3metrics*4任务*2公开树*Hyperboloid*GCN*二维
    mark = [['o2'], ['o3']]
    paras_L_L.append(穷举构建简单任务方法({
        'manifold': [1],
        'dh_L': [['Classification', 'LinkPred'], ['LinkPred', 'GraphDistor'],
                 ['GraphDistor', 'LinkPred'], ['HypernymyRel', 'GraphDistor']],
        'mixedType': [0, 1, 2, 3, 9],
        'task_weight': [0.9],
        'data_result': mark,
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])

    # 8个雷达图: (comb单树+Poincare*(单树+子树))*8子树*4指标*2维度*LP*GCN
    mark = [['t5'], ['t5.1.1'], ['t5.1.2'], ['t5.1.3'], ['t5.1.4'], ['t5.2.1'], ['t5.2.2'], ['t5.2.3'], ['t5.2.4']]
    paras_L_L.append(穷举构建简单任务方法({
        'layerManifold': [2],
        'dim': [2, 16],
        'data_result': mark,
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])
    paras_L_L.append(穷举构建简单任务方法({
        'layer': ['comb'],
        'dtype': [3000],
        'dim': [2, 16],
        'data_result': mark[1:],
    }, obj, mark_re_D, 允许重复mark=允许重复mark, 训练生成任务_obj=训练生成任务_obj)[0])

    return paras_L_L


if __name__ == '__main__':
    # 训练生成任务.test()

    构建新任务 = False
    重新构建未完成任务 = True  # 当 构建新任务=False, 删除所有未执行和已执行但是没有数据的任务, 然后加入 构训练任务() 中未执行的任务. 意思就是已执行的就算了, 未执行的都和 构训练任务() 一致.
    路径 = 'am_all_train'
    数据生成任务_obj = 数据生成任务('al_all_data')

    start = time.time()
    if not os.path.exists(路径):
        构建新任务 = True
    if 构建新任务:
        print('构建新任务:')
        obj = 训练生成任务(路径, new=True)
        for paras_L in 构训练任务(obj, 数据生成任务_obj):
            for paras in paras_L:
                obj.add_task(paras)
    else:
        obj = 训练生成任务(路径)
        if 重新构建未完成任务:
            print('重新构建未完成任务:')
            obj.clean()
            print('删除未完成任务数量:', len(obj.del_task({'executed': False})))
            更新任务 = 1
            for paras_L in 构训练任务(obj, 数据生成任务_obj):
                for paras in paras_L:
                    tasks = obj.que_task({'paras': {'mark': paras['mark']}})
                    if len(tasks) == 0:
                        obj.add_task(paras)
                        print(f'增加了{更新任务}个任务, mark:', paras['mark'])
                        更新任务 += 1
    obj.clean()
    obj.run_task(dataset_db_path=数据生成任务_obj.db_dir)
    print('=' * 10, '统计结果:')
    obj.统计结果()
    obj.close()
    print('总耗时:', (time.time() - start) / 3600, '小时')