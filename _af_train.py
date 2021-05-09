from _ad_graph_model import *
from copy import deepcopy
import datetime


def draw_line_chart_simplify(v=None):
    '''
    简化的 draw_line_chart 折线图绘制方法, 主要用于绘制指标. 固定参数可适当修改
    :param v: None or dict; 折线图参数, 如果为None则返回参数模版
    :return:
    '''
    if v is None:
        return deepcopy({
            'title': '',
            'x': [],  # epochs
            'xaxis': 'epochs',
            'y_left': [],  # 左, 范围在 [0, 1] 的指标结果
            'yaxis_left': '$y_{left}$',  # 左, 坐标轴
            'ylabel_left': [],  # 左, 每个指标的名称
            'ylim_left': [0, 1],  # 左侧坐标轴范围,  左侧指标一般都是 [0,1]
            # 'ylim_left': None,
            'y_right': [],  # 右, 可能不限制范围的指标结果
            'yaxis_right': '$y_{right}$',  # 右, 坐标轴
            'ylabel_right': [],  # 右, 每个指标的名称
            'ylim_right': None,  # 右侧坐标轴范围
            'save_path': '',
        })
    draw_line_chart(
        title=v['title'],
        x=v['x'], xaxis=v['xaxis'],
        y_left=v['y_left'], yaxis_left=v['yaxis_left'], ylim_left=v['ylim_left'], ylabel_left=v['ylabel_left'],
        y_right=v['y_right'], yaxis_right=v['yaxis_right'], ylim_right=v['ylim_right'], ylabel_right=v['ylabel_right'],
        length=12, width=6,
        save_path=v['save_path'],
        lw=1, ylabel_display=True,
        # grid_ls_x=':', grid_axis_yl=True, grid_axis_yr=True, grid_alpha=0.2,
        annotate=True, annotate_in_left=True, annotate_color=True, legend_right=True,
        xl_text_margin=0.1, xl_arrow_len=0.3, xr_text_margin=0.05, xr_arrow_len=0.3, custom_dpi=90,
    )


def metrics_to_results(metrics):
    """
    将 metrics 转换为 result['epoch'][epoch]['to_m'][i] 的所需形式
    :param metrics: dataHelper.评估() 的返回结果
    :return:
    """
    results = {j[1]: metrics[j[0]] for j in metrics['.shorthand']}
    results['d_mean'] = metrics['两点距离_均值']
    results['d_std'] = metrics['两点距离_标准差']
    results['d_max'] = metrics['两点距离_最大值']
    return results


def train(dataHelper, encoder, decoderParas, path, pictureFromat, epochs=500, epoch_out=50, decoder=None, check=False,
          reDataSetting=True, dataParas=None, encoderEmbTest=False, stop_strategy=None, tiems_eval_draw=1):
    '''
    训练, 无批次
    :param dataHelper: DataHelper()
    :param encoder: Encoder()
    :param decoderParas: dict; 用于decoder构建的额外参数
    :param path: 所有图输出的主目录
    :param pictureFromat: 图片输出格式(没有.开头), 如果eps会导致gif模糊, 如果是pdf会导致无法输出gif图片
    :param epochs: 模型训练的轮数
    :param epoch_out: 模型训练多少轮评估一次
    :param decoder: Decoder; 可选, 使用这个就不需要自己配置模型
    :param check: bool; 每轮训练之后是否直接终止并输出梯度的情况
    :param reDataSetting: bool; 是否每轮训练都重新获取一次训练集
    :param dataParas: None or dict; 获取数据集的参数
    :param encoderEmbTest: bool; 是否用 dataHelper 绘制 encoder 的初始向量图像, 路径见代码. 用于测试 encoder 输出是否与预训练的一致
    :param stop_strategy: None or dict; 停止策略, 有以下键值可选
        : devLossStop: int; epochs至少多少次dev数据集的loss不再降低就跳出训练, <1表示不使用
        : devMetricStop: int; epochs至少多少次后数据集的对应指标不再降低就跳出训练, <1表示不使用
        : trainLossStop: int; epochs至少多少次后数据集的对应指标不再降低就跳出训练, <1表示不使用
        : trainMetricStop: int; epochs至少多少次后数据集的对应指标不再降低就跳出训练, <1表示不使用
        : testLossStop: int; epochs至少多少次后数据集的对应指标不再降低就跳出训练, <1表示不使用
        : testMetricStop: int; epochs至少多少次后数据集的对应指标不再降低就跳出训练, <1表示不使用
    :param tiems_eval_draw: int; 评估几次绘制可视化的散点图, 等于训练 epoch_out*tiems_eval_draw 轮评估一次
    :return:
    '''
    print(Fore.BLUE + '=' * 100 + '数据/模型类型: ' + dataHelper.data['type'] + Style.RESET_ALL)
    # 停止策略
    if stop_strategy is None:
        stop_strategy = {}
    stop_strategy_ = {
        'devLossStop': 0,
        'devMetricStop': 0,
        'trainLossStop': 0,
        'trainMetricStop': 0,
        'testLossStop': 0,
        'testMetricStop': 0,
    }
    stop_strategy_.update(stop_strategy)
    stop_strategy = stop_strategy_
    # 数据生成器
    if not dataParas:
        dataParas = {'devRate': 0.1, 'testRate': 0.1, 'shuffle': True, 'dsMult': 10}
    generateDataset = lambda dataHelper: dataHelper.generateDataset(**dataParas)
    # 获取数据
    generateDataset(dataHelper)
    dataset = dataHelper.getDataset()
    print('getDataset 获得的数据', dataset.keys())
    # 生成目录
    if not os.path.exists(path):
        os.makedirs(path)

    # 这个绘图结果应当与 getPreEncoder-emb-mX-dh(Y) 的结果以及 Y 文件夹中对应epoch的结果一致
    if encoderEmbTest and encoder.encoderOutDim[-1] == 2:
        test_name = f'train-encoderEmb-m{encoder.manifold.s}-dh({dataHelper.data["type"]})'
        print('-' * 10, test_name, 'encoder初始输出嵌入')
        节点坐标D = {}  # 构建节点坐标字典
        embedding = encoder.model(encoder.getData(dataset)).numpy()
        for j, c in enumerate(embedding):
            节点坐标D[j] = c.tolist()
        dataHelper.绘图(节点坐标D, 使用分类颜色=True, 使用树结构颜色=False, title=test_name, saveName=f'{path}/{test_name}.pdf')

    print('-' * 20 + '配置对应模型')
    if decoder is None:
        decoder = Decoder.dhToDecoder(dataHelper)[0](dataset, encoder, **decoderParas)
    分类颜色绘图 = 树结构颜色绘图 = False
    if dataHelper.data['trees'] and len(dataHelper.data['trees']) > 1:
        树结构颜色绘图 = True
    if dataHelper.data['type'] == Classification.__name__:
        分类颜色绘图 = True
    decoder.summary(path + 'modelD.png')  # 输出模型图

    # 指标记录
    最好结果_ = {
        'train': [0, 0],  # [loss,metric]
        'dev': [0, 0],  # [loss,metric]
        'test': [0, 0],  # [loss,metric]
        'epoch': 0,
    }
    最好结果 = {
        'loss': {  # 最好loss下的结果
            'train': copy.deepcopy(最好结果_),
            'dev': copy.deepcopy(最好结果_),
            'test': copy.deepcopy(最好结果_),
        },
        'metric': {  # 最好metric下的结果
            'train': copy.deepcopy(最好结果_),
            'dev': copy.deepcopy(最好结果_),
            'test': copy.deepcopy(最好结果_),
        },
    }
    本次结果 = copy.deepcopy(最好结果_)
    best_dev_encoder = encoder  # dev最好的那次结果的encoder

    # 折线图相关
    折线图_overall = draw_line_chart_simplify()  # 与流形无关的折线图, 比如 metric/loss
    折线图_流形 = {i: [  # 与流形有关的折线图, 最后一组是距离
        ['', draw_line_chart_simplify()],  # [描述,折线图参数], 这里是 第一棵树 或 唯一的图
    ] for i in [0, 1, 2]}  # 3个流形
    待合成gif图片_D = {}  # {gif路径:[静态图片,..],..}, 需要将多批次合成的gif图片, 包括折线图/一般绘图

    # 要返回的所有结果
    result = {
        'time': str(datetime.datetime.now()),
        'epoch': {},
        'dataParas': dataParas,
        'decoderParas': decoderParas,
        'best_result': 最好结果,
    }

    print('-' * 20 + '训练')
    评估次数 = 0
    for epoch in range(epochs + 1):
        # 训练
        if epoch > 0:
            out = decoder.compute(dataType='train', check=check)
        else:
            out = decoder.compute(training=False, dataType='train', check=check)
        if epoch % epoch_out == 0 or epoch <= 0:
            本次结果['epoch'] = epoch
            print('\n\tloss\t%s' % decoder.metric_name())
            # 输出训练集指标
            metric = out['metric'].numpy()
            loss = out['loss'].numpy()
            print(epoch, loss, metric)
            本次结果['train'] = [loss, metric]
            # 输出验证集指标
            dataType = 'dev'
            if decoder.check_data(dataType=dataType):
                out_ = decoder.compute(training=False, dataType=dataType)
                print('dev:', out_['loss'].numpy(), out_['metric'].numpy())
                本次结果['dev'] = [out_['loss'].numpy(), out_['metric'].numpy()]
            else:
                print(f'没有数据集-{dataType}!')
            # 输出测试集指标
            dataType = 'test'
            if decoder.check_data(dataType=dataType):
                out_ = decoder.compute(training=False, dataType=dataType)
                print('test:', out_['loss'].numpy(), out_['metric'].numpy())
                本次结果['test'] = [out_['loss'].numpy(), out_['metric'].numpy()]
            else:
                print(f'没有数据集-{dataType}!')
            # 构建 result
            result['epoch'][str(epoch)] = {
                'to_m': {},
                'dataset': {
                    'train': {
                        'metric': 本次结果['train'][1],
                        'loss': 本次结果['train'][0],
                    },
                    'dev': {
                        'metric': 本次结果['dev'][1],
                        'loss': 本次结果['dev'][0],
                    },
                    'test': {
                        'metric': 本次结果['test'][1],
                        'loss': 本次结果['test'][0],
                    },
                },
            }

            # 获得不同流形嵌入及其距离
            m_e8d8d0_L = encoder.get_mutilManifold_embeddingANDdist(
                out['embedding'],
                # edges=[(0, 1), (0, 2)],  # 测试
            )
            for i, x in enumerate(m_e8d8d0_L):
                embedding = x['embedding']
                距离三元组 = x['dist']
                原点距离二元组 = x['dist_0']
                # 输出可能出现nan的距离
                if len(x['dist_nan']) > 0:
                    print("x['dist_nan']:")
                    pprint(x['dist_nan'])
                if len(x['dist_0_nan']) > 0:
                    print("x['dist_0_nan']:")
                    pprint(x['dist_0_nan'])

                # 要输出的图片名字
                saveName = path + 'm%d_%d.%s' % (i, epoch, pictureFromat)
                # 指标输出
                print(saveName, '- 指标:')
                metrics = dataHelper.评估(距离三元组, 原点距离二元组, 计算节点距离保持指标=True, 强制计算图失真=True)
                pprint({k: v for k, v in metrics.items() if k[0] != '.'})
                # 构建 result
                result['epoch'][str(epoch)]['to_m'][str(i)] = metrics_to_results(metrics)

                if 评估次数 % tiems_eval_draw == 0 and encoder.encoderOutDim[-1] == 2:
                    # 绘图准备参数: 构建节点坐标字典
                    节点坐标D = {}
                    for j, c in enumerate(embedding):
                        节点坐标D[j] = c.tolist()
                    # 开始绘图
                    saveName_trees = dataHelper.绘图(
                        节点坐标D, 使用分类颜色=分类颜色绘图, 使用树结构颜色=树结构颜色绘图,
                        title='%s, e:%d, m:%.4f, ' % (Manifold.s_to_tex(i), epoch, metric),
                        saveName=saveName, metrics=metrics,
                    )

                    # 整体图的 gif 图片预保存
                    gifName = path + 'm%d.gif' % i
                    if gifName in 待合成gif图片_D:
                        待合成gif图片_D[gifName].append(saveName)
                    else:
                        待合成gif图片_D[gifName] = [saveName]
                    # trees gif 图片预保存
                    if saveName_trees:
                        gifName_trees = gifName[:-4] + '-trees.gif'
                        if gifName_trees in 待合成gif图片_D:
                            待合成gif图片_D[gifName_trees].append(saveName_trees)
                        else:
                            待合成gif图片_D[gifName_trees] = [saveName_trees]

                # 与流形有关的折线图 指标参数
                metrics_names = [  # 修改时相关对应变量的使用都要检查 metrics_v, shorthand_tex, 距离相关
                    '图失真指标', '图失真指标_密度', '根节点层级指标', '父节点层级指标', '原点层级指标',
                    '兄节点距离保持指标', '原点距离_度NDCG',
                ]
                shorthand_D = {j[0]: j[2] for j in metrics['.shorthand']}  # {全称:tex简称,..}
                metrics_v = [metrics[j] for j in metrics_names]  # [[指标值,..],..], 与 metrics_names 对应
                metrics_v = [i if i else [0] for i in metrics_v]  # 防止指标为None
                shorthand_tex = [shorthand_D[j] for j in metrics_names]  # [tex简称,..], 与 metrics_names 对应
                n = len(metrics_v[2])  # tree 数量
                # 与流形有关的折线图 参数初始化
                d_v_L = 折线图_流形[i]
                if len(d_v_L[0][1]['y_left']) == 0:
                    if n == 1:
                        d_v_L[0][1]['y_left'] = [[], [], [], [], [], []]  # 6个指标
                        d_v_L[0][1]['ylabel_left'] = shorthand_tex[1:]
                        d_v_L[0][1]['y_right'] = [[]]  # 1个指标
                        d_v_L[0][1]['ylabel_right'] = [shorthand_tex[0]]
                    else:
                        for j in range(n):
                            describe = 'tree=%d' % (j + 1)
                            d_v_L.append([describe, deepcopy(d_v_L[0][1])])
                            d_v_L[j + 1][1]['y_left'] = [[], [], [], [], [], [], []]  # 6个指标 + 整体 原点距离_度NDCG
                            d_v_L[j + 1][1]['ylabel_left'] = shorthand_tex[1:] + [shorthand_tex[-1] + '*']
                            d_v_L[j + 1][1]['y_right'] = [[]]  # 1个指标
                            d_v_L[j + 1][1]['ylabel_right'] = [shorthand_tex[0]]
                        del d_v_L[0]
                    # 距离
                    d_v_L.append(['distance', draw_line_chart_simplify()])
                    d_v_L[-1][1]['y_left'] = [[], []]
                    d_v_L[-1][1]['ylabel_left'] = ['mean', 'std']
                    d_v_L[-1][1]['y_right'] = [[]]
                    d_v_L[-1][1]['ylabel_right'] = ['max']
                # 与流形有关的折线图 参数赋值
                for j, (describe, v) in enumerate(d_v_L[:-1]):
                    v['y_right'][0].append(metrics_v[0][0])  # 图失真指标
                    v['y_left'][0].append(metrics_v[1][0])  # 图失真指标_密度
                    for k in range(2, len(metrics_names)):  # 后面的指标, 每棵树一个的
                        v['y_left'][k - 1].append(metrics_v[k][j])
                if n > 1:  # 整体 原点距离_度NDCG, 大于树数量的指标
                    for j, (describe, v) in enumerate(d_v_L[:-1]):
                        v['y_left'][-1].append(metrics_v[-1][-1])
                # 距离 参数赋值
                d_v_L[-1][1]['y_left'][0].append(metrics['两点距离_均值'][0])
                d_v_L[-1][1]['y_left'][1].append(metrics['两点距离_标准差'][0])
                d_v_L[-1][1]['y_right'][0].append(metrics['两点距离_最大值'][0])

            # 与流形无关的折线图 参数初始化
            if len(折线图_overall['y_left']) == 0:
                # 折线图_overall
                折线图_overall['y_left'] = [[], [], []]  # metric
                折线图_overall['ylabel_left'] = [i + decoder.metric_name() for i in ['tr-', 'de-', 'te-']]
                折线图_overall['y_right'] = [[], [], []]  # loss
                折线图_overall['ylabel_right'] = [i + decoder.loss_name() for i in ['tr-', 'de-', 'te-']]
            # 与流形无关的折线图 参数赋值
            折线图_overall['y_left'][0].append(本次结果['train'][1])
            折线图_overall['y_left'][1].append(本次结果['dev'][1])
            折线图_overall['y_left'][2].append(本次结果['test'][1])
            折线图_overall['y_right'][0].append(本次结果['train'][0])
            折线图_overall['y_right'][1].append(本次结果['dev'][0])
            折线图_overall['y_right'][2].append(本次结果['test'][0])
            # 折线图的横坐标 赋值
            折线图_overall['x'].append(epoch)  # 与流形无关
            for m, d_v_L in 折线图_流形.items():  # 与流形有关
                for describe, v in d_v_L:
                    v['x'].append(epoch)

            # 记录 最好结果
            for k, v in 最好结果.items():
                for data, res_D in v.items():
                    if k == 'loss':
                        if res_D['epoch'] == 0 or res_D[data][0] > 本次结果[data][0]:  # loss越小越好
                            v[data] = copy.deepcopy(本次结果)
                            if data == 'dev' and 0 < stop_strategy['devLossStop']:
                                best_dev_encoder = encoder.copy_encoder()
                    elif k == 'metric':
                        if res_D['epoch'] == 0 or res_D[data][1] < 本次结果[data][1]:  # metric越大越好
                            v[data] = copy.deepcopy(本次结果)
                            if data == 'dev' and 0 < stop_strategy['devMetricStop']:
                                best_dev_encoder = encoder.copy_encoder()
                    else:
                        raise NameError('最好结果中类型错误!', 最好结果, k)
            # 终止条件
            if 0 < stop_strategy['devLossStop'] <= 本次结果['epoch'] - 最好结果['loss']['dev']['epoch']:
                break
            if 0 < stop_strategy['devMetricStop'] <= 本次结果['epoch'] - 最好结果['metric']['dev']['epoch']:
                break
            if 0 < stop_strategy['trainLossStop'] <= 本次结果['epoch'] - 最好结果['loss']['train']['epoch']:
                break
            if 0 < stop_strategy['trainMetricStop'] <= 本次结果['epoch'] - 最好结果['metric']['train']['epoch']:
                break
            if 0 < stop_strategy['testLossStop'] <= 本次结果['epoch'] - 最好结果['loss']['test']['epoch']:
                break
            if 0 < stop_strategy['testMetricStop'] <= 本次结果['epoch'] - 最好结果['metric']['test']['epoch']:
                break
            评估次数 += 1
            print()
        else:
            print('.', end='', flush=True)  # 一次 epoch 一个点
        # 重新采样一批数据
        if reDataSetting:
            generateDataset(dataHelper)
            decoder.reDataSetting(dataHelper.getDataset())
    print('最好结果:')
    pprint(最好结果)
    result['best_result'] = 最好结果

    # 指标简写
    print('指标简写:')
    pprint(metrics['.shorthand'])

    # 绘制 与流形无关的折线图
    折线图_overall['title'] = 'performance'
    折线图_overall['save_path'] = path + 'lineChart_performance.' + pictureFromat
    draw_line_chart_simplify(折线图_overall)
    # 绘制 与流形有关的折线图
    for m, d_v_L in sorted(折线图_流形.items()):
        for describe, v in d_v_L:
            if describe:
                title = ('m=%d; ' % m) + describe
            else:
                title = 'm=%d' % m
            # 折线图特殊参数配置
            v['title'] = title
            v['save_path'] = path + 'lineChart_' + title + '.' + pictureFromat
            draw_line_chart_simplify(v)
            # gif 图片预保存
            gif_path = path + 'lineChart_' + describe + '.gif'
            if gif_path in 待合成gif图片_D:
                待合成gif图片_D[gif_path].append(v['save_path'])
            else:
                待合成gif图片_D[gif_path] = [v['save_path']]
    # 生成gif
    print('生成gif...')
    for k, v in 待合成gif图片_D.items():
        create_gif(v, k, 0.99)
    return decoder, result, best_dev_encoder


def getPreEncoder(encoder: Encoder, dataHelper: MixedDataset, mainPath: str, train_wrap, mixedType=0, dataParas=None):
    """
    获取预训练的encoder和主任务dataHelper用于混合预训练encoder
    输入的 dataHelper 如果重新存取后打乱数据集顺序就可能导致和有依赖的副任务(邻接矩阵)冲突
    注意 dh.copy_dataHelper 是 self 和 self.data 的浅拷贝

    :param encoder: Encoder(); 初始 encoder
    :param dataHelper: MixedDataset(); MixedDataset 的实例化对象
    :param mainPath: str; 保存副任务结果的主文件夹, 一般是主任务结果的文件夹
    :param train_wrap: train 的包裹函数, 只留参数: dataHelper/path/encoder/dataParas
    :param mixedType: int; 混合的方式, 1=E参数不固定, 2=E参数固定-新增E, 3=E参数固定
    :param dataParas: None or dict; dataHelper.generateDataset 的初始参数, 用于副任务训练的参数设定
    :return: Encoder(), dataHelper(), pre_encoder
    """
    print('-' * 10, f'getPreEncoder: mixedType={mixedType}, dh_name:{[i.data["type"] for i in dataHelper.task_dh]}')
    dataParas = copy.deepcopy(dataParas)  # 防止参数干扰
    dataParas.update({
        'devRate': 0,
        'testRate': 0,
        'shuffle': False,  # 打乱正例就对统一有依赖的任务的邻接矩阵没有意义了
        'noTrainMinNum': 1000000,
    })
    pre_encoder = {}  # 用于收集pre_encoder的参数
    for i, dh in enumerate(dataHelper.task_dh[1:]):  # 按E参数不固定/顺序的方式训练所有副任务
        task_name = f'{(i + 1)}.{dh.data["type"]}'
        result, encoder = train_wrap(dataHelper=dh, path=f'{mainPath}/{task_name}/', encoder=encoder,
                                     dataParas=dataParas)[1:3]
        pre_encoder[dh.__class__.__name__] = result
        print()
    embedding = encoder.model(encoder.getData(dh.getDataset())).numpy()  # 只需要嵌入

    # 绘图测试 embedding,  用于测试 encoder 输出是否与预训练的一致, 适用于 E参数(不)固定 情况, 不适用于新增E
    if encoder.encoderOutDim[-1] == 2:
        test_name = f'getPreEncoder-emb-m{encoder.manifold.s}-dh({dh.data["type"]})'
        print('-' * 10, test_name, 'encoder预训练后输出嵌入')
        节点坐标D = {}  # 构建节点坐标字典
        for j, c in enumerate(embedding):
            节点坐标D[j] = c.tolist()
        dh.绘图(节点坐标D, 使用分类颜色=True, 使用树结构颜色=False, title=test_name, saveName=f'{mainPath}/{test_name}.pdf')

    dh = dataHelper.task_dh[0]  # type: DataHelper
    if mixedType == 1:
        return encoder, dh, pre_encoder
    elif mixedType == 2:
        return encoder.new_encoder(), dh.copy_dataHelper(newData={'feats': embedding}), pre_encoder
    elif mixedType == 3:
        return encoder.new_encoder(layer=HyperLayer.fixed), dh.copy_dataHelper(newData={'feats': embedding}), \
               pre_encoder
    else:
        NameError(f'mixedType 错误! mixedType={mixedType}')


def main_api(trainParas=None, dataParas=None, decoderParas=None, encoderParas=None, memory_limit=None, **kwargs):
    """
    :param trainParas:
    :param dataParas:
    :param decoderParas:
    :param encoderParas:
    :param memory_limit: int or float or None; 最大显存限制, 单位MB, None表示不限制. 过大可能导致 out of memory 报错
    :param kwargs:
    :return:
    """
    # 普通参数
    trainParas_ = {
        # 使用的任务
        # 'dh_L': ['LinkPred'],
        # 'dh_L': ['Classification'],
        # 'dh_L': ['GraphDistor'],
        # 'dh_L': ['HypernymyRel'],
        'dh_L': ['Classification', 'LinkPred'],
        # 'dh_L': ['Classification', 'LinkPred', 'GraphDistor', 'HypernymyRel'],

        'ap': 'af_graph/',  # 图片等等输出的主目录
        'dh_path': 'ac_test.pkl',  # DataHelper 的 load_file 数据路径
        'pictureFromat': 'pdf',  # 所有静态图片的格式, eps会导致gif不清晰
        'epochs': 10,  # 最多训练多少轮
        'epoch_out': 10,  # 多少轮评估一次
        'tiems_eval_draw': 1,  # 评估多少次进行一次可视化绘图
        'stop_strategy': {
            'devLossStop': 100,  # 至少多少轮dev数据集的loss不再降低就跳出训练
            'devMetricStop': 0,  # epochs至少多少次后数据集的对应指标不再降低就跳出训练
            'trainLossStop': 0,
            'trainMetricStop': 0,
            'testLossStop': 0,
            'testMetricStop': 0,
        },
        'mixedType': 1,  # 多个dh时的混合方式: 0=损失加权求和, 1=E参数不固定, 2=E参数固定-新增E, 3=E参数固定
    }
    # train通过datahelper获取数据集的参数
    dataParas_ = {
        'devRate': 0.1,  # 验证集的比例
        'testRate': 0.1,  # 测试集的比例
        'shuffle': False,  # 是否每次都随机划分数据集
        'dsMult': None,  # GraphDistor 边采样倍数
        '等类别': True,  # Classification 是否保证 train/dev/test 中每个分类的节点数量相等
        'negNum': 10,  # HypernymyRel 负例边是正例边的几倍
        'shuffle_neg': True,  # 每轮训练是否打乱负例
    }
    # decoder模型其他参数getPreEncoder
    decoderParas_ = {
        'task_weight': {},  # Decoder_mixedLoss 中不同损失函数的权重
        'compute_all_dist': True,  # 是否在图失真模型中一次计算好所有点间距离
        'learning_rate': 0.01,  # 学习率
    }
    # encoder参数
    encoderParas_ = {
        'manifold': 2,  # encoder 输出的流形
        'encoderOutDim': (100, 2),  # encoder 每一层的输出维度
        'layer': 'gcn',  # encoder 层的神经网络, 参考 HyperLayer 中的方法名
        'layerManifold': None,  # encoder 层的流形, None表示与manifold一致
        'inM': 0,  # 输入encoder的初始特征流形
        'layer_paras': {  # layer 的额外参数, 参考 HyperLayer 对应参数解释
            'actM_L': 1,  # 每一层激活函数在什么流形空间上转换
            'act_L': None,  # 每一层使用什么激活函数
            'head_L': None,  # 每一层 gat 使用几个注意力头, 为None表示全部使用4个, 为int表示每层都是int
        },
        'compute_all_dist': True,  # 是否在encoder距离模型中一次计算好所有点间距离
    }

    if trainParas is None:
        trainParas = {}
    if dataParas is None:
        dataParas = {}
    if decoderParas is None:
        decoderParas = {}
    if encoderParas is None:
        encoderParas = {}
    trainParas_.update(trainParas)
    dataParas_.update(dataParas)
    decoderParas_.update(decoderParas)
    encoderParas_.update(encoderParas)
    trainParas = trainParas_
    dataParas = dataParas_
    decoderParas = decoderParas_
    encoderParas = encoderParas_
    # 所有结果
    result_all = {
        # 一次训练包含的返回值
        'time': str(datetime.datetime.now()),  # 比如 2021-03-29 21:50:16.269569
        'epoch': {
            '10': {
                'to_m': {
                    '0': {
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
                'dataset': {
                    'train': {
                        'metric': 1.0,
                        'loss': 1.0,
                    },
                    'dev': {},
                    'test': {},
                },
            },
            # ...
        },
        'best_result': {
            'loss': {  # 最好loss下的结果
                'train': {
                    'train': [0, 0],  # [loss,metric]
                    'dev': [0, 0],  # [loss,metric]
                    'test': [0, 0],  # [loss,metric]
                    'epoch': 0,
                },
                'dev': {},
                'test': {},
            },
            'metric': {  # 最好metric下的结果
                'train': {},
                'dev': {},
                'test': {},
            },
        },
        'dataParas': copy.deepcopy(dataParas),  # 不含非基本类型参数
        'decoderParas': copy.deepcopy(decoderParas),  # 不含非基本类型参数
        # 最终结果包含的值
        'pre_encoder': {  # 预训练encoder的整个train返回结果
            # 'LinkPred': result,
        },
        'trainParas': copy.deepcopy(trainParas),
        'encoderParas': copy.deepcopy(encoderParas),  # copy防止后续更改为非str和base类型数据
        'dh_graph_info': {},  # dataHelper.data['图统计信息']
    }
    # 普通参数转换
    dh_L = [Decoder.dhToDecoder(i)[2] for i in trainParas['dh_L']]
    encoderParas['manifold'] = Manifold(s=encoderParas['manifold'])
    if isinstance(encoderParas['layerManifold'], int):
        encoderParas['layerManifold'] = Manifold(s=encoderParas['layerManifold'])
    encoderParas['layer'] = eval('HyperLayer.' + encoderParas['layer'])
    if not os.path.exists(trainParas['ap']):
        os.makedirs(trainParas['ap'])

    # 显存限制
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and memory_limit:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"显存限制({memory_limit}MB):", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    # 数据和encoder
    data = DataHelper(load_file=trainParas['dh_path']).data  # 数据集模型
    encoder = Encoder(data['feat_oh'].shape[0], data['feat_oh'].shape[1], **encoderParas)
    encoder.summary(trainParas['ap'] + 'af_modelE.png')  # 输出模型图

    # 构建 dataHelper 为 mixedType 服务
    if len(dh_L) == 1:
        dataHelper = dh_L[0](data=data)
        mainPath = f'{trainParas["ap"]}/{dh_L[0].__name__}/'
    else:
        dataHelper = MixedDataset(data=data, main_task=dh_L[0],
                                  regularization_task=dh_L[1:])
        mainPath = f'{trainParas["ap"]}/{dh_L[0].__name__}-{trainParas["mixedType"]}/'
        if trainParas['mixedType'] > 0:  # 非损失混合
            # 构建 train_wrap
            devMetricStop = trainParas['stop_strategy']['devMetricStop']
            train_wrap = lambda dataHelper, path, encoder, dataParas: train(
                dataHelper=dataHelper, encoder=encoder, decoderParas=decoderParas, path=path,
                pictureFromat=trainParas['pictureFromat'], check=False, reDataSetting=True, dataParas=dataParas,
                # 预训练encoder的停止策略
                epochs=trainParas['epochs'], epoch_out=trainParas['epoch_out'],
                stop_strategy={'devMetricStop': 100 if devMetricStop <= 0 else devMetricStop},  # 强制加一个停止策略
                tiems_eval_draw=trainParas['tiems_eval_draw'],
            )
            encoder, dataHelper, pre_encoder = getPreEncoder(encoder, dataHelper, mainPath, train_wrap,
                                                             trainParas['mixedType'], dataParas)
            result_all['pre_encoder'] = pre_encoder
    # 训练
    # print('-' * 20 + '初始自动评估绘图')
    # dataHelper.自动评估绘图(titleFrontText='$\\mathbb{R}$, ', saveName=mainPath + 'dataGraph.eps')
    print('-' * 20 + '生成数据的绘制模型')
    result = train(
        dataHelper, encoder, decoderParas, mainPath, trainParas['pictureFromat'], trainParas['epochs'],
        trainParas['epoch_out'], check=False, reDataSetting=True, dataParas=dataParas, encoderEmbTest=True,
        stop_strategy=trainParas['stop_strategy'], tiems_eval_draw=trainParas['tiems_eval_draw'],
    )[1]
    result_all['epoch'] = result['epoch']
    result_all['best_result'] = result['best_result']
    result_all['dh_graph_info'] = dataHelper.data['图统计信息']
    return result_all


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    result_all = main_api()
    pprint(result_all)
