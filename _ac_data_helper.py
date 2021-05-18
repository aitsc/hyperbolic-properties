import scipy.sparse as sp
import copy
from _ab_随机树生成 import *
from _ae_utils import *
import inspect


class DataHelper:
    def __init__(self, g: 随机图 = None, load_file=None, data: dict = None, print_stat=True, **kwargs):
        '''
        g, load_file, data 至少有一个. 按该倒序优先, 多选无效
        :param g: 要求随机图节点必须从0开始, 并连续, 否则影响训练中向量的look_up, 以及树结构评估和绘图
        :param load_file: 存储文件
        :param data: dict, self.data 信息
        :param print_stat: bool; 是否输出 图统计信息
        '''
        assert g is not None or load_file is not None or data is not None, '至少输入随机图/加载文件/data!'
        if data:
            self.data = {  # 固定后不再被子类改变
                '图统计信息': data['图统计信息'],
                'nodes': data['nodes'],
                'edges': data['edges'],
                'edgesCrs': data['edgesCrs'],
                'node_class_D': data['node_class_D'],
                'feat_oh': data['feat_oh'],
                'trees': data['trees'],
                '点编号_树D': data['点编号_树D'],
                'feats': data['feats'] if 'feats' in data else None,
            }
            if 'short_path_matrix' in data and data['short_path_matrix'] is not None:
                self.data['short_path_matrix'] = data['short_path_matrix']
            if 'short_path_matrix_st' in data and data['short_path_matrix_st'] is not None:  # 多树混合图子树
                self.data['short_path_matrix_st'] = data['short_path_matrix_st']
        elif load_file:
            self.读取数据(load_file)
        else:
            if g.type == 随机图.类型.nx图:
                nodes = list(g.graph.nodes)
                edges = list(g.graph.edges)
                trees = None
                点编号_树D = None
            else:
                nodes, edges, 点编号_树D = g.树_统计所有点和边()[:3]  # 森林还需考虑 除了树中节点以外的三元组
                edges = [i[:2] for i in edges]
                trees = copy.deepcopy(g.tree)
            nodes.sort()  # 节点顺序编号
            edges = [tuple(sorted(i)) for i in edges]  # 边顺序编号
            data = np.ones(len(edges))
            row = [i[0] for i in edges]
            col = [i[1] for i in edges]
            edgesCrs = sp.csr_matrix((data, (row, col)), shape=(len(nodes), len(nodes)))
            assert len(nodes) == max(nodes) + 1, '节点必须从0开始, 并连续, 否则影响训练中向量的look_up'

            self.data = {  # 固定后不再被子类改变
                '图统计信息': g.统计信息(),
                'nodes': nodes,  # [点编号,..], 节点必须从0开始, 并连续, 否则影响训练中向量的look_up
                'edges': edges,  # [(点l,点r),..], 严禁正反边和自环存储→否则导致度统计/标准化adj/边采样问题
                'edgesCrs': edgesCrs,  # sp.csr_matrix, 等价 edges
                'node_class_D': g.node_class_D,  # {节点:类别,..}, 类别是int从0开始不间断, 用于数据one-hot类别交叉熵
                'feat_oh': sp.eye(len(nodes)),  # sp.csr_matrix, 节点特征的 one-hot 向量, 与 nodes 序号一致
                'trees': trees,  # 所有树的列表, [[[节点编号,父,[子,..],层数,该树高度,该树节点数],..],..]=[[层次遍历节点1,..],..]=[树1,..]
                '点编号_树D': 点编号_树D,  # {点编号:[树序号,..],..})
                'feats': kwargs['feats'] if 'feats' in kwargs else None,
                # sp.csr_matrix or ndarray, 节点的外部输入特征, 与 nodes 序号一致
            }
        self.data.update({'type': None})
        if print_stat:
            print('图统计信息:', self.data['图统计信息'])

    def copy_dataHelper(self, newData=None):
        """
        浅 copy 一份 dataHelper 实例, 用于 self.data 固定后不再被子类改变的问题, 同时保留其他变量, 一般用于混合数据-预训练encoder
        这里 self 变量和 self.data 变量使用浅copy, 如果有更深层修改需要注意
        :param newData: None or dict; 要修改的属性名属于 self.data, 只能修改类内不被依赖的参数, 因为没有init
        :return: DataHelper()
        """
        dataHelper = copy.copy(self)
        dataHelper.data = copy.copy(self.data)
        if newData:
            dataHelper.data.update(newData)
        return dataHelper

    def getFeats(self):
        if 'feats' in self.data and self.data['feats'] is not None:
            return self.data['feats']
        else:
            return self.data['feat_oh']

    def generateDataset(self, *args, **kwargs):
        return None

    def getDataset(self):
        return None

    def 保存数据(self, path):
        '''
        :param path: str, 保存路径
        :return:
        '''
        buff = 10 ** 4
        bit = pkl.dumps(self.data)
        with open(path, 'wb') as w:
            for i in range(0, len(bit), buff):
                w.write(bit[i:i + buff])

    def 读取数据(self, path):
        '''
        :param path: str, 保存路径
        :return:
        '''
        with open(path.encode('utf-8'), 'rb') as r:
            self.data = pkl.load(r)

    @staticmethod
    def normalize(mx):
        """Row-normalize sparse matrix."""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def 评估(self, 距离三元组, 原点距离二元组, 计算节点距离保持指标=False, 强制计算图失真=False, 计算子树图失真=True):
        '''
        对于 图失真指标, 距离三元组 有多少组距离就算多少组平均.
        :param 距离三元组: [(点l,点r,距离),..], 可以只是上三角或下三角, 没有的距离会默认为0
        :param 原点距离二元组: [(点编号,离原点的距离),..]
        :param 计算节点距离保持指标: bool, 是否计算节点距离保持指标, 需要完整的 距离三元组, 否则报错
        :param 强制计算图失真: bool, 是否强制计算图失真指标, 强制计算需要计算全图最短距离
        :param 计算子树图失真: bool, 对于多树混合图的子树, 是否强制计算图失真指标, 强制计算需要计算子树最短距离
            前提: 强制计算图失真==True
        :return:
        '''
        trees = self.data['trees']
        nodes = self.data['nodes']
        edges = self.data['edges']  # [(左点坐标,右点坐标),..], 所有树上的所有边
        # edges_c_dist = {tuple(sorted(i[:2])): i[2] for i in 距离三元组}  # {(左点坐标,右点坐标):距离,..}; 非常耗时,animal数据4017个节点耗时超过9秒
        nodes_c_dist = {p: d for p, d in 原点距离二元组}  # {点编号:距离,..}
        if not isinstance(距离三元组, np.ndarray):
            距离三元组_np = np.array(距离三元组)
        else:
            距离三元组_np = 距离三元组
        # 构建距离矩阵
        assert max(nodes) == len(nodes) - 1, '节点编号需要是序号并且从0开始,否则距离矩阵失效!'
        距离矩阵 = sp.csr_matrix((距离三元组_np[:, 2], (距离三元组_np[:, 0], 距离三元组_np[:, 1])),
                             shape=(len(nodes), len(nodes))).toarray()  # 半三角无自环, 值为距离

        # 边交集数量 = len(set(edges) & set(edges_c_dist))
        # assert 边交集数量 == len(edges), '距离三元组 缺少部分数据中的边!'
        点交集数量 = len(set(nodes_c_dist) & set(nodes))
        assert 点交集数量 == len(nodes), '原点距离二元组 缺少部分数据中的点!'
        metrics = {  # 指标全是列表, 前面带.的不是指标
            '图失真指标': None,  # [float,..], 距离三元组 不全这个值算的就不全, 如果多个树则最后一个是整体, 只有一个树第一个就是整体
            '根节点层级指标': None,  # [float,..], 按序一个树一个
            '父节点层级指标': None,  # [float,..], 按序一个树一个
            '原点层级指标': None,  # [float,..], 按序一个树一个
            '兄节点距离保持指标': None,  # [float,..], 按序一个树一个
            '原点距离_度NDCG': None,  # [float,..], 按序一个树一个, 如果多个树则最后一个是整体, 只有一个树第一个就是整体
            '图失真指标_密度': None,  # [float,..], 距离三元组 不全这个值算的就不全, 如果多个树则最后一个是整体, 只有一个树第一个就是整体
            '两点距离_均值': None,  # [float], 距离三元组 不全这个值算的就不全
            '两点距离_标准差': None,  # [float], 距离三元组 不全这个值算的就不全, 总体标准差
            '两点距离_最大值': None,  # [float], 距离三元组 不全这个值算的就不全
            '.shorthand': [  # 指标名称, 简写, latex简写, 会用于可视化图片 title
                ['根节点层级指标', 'M1', '$\\mathrm{M}_{r}$'],
                ['原点层级指标', 'M2', '$\\mathrm{M}_{o}$'],
                ['父节点层级指标', 'M3', '$\\mathrm{M}_{p}$'],
                ['兄节点距离保持指标', 'M4', '$\\mathrm{M}_{b}$'],
                ['原点距离_度NDCG', 'M5', '$\\mathrm{M}_{5}$'],
                ['图失真指标', 'M6', '$\\mathrm{M}_{d}$'],
                ['图失真指标_密度', 'M7', '$\\mathrm{M}_{dd}$'],
            ],
        }
        dist = lambda p1, p2: max(距离矩阵[p1, p2], 距离矩阵[p2, p1])

        # 两点距离_均值 标准差 方差 指标
        metrics['两点距离_均值'] = [距离三元组_np[:, 2].mean()]
        metrics['两点距离_标准差'] = [((距离三元组_np[:, 2] - metrics['两点距离_均值']) ** 2).mean() ** 0.5]
        metrics['两点距离_最大值'] = [距离三元组_np[:, 2].max()]

        # 图失真指标
        if 'short_path_matrix' in self.data and self.data['short_path_matrix'] is not None:
            short_path_matrix = self.data['short_path_matrix']
        else:
            if 强制计算图失真:
                # 计算多源最短路径
                print('计算多源最短路径...')
                short_path_matrix = multi_source_shortest_path(nodes, edges)  # 全节点-连通图, 否则有 float('inf')
                self.data['short_path_matrix'] = short_path_matrix
            else:
                short_path_matrix = None
        if short_path_matrix is not None:
            边标记矩阵 = 距离矩阵 != 0  # 半三角无自环, 值为1. 距离为0则缺标记边, 这里不影响结果
            最短路径矩阵 = short_path_matrix * 边标记矩阵  # 半三角无自环, 值为最短路径, 其他没有相应边的值为0
            # 用于除以距离密度以保证不受距离相对值影响
            距离密度 = metrics['两点距离_均值'] / (最短路径矩阵.sum() / len(距离三元组_np))
            m = (距离矩阵 / 最短路径矩阵) ** 2
            m[np.isnan(m)] = 0  # nan 置为0
            m_density = (abs(m / (距离密度 ** 2) - 1) * 边标记矩阵).sum() / len(距离三元组_np)
            m = (abs(m - 1) * 边标记矩阵).sum() / len(距离三元组_np)
            metrics['图失真指标'] = [m]
            metrics['图失真指标_密度'] = [2 / (1 + math.e ** -m_density) - 1]
            # 每棵子树的图失真
            if 计算子树图失真 and trees and len(trees) > 0:
                print('每棵子树的图失真...')
                图失真_subtrees = []
                图失真_密度_subtrees = []
                # 每棵子树的多源最短路径
                if 'short_path_matrix_st' in self.data and self.data['short_path_matrix_st'] is not None:
                    short_path_matrix_st = self.data['short_path_matrix_st']
                else:
                    short_path_matrix_st = []
                    for i, t in enumerate(trees):
                        树_边二元组 = []  # [(父节点编号,子节点编号),..]
                        for p in t:
                            树_边二元组 += [(p[0], sp) for sp in p[2]]
                        print('multi_source_shortest_path:', i + 1)
                        spm = multi_source_shortest_path(nodes, 树_边二元组)
                        spm[np.isinf(spm)] = 0  # 去除非子树点的无穷大值
                        short_path_matrix_st.append(spm)
                    self.data['short_path_matrix_st'] = short_path_matrix_st
                # 开始计算
                for i, (spm, t) in enumerate(zip(short_path_matrix_st, trees)):
                    spm = spm * 边标记矩阵  # spm 变为上三角或下三角, 0距离的会被标记0
                    edges_mark = (spm != 0)  # 子树节点可达的边
                    edge_num = np.count_nonzero(edges_mark)  # 子树节点可达的数量
                    # 距离为0导致的 edge_num 不准, 矫正
                    if edge_num != len(t) * (len(t) - 1) / 2:
                        print(f'{i}-subtree 存在距离为0边: np.count_nonzero(spm * 边标记矩阵) != n*(n-1)/2 ; '
                              f'{edge_num} != {len(t) * (len(t) - 1) / 2}')
                        edge_num = len(t) * (len(t) - 1) / 2
                    dis_mat = edges_mark * 距离矩阵  # 距离矩阵, 去除非子树节点距离
                    距离密度 = dis_mat.sum() / edge_num / (spm.sum() / edge_num)
                    m = (dis_mat / (spm + (spm == 0))) ** 2
                    m_density = (abs(m / (距离密度 ** 2) - 1) * edges_mark).sum() / edge_num
                    m = (abs(m - 1) * edges_mark).sum() / edge_num
                    图失真_subtrees.append(m)
                    图失真_密度_subtrees.append(2 / (1 + math.e ** -m_density) - 1)
                metrics['图失真指标'] = 图失真_subtrees + metrics['图失真指标']
                metrics['图失真指标_密度'] = 图失真_密度_subtrees + metrics['图失真指标_密度']

        # 根节点层级指标, 父节点层级指标, 原点层级指标
        if trees and len(trees) > 0:
            m_r = []  # 根节点层级指标
            m_f = []  # 父节点层级指标
            m_o = []  # 原点层级指标
            for t in trees:
                m_r.append(1)  # 根节点默认准确
                m_f.append(1)
                m_o.append(1)
                根节点 = t[0][0]
                for p in t:
                    父节点 = p[0]  # 当前节点作为父节点
                    祖节点 = p[1]
                    for 子节点 in p[2]:  # 子节点作为当前节点
                        d_fr = dist(父节点, 根节点)
                        d_sr = dist(子节点, 根节点)
                        d_fs = dist(父节点, 子节点)
                        if d_fr < d_sr:  # 如果父节点到根节点距离<子节点到根节点距离
                            m_r[-1] += 1
                        if 祖节点 is None or d_fs < dist(祖节点, 子节点):  # 如果父节点到子节点距离<祖节点距离到子节点距离
                            m_f[-1] += 1
                        if nodes_c_dist[父节点] < nodes_c_dist[子节点]:  # 如果父节点到原点的距离<子节点到原点的距离
                            m_o[-1] += 1
                m_r[-1] /= len(t)
                m_f[-1] /= len(t)
                m_o[-1] /= len(t)
            metrics['根节点层级指标'] = m_r
            metrics['父节点层级指标'] = m_f
            metrics['原点层级指标'] = m_o

        # 兄节点距离保持指标
        if 计算节点距离保持指标 and trees and len(trees) > 0:
            m = []  # 每棵树的 兄节点距离保持指标
            for t in trees:
                m.append(1 + len(t[0][2]))  # 根节点和根子节点必然满足条件
                front_nodes = [t[0][0]] + t[0][2]  # 根节点和根子节点
                for p in t[1:]:
                    父节点 = p[0]  # 当前节点作为父节点
                    子节点L = p[2]
                    if len(子节点L) == 0:  # 没有节点
                        continue
                    # 计算 兄弟节点最大距离
                    兄弟节点最大距离 = 0
                    父子节点L = 子节点L  # + [父节点]  # [包括父节点]
                    for i in range(len(父子节点L) - 1):
                        for j in range(i + 1, len(父子节点L)):
                            d = dist(i, j)  # 如果 距离三元组 不全这里会报错
                            if d > 兄弟节点最大距离:
                                兄弟节点最大距离 = d
                    # 计算每个节点是否满足条件
                    for 子节点 in p[2]:  # 子节点作为当前节点
                        得分, 总数 = 0, 0
                        for f in front_nodes:
                            if f == 父节点:
                                continue
                            d = dist(子节点, f)  # 如果 距离三元组 不全这里会报错
                            得分 += 1 if d > 兄弟节点最大距离 else 0
                            总数 += 1
                        m[-1] += 得分 / 总数
                    front_nodes += 子节点L
                m[-1] /= len(front_nodes)
            metrics['兄节点距离保持指标'] = m

        # 原点距离-度排名指标
        node_degree_D = {}  # {节点编号:度数量,..}
        # 度统计
        for l, r in self.data['edges']:
            if l in node_degree_D:
                node_degree_D[l] += 1
            else:
                node_degree_D[l] = 1
            if r in node_degree_D:
                node_degree_D[r] += 1
            else:
                node_degree_D[r] = 1
        u8d_L = sorted([i for i in node_degree_D.items()], key=lambda t: t[1], reverse=True)  # [(节点编号,度),..], 倒叙排序的度值
        p8d8d_L = [(p, d, node_degree_D[p]) for p, d in 原点距离二元组]  # [(节点编号,距离,度),..]
        p8d8d_L = sorted(p8d8d_L, key=lambda t: (t[1], -t[2]))  # [(节点编号,距离,度),..], 距离顺序, 度倒序
        v8d_L = [(i[0], i[2]) for i in p8d8d_L]  # [(节点编号,度),..], 按距离顺序排序的度值

        # 对应节点的指标计算公式
        def mNDCG(nodes_S):
            dcg = idcg = 0
            i = 1
            for p, d in u8d_L:
                if p in nodes_S:
                    idcg += (2 ** d - 1) / math.log(i + 1, 2)
                    i += 1
            i = 1
            for p, d in v8d_L:
                if p in nodes_S:
                    dcg += (2 ** d - 1) / math.log(i + 1, 2)
                    i += 1
            return dcg / idcg

        metrics['原点距离_度NDCG'] = []
        if trees and len(trees) > 1:  # 如果有多棵树
            for t in trees:
                metrics['原点距离_度NDCG'].append(mNDCG(set([p[0] for p in t])))
        metrics['原点距离_度NDCG'].append(mNDCG(set(nodes)))
        return metrics

    def 绘图(self, 节点坐标D, 使用分类颜色=False, 使用树结构颜色=False, ns=2, length=10, width=10, 保留坐标=True, title=None, saveName=None,
           treesTitle=None, 使用层次颜色=8, useLegend=True, 多树图分子树绘制=True, 多树图分子树重制坐标=False, metrics=None,
           exclude_metrics=('M5',)):
        '''
        对于森林, 先画整体图, 子树分开的图按照 从左到右/从上到下 依次绘制trees中的树.
        :param 节点坐标D: {节点编号:(坐标x,坐标y),..}
        :param 使用分类颜色: bool, 优先考虑
        :param 使用树结构颜色: bool
        :param ns: 节点大小
        :param length: int, 画布长度, 单位100像素
        :param width: int, 画布宽度/高度, 单位100像素
        :param 保留坐标: boole, 是否保留横纵坐标
        :param title: None or str, 整体图的标题, 空则不画标题. 有metrics则作为frontText
        :param saveName: None or str, 图片保存位置, 空则不保存, 子树图自动加后缀
        :param treesTitle: None or str, 如果有多棵树, 表示多棵树图的标题, 空则不画标题
        :param 使用层次颜色: int=n, 表示前n个层次使用不同的颜色绘制点, 优先级最次, 需要树的数量等于1. n<2则不用层次颜色绘制
        :param useLegend: bool, 是否绘制图例
        :param 多树图分子树绘制: bool, 对于多树混合图是否单独绘制每颗子树并放在一个图里
        :param 多树图分子树重制坐标: bool, 对于绘制多树混合图的每颗子树时, 按 随机图._树_节点坐标生成 的方式重新绘制每个子树
        :param metrics: None or dict, DataHelper.评估()的返回值, 可以将指标作为标题的一部分. format_metrics参数可在代码中修改
        :param exclude_metrics: tuple; 排除不展示的指标简写, metrics['.shorthand'][1]
        :return: None or saveName_trees, 如果有多树图输出则返回多树图输出地址
        '''
        exclude_metrics = set(exclude_metrics)
        if isinstance(metrics, dict) and metrics:
            title = format_metrics(  # 简化指标描述用于图片 title
                [metrics[j[0]] for j in metrics['.shorthand'] if j[1] not in exclude_metrics],
                names=[j[2] for j in metrics['.shorthand'] if j[1] not in exclude_metrics],
                seg=', ',
                lineMaxLen=550,
                frontText=title,
            )
        # 约束
        if not (self.data['trees'] and len(self.data['trees']) > 1):  # 如果树的数量不大于1
            使用树结构颜色 = False
        if not (self.data['node_class_D'] and len(self.data['node_class_D']) > 1):  # 如果分类数量不大于1
            使用分类颜色 = False
        if not (self.data['trees'] and len(self.data['trees']) == 1) or 使用树结构颜色 or 使用分类颜色:
            使用层次颜色 = 0
        # 重要参数
        nodes = self.data['nodes']  # [节点编号,..]
        nodesXY = [节点坐标D[i] for i in nodes]  # [(x,y),..], 节点坐标不全这里会报错
        assert len(nodesXY[0]) == 2, '节点坐标D 不是二维坐标! dim=%d' % len(nodesXY[0])
        # 优先考虑按分类划分颜色
        if 使用分类颜色:
            c_S = set([v for v in self.data['node_class_D'].values()])  # 类别集合
            nc = ncolors(len(c_S))  # 颜色
            if None in c_S:  # 第一色是空类别
                c_S -= {None}
                c_L = [None] + sorted(c_S)  # 类别list
            else:
                c_L = sorted(c_S)  # 类别list
            c_label_D = {i: '$c_{%s}$' % str(j) for i, j in zip(nc, c_L)}  # {颜色:label,..}
            c_c_D = {i: j for i, j in zip(c_L, nc)}  # {类别:颜色,..}
            nodeC_L = [c_c_D[self.data['node_class_D'][i]] for i in nodes]  # [颜色,..], 与 nodes 顺序对应
        elif 使用树结构颜色:
            nc = ncolors(len(self.data['trees']) + 1)  # 颜色
            c_label_D = {j: '$T_{%s}$' % str(i + 1) for i, j in enumerate(nc[1:])}  # {颜色:label,..}
            c_label_D.update([(nc[0], '$T_{None}$')])
            nodeC_L = []  # [颜色,..], 与 nodes 顺序对应
            for i in nodes:
                树序号L = self.data['点编号_树D'][i]
                if len(树序号L) == 1:
                    nodeC_L.append(nc[树序号L[0] + 1])
                else:
                    nodeC_L.append(nc[0])  # 重合节点使用第一色
        elif 使用层次颜色:
            # 最大层次颜色约束
            max_h = self.data['trees'][0][0][4]
            使用层次颜色 = min(max_h, 使用层次颜色)
            # 节点颜色
            black_colors = ['#222222', '#888888']
            black_colors = black_colors[:max(0, 使用层次颜色 - 4)]  # 至少5种颜色使用黑色
            nc = ncolors(使用层次颜色 - len(black_colors)) + black_colors  # 颜色
            # 颜色对应标签
            c_label_D = {j: '$h_{%s}$' % str(i + 1) for i, j in enumerate(nc[:-1])}  # {颜色:label,..}
            c_label_D.update([(nc[-1], '$h_{%d^+}$' % 使用层次颜色)])
            # 树层次
            点编号_树层次D = {}  # {点编号:树层次,..}
            for p in self.data['trees'][0]:
                点编号_树层次D[p[0]] = p[3]
            # 节点颜色赋值
            nodeC_L = []  # [颜色,..], 与 nodes 顺序对应
            for i in nodes:
                floor = 点编号_树层次D[i] - 1
                if floor >= 使用层次颜色:
                    floor = 使用层次颜色 - 1
                nodeC_L.append(nc[floor])
        else:  # 只使用一种颜色
            nc = ncolors(1)[0]
            c_label_D = {nc: None for i in range(len(nodes))}  # {颜色:label,..}
            nodeC_L = [nc] * len(nodes)
        # 绘图
        plt.rcParams['figure.figsize'] = (length, width)
        plt.figure(1)
        if title:
            plt.title(title,
                      # fontsize=9,
                      )  # 标题
        if not 保留坐标:
            plt.xticks([])  # 去掉横坐标值
            plt.yticks([])  # 去掉纵坐标值
            plt.axis('off')  # 关闭坐标
        # 绘点
        xValue = [i[0] for i in nodesXY]
        yValue = [i[1] for i in nodesXY]
        label_all = {}  # {label:[color,x_L,y_L],..}
        for x, y, c in zip(xValue, yValue, nodeC_L):  # 按图例分配
            label = c_label_D[c]
            if label in label_all:
                label_all[label][0].append(c)
                label_all[label][1].append(x)
                label_all[label][2].append(y)
            else:
                label_all[label] = [[c], [x], [y]]
        for i, (label, all) in enumerate(sorted(label_all.items())):  # 按图例绘制
            if 使用层次颜色 and i == 0:  # 层次绘图时第一个层次点绘制的大一点, 并且空心
                plt.scatter(all[1], all[2], s=ns * 3, c='w', marker='o', zorder=10, label=label, edgecolors=all[0])
            else:
                plt.scatter(all[1], all[2], s=ns, c=all[0], marker='o', zorder=10, label=label)
        if useLegend and isinstance(nc, list):  # 图例
            plt.legend(loc='best')
        # 绘边
        for 点l, 点r in self.data['edges']:
            x1, y1 = 节点坐标D[点l]
            x2, y2 = 节点坐标D[点r]
            plt.plot((x1, x2), (y1, y2), linewidth=1, zorder=1, c='#000000')
        # 保存
        plt.tight_layout()  # 去除边缘空白
        if saveName:
            # 多树结构的文件名
            fname, fextension = os.path.splitext(saveName)
            saveFormat = fextension[1:]
            plt.tight_layout()
            saveName_trees = fname + '-trees' + fextension
            plt.savefig(saveName, format=saveFormat)
        else:
            saveFormat = None
            saveName_trees = None
        plt.show()
        plt.close()  # 防止图片叠加变大, 无界面的Linux
        # 单独绘制每棵树的图, 如果有多棵树. 行=列 or 行+1=列
        if self.data['trees'] and len(self.data['trees']) > 1 and 多树图分子树绘制:
            fig = plt.figure(2)
            tree_num = len(self.data['trees'])
            for i, tree in enumerate(self.data['trees']):
                子树重制坐标D = {p: (x, y) for p, x, y in 随机图._树_节点坐标生成(tree)}
                # 子树位置
                长 = int(tree_num ** 0.5)
                ax = plt.subplot(长, math.ceil(tree_num / 长), i + 1)
                # ax.set_title('tree=%d' % i, fontsize=9)  # 每个子图标题
                if not 保留坐标:
                    plt.xticks([])  # 去掉横坐标值
                    plt.yticks([])  # 去掉纵坐标值
                    plt.axis('off')  # 关闭坐标
                # 点, 颜色
                tree_nodes_S = {i[0] for i in tree}
                点_x_y_L = []  # [[点编号,x,y],..]
                colors = []  # [颜色,..]
                for p, (x, y), c in zip(nodes, nodesXY, nodeC_L):
                    if p not in tree_nodes_S:
                        continue
                    if 多树图分子树重制坐标:
                        x, y = 子树重制坐标D[p]
                    点_x_y_L.append([p, x, y])
                    colors.append(c)
                点_x_y_D = {i[0]: i[1:] for i in 点_x_y_L}
                # 绘点
                xValue = [i[1] for i in 点_x_y_L]
                yValue = [i[2] for i in 点_x_y_L]
                plt.scatter(xValue, yValue, s=ns, c=colors, marker='o', zorder=10)
                # 绘边
                for p in tree:
                    x1, y1 = 点_x_y_D[p[0]]
                    for sp in p[2]:
                        x2, y2 = 点_x_y_D[sp]
                        plt.plot((x1, x2), (y1, y2), linewidth=1, zorder=1, c='#000000')
            if treesTitle:  # 标题
                fig.suptitle(treesTitle)
                fig.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题空出位置, 不准确
            else:
                fig.tight_layout()
            if saveName_trees:
                plt.savefig(saveName_trees, format=saveFormat)  # 小心子树图重名覆盖
        else:
            saveName_trees = None
        plt.show()
        plt.close()  # 防止图片叠加变大, 无界面的Linux
        return saveName_trees

    def 自动评估绘图(self, titleFrontText='', 使用分类颜色=True, 使用树结构颜色=True, 使用层次颜色=True, saveName=None, 多树图绘制nx坐标=True):
        '''
        使用默认的图展示算法进行评估和绘图展示, 绘图和数据类型无关.
        :param titleFrontText: str; 图标题的前缀, 后面是指标描述, 允许$tex$
        :param 使用分类颜色: bool; 多个颜色都是True一般就绘制多个图
        :param 使用树结构颜色: bool
        :param 使用层次颜色: bool
        :param saveName: None or str; 图片保存路径. 不同图例类型的图会加不同的后缀, 注意重名
        :param 多树图绘制nx坐标: bool; 绘制多个子树的图时, 是否用nx图的方式来绘制, 单个子树绘制是否重制为标准坐标
        :return: 节点坐标D, 距离三元组, 原点距离二元组, metrics
        '''
        节点坐标D = {}
        if self.data['trees'] and len(self.data['trees']) > 0 and not (
                多树图绘制nx坐标 and len(self.data['trees']) > 1):  # 树测试
            for i, t in enumerate(self.data['trees'][::-1]):
                n = len(t)
                # 相同节点坐标后者覆盖
                节点坐标D.update({p: (x, y) for p, x, y in 随机图._树_节点坐标生成(t, 平移x=-n / 2 - i * n, 平移y=-n - i * n)})
        else:  # 图测试
            g = nx.Graph()
            g.add_nodes_from(self.data['nodes'])
            g.add_weighted_edges_from([(i[0], i[1], 1) for i in self.data['edges']])
            节点坐标D = nx.spring_layout(g)  # {节点编号:array坐标,..}
        距离三元组 = []
        for i in range(len(self.data['nodes']) - 1):
            x = self.data['nodes'][i]
            x_coo = 节点坐标D[x]
            for j in range(i + 1, len(self.data['nodes'])):
                y = self.data['nodes'][j]
                y_oo = 节点坐标D[y]
                d = sum([(i1 - j1) ** 2 for i1, j1 in zip(x_coo, y_oo)]) ** 0.5
                距离三元组.append((x, y, d))
        原点距离二元组 = [(p, (x ** 2 + y ** 2) ** 0.5) for p, (x, y) in 节点坐标D.items()]
        print('评估...')
        metrics = self.评估(距离三元组, 原点距离二元组, 计算节点距离保持指标=True, 强制计算图失真=True)
        pprint({k: v for k, v in metrics.items() if k[0] != '.'})
        print('绘图...')
        saveName_ = None
        if saveName:
            fname, fextension = os.path.splitext(saveName)
            # 生成目录
            path = os.path.split(os.path.abspath(saveName))[0]
            if not os.path.exists(path):
                os.makedirs(path)
        if self.data['trees'] and len(self.data['trees']) > 1 and 使用树结构颜色:
            if saveName:
                saveName_ = fname + '.T' + fextension
            self.绘图(节点坐标D, 使用树结构颜色=True, title=titleFrontText, saveName=saveName_, useLegend=True,
                    多树图分子树重制坐标=多树图绘制nx坐标, metrics=metrics)
        if 使用分类颜色:
            if saveName:
                saveName_ = fname + '.c' + fextension
            self.绘图(节点坐标D, 使用分类颜色=True, title=titleFrontText, saveName=saveName_, useLegend=True,
                    多树图分子树重制坐标=多树图绘制nx坐标, metrics=metrics)
        if self.data['trees'] and len(self.data['trees']) == 1 and 使用层次颜色:  # 使用层次颜色
            if saveName:
                saveName_ = fname + '.h' + fextension
            self.绘图(节点坐标D, title=titleFrontText, saveName=saveName_, useLegend=True, metrics=metrics)
        return 节点坐标D, 距离三元组, 原点距离二元组, metrics


class LinkPred(DataHelper):
    def __init__(self, g: 随机图 = None, load_file=None, data: dict = None, shuffle_data=False, **kwargs):
        """
        :param g:
        :param load_file:
        :param data: self.data 会尝试读取 data 中的复杂初始化数据, 需要 type 一致
        :param shuffle_data: bool; 是否随机打乱数据集顺序
            用于 generateDataset 方法的 shuffle=False 情况下的初始随机划分数据集
        :param kwargs:
        """
        super(LinkPred, self).__init__(g, load_file, data, **kwargs)
        self.data.update({'type': self.__class__.__name__})
        if not load_file:
            get_data_f = lambda x: data[x] \
                if data and 'type' in data and x in data and data['type'] == self.data['type'] else None
            self.data.update({
                'trainPos': get_data_f('trainPos'),  # numpy.ndarray
                'trainNeg': get_data_f('trainNeg'),  # numpy.ndarray
                'devPos': get_data_f('devPos'),  # numpy.ndarray
                'devNeg': get_data_f('devNeg'),  # numpy.ndarray
                'testPos': get_data_f('testPos'),  # numpy.ndarray
                'testNeg': get_data_f('testNeg'),  # numpy.ndarray
                'adj_train': get_data_f('adj_train'),  # sp.csr_matrix
            })
        # 负例边
        x, y = sp.triu(
            sp.csr_matrix(1. - self.data['edgesCrs'].toarray() - sp.eye(len(self.data['nodes'])))).nonzero()
        self.neg_edges = list(zip(x, y))
        # 正例边
        if shuffle_data:
            self.pos_edges = copy.deepcopy(self.data['edges'])
            random.shuffle(self.pos_edges)
        else:
            self.pos_edges = self.data['edges']
        print('type:', self.data['type'])

    def generateDataset(self, devRate, testRate, shuffle=False, shuffle_neg=True, customTrainSet=None,
                        noTrainMinNum=None, **kwargs):
        '''
        :param devRate: 验证集的比例
        :param testRate: 测试集的比例
        :param shuffle: 是否打乱正例, 即每次都随机划分数据集
        :param shuffle_neg: 是否打乱负例
        :param customTrainSet: None or [(点1,点2),..]; 自定义训练集
            用参数 devRate 和 testRate 按比例获取非训练集, 训练集不受 shuffle 参数影响
        :param noTrainMinNum: None or int; 非训练集的最少数量, 数据不够会用训练集按顺序填充, 用于辅助用途(比如当rate=0时用)
        :return:
        '''
        assert 1 > devRate + testRate >= 0, '比例错误!'
        if customTrainSet is None:
            # 数据集大小
            devNum = int(len(self.pos_edges) * devRate)
            testNum = int(len(self.pos_edges) * testRate)
            trainNum = len(self.pos_edges) - devNum - testNum
            # 边
            if shuffle:
                pos_edges = copy.deepcopy(self.pos_edges)
                random.shuffle(pos_edges)
            else:
                pos_edges = self.pos_edges
        else:
            if not isinstance(customTrainSet, list):
                customTrainSet = customTrainSet.tolist()  # numpy 转 list
            train_set_S = set()  # 自定义边集合
            for i in customTrainSet:
                train_set_S.add((i[0], i[1]))
                train_set_S.add((i[1], i[0]))
            no_train_edges = [i for i in self.pos_edges if tuple(i) not in train_set_S]  # 非训练集边
            trainNum = len(customTrainSet)  # 训练集边数量
            # 测试集和验证集的边数量
            if devRate == testRate == 0:
                devNum = testNum = 0
            else:
                devNum = int(len(no_train_edges) * devRate / (devRate + testRate))
                testNum = len(no_train_edges) - devNum
            # 打乱 测试集和验证集
            if shuffle:
                random.shuffle(no_train_edges)
            pos_edges = customTrainSet + no_train_edges
        # 负例
        if shuffle_neg:
            neg_edges = random.sample(self.neg_edges, len(pos_edges))
        else:
            neg_edges = self.neg_edges[:len(pos_edges)]
        # 数据
        s = 0
        self.data['trainPos'] = np.array(pos_edges[s:s + trainNum])
        self.data['trainNeg'] = np.array(neg_edges[s:s + trainNum])
        s += trainNum
        self.data['devPos'] = np.array(pos_edges[s:s + devNum])
        self.data['devNeg'] = np.array(neg_edges[s:s + devNum])
        s += devNum
        self.data['testPos'] = np.array(pos_edges[s:s + testNum])
        self.data['testNeg'] = np.array(neg_edges[s:s + testNum])
        # 补充非训练集
        if noTrainMinNum and noTrainMinNum > 0:
            concat = lambda x, y, n: np.concatenate((self.data[x], self.data[y][0:n]), axis=0) \
                if self.data[x].size > 0 else self.data[y][0:n]
            n = noTrainMinNum - len(self.data['devPos'])
            if n > 0:
                self.data['devPos'] = concat('devPos', 'trainPos', n)
                self.data['devNeg'] = concat('devNeg', 'trainNeg', n)
            n = noTrainMinNum - len(self.data['testPos'])
            if n > 0:
                self.data['testPos'] = concat('testPos', 'trainPos', n)
                self.data['testNeg'] = concat('testNeg', 'trainNeg', n)
        # 标准化邻接矩阵
        adj_train = sp.csr_matrix(  # 正例边设置为one, 矩阵大小与nodes一致
            (np.ones(self.data['trainPos'].shape[0]), (self.data['trainPos'][:, 0], self.data['trainPos'][:, 1])),
            shape=(len(self.data['nodes']), len(self.data['nodes'])))
        adj_train = adj_train + adj_train.T + sp.eye(adj_train.shape[0])
        self.data['adj_train'] = self.normalize(adj_train)

    def getDataset(self):
        assert self.data['trainPos'] is not None, '还未生成数据集!'
        out = {
            'train_edges_pos': self.data['trainPos'],
            'train_edges_neg': self.data['trainNeg'],
            'dev_edges_pos': self.data['devPos'],
            'dev_edges_neg': self.data['devNeg'],
            'test_edges_pos': self.data['testPos'],
            'test_edges_neg': self.data['testNeg'],
            'adj_train': self.data['adj_train'],
            'features': self.getFeats(),
        }
        return out


class Classification(DataHelper):
    def __init__(self, g: 随机图 = None, load_file=None, data: dict = None, shuffle_data=False, **kwargs):
        """
        :param g:
        :param load_file:
        :param data: self.data 会尝试读取 data 中的复杂初始化数据, 需要 type 一致
        :param shuffle_data: bool; 是否随机打乱数据集顺序
            用于 generateDataset 方法的 shuffle=False 情况下的初始随机划分数据集
        :param kwargs:
        """
        super(Classification, self).__init__(g, load_file, data, **kwargs)
        assert len(self.data['nodes']) == len(self.data['node_class_D']), '节点分类不匹配!'
        self.data.update({'type': self.__class__.__name__})
        if not load_file:
            get_data_f = lambda x: data[x] \
                if data and 'type' in data and x in data and data['type'] == self.data['type'] else None
            self.data.update({
                'trainNodes': get_data_f('trainNodes'),  # numpy.ndarray
                'trainLables': get_data_f('trainLables'),  # numpy.ndarray
                'devNodes': get_data_f('devNodes'),  # numpy.ndarray
                'devLables': get_data_f('devLables'),  # numpy.ndarray
                'testNodes': get_data_f('testNodes'),  # numpy.ndarray
                'testLables': get_data_f('testLables'),  # numpy.ndarray
                'adj_train': get_data_f('adj_train'),  # sp.csr_matrix
                'classNum': get_data_f('classNum'),  # int
            })
            # 标准化邻接矩阵
            if data and 'type' in data and 'adj_train' in data and data['type'] == self.data['type']:
                self.data['adj_train'] = data['adj_train']
            else:
                adj_train = self.data['edgesCrs']
                adj_train = adj_train + adj_train.T + sp.eye(adj_train.shape[0])
                self.data['adj_train'] = self.normalize(adj_train)
        # 构建非空分类节点
        all_class = set()
        self.node_class_L = []  # [(节点编号,类别int),..], 类别从0开始 不间断
        for node in self.data['nodes']:
            c = self.data['node_class_D'][node]
            # 去除 None 类别
            if c is None:
                continue
            all_class.add(c)
            self.node_class_L.append((node, c))
        assert len(all_class) == max(all_class) + 1, "self.data['node_class_D'] 需要类别是int从0开始 不间断"
        # 打乱数据集顺序
        if shuffle_data:
            random.shuffle(self.node_class_L)
        # 计算每类节点数
        self.class_nodes_L = [[] for i in range(len(all_class))]  # [[第0类节点编号,..],..]; 类别从0开始
        for node, c in self.node_class_L:
            self.class_nodes_L[c].append(node)
        self.minClassNodesNum = min([len(i) for i in self.class_nodes_L])
        self.data['classNum'] = len(all_class)
        print('type:', self.data['type'], ', 总节点数:', len(self.data['nodes']), ', 类别数:', len(all_class), ', 有类别节点数:',
              len(self.node_class_L), ', 类别最小节点数:', self.minClassNodesNum)

    def generateDataset(self, devRate, testRate, shuffle=False, 等类别=True, noTrainMinNum=None, **kwargs):
        '''
        总节点数>=数据集数, 因为可能存在None类别节点
        :param devRate: 验证集的比例
        :param testRate: 测试集的比例
        :param shuffle: 是否打乱节点的顺序, 即每次都随机划分数据集
        :param 等类别: 是否保证 train/dev/test 中每个分类的节点数量相等, 选择后超过 minClassNodesNum*类别数 的节点不再选
        :param noTrainMinNum: None or int; 非训练集的最少数量, 数据不够会用训练集按顺序填充, 用于辅助用途(比如当rate=0时用)
        :return:
        '''
        assert 1 > devRate + testRate >= 0, '比例错误!'
        if 等类别:
            n = self.minClassNodesNum
            assert n > 3, '类别最小节点数过少,不能使用等类别! 最少节点数n=%d' % n
        else:
            n = len(self.node_class_L)
        # 数据集大小
        devNum = int(n * devRate)
        testNum = int(n * testRate)
        trainNum = n - devNum - testNum
        if not 等类别:
            # 点
            if shuffle:
                node_class_L = copy.deepcopy(self.node_class_L)
                random.shuffle(node_class_L)
            else:
                node_class_L = self.node_class_L
            nodesF = lambda s, n: [i[0] for i in node_class_L[s:s + n]]
            labelsF = lambda s, n: [i[1] for i in node_class_L[s:s + n]]
        else:
            if shuffle:
                class_nodes_L = copy.deepcopy(self.class_nodes_L)
                for i in class_nodes_L:
                    random.shuffle(i)
            else:
                class_nodes_L = self.class_nodes_L
            nodesF = lambda s, n: sum([i[s:s + n] for i in class_nodes_L], [])
            labelsF = lambda s, n: sum([[i] * n for i in range(len(class_nodes_L))], [])
        # 数据
        s = 0
        self.data['trainNodes'] = np.array(nodesF(s, trainNum))
        self.data['trainLables'] = np.array(labelsF(s, trainNum))
        s += trainNum
        self.data['devNodes'] = np.array(nodesF(s, devNum))
        self.data['devLables'] = np.array(labelsF(s, devNum))
        s += devNum
        self.data['testNodes'] = np.array(nodesF(s, testNum))
        self.data['testLables'] = np.array(labelsF(s, testNum))
        # 补充非训练集
        if noTrainMinNum and noTrainMinNum > 0:
            concat = lambda x, y, n: np.concatenate((self.data[x], self.data[y][0:n]), axis=0) \
                if self.data[x].size > 0 else self.data[y][0:n]
            n = noTrainMinNum - len(self.data['devNodes'])
            if n > 0:
                self.data['devNodes'] = concat('devNodes', 'trainNodes', n)
                self.data['devLables'] = concat('devLables', 'trainLables', n)
            n = noTrainMinNum - len(self.data['testNodes'])
            if n > 0:
                self.data['testNodes'] = concat('testNodes', 'trainNodes', n)
                self.data['testLables'] = concat('testLables', 'trainLables', n)

    def getDataset(self):
        assert self.data['trainNodes'] is not None, '还未生成数据集!'
        out = {
            'train_nodes': self.data['trainNodes'],
            'train_nodes_class': self.data['trainLables'],
            'dev_nodes': self.data['devNodes'],
            'dev_nodes_class': self.data['devLables'],
            'test_nodes': self.data['testNodes'],
            'test_nodes_class': self.data['testLables'],
            'adj_train': self.data['adj_train'],
            'features': self.getFeats(),
            'classNum': self.data['classNum'],
        }
        return out


class GraphDistor(DataHelper):
    def __init__(self, g: 随机图 = None, load_file=None, data: dict = None, shuffle_data=False, **kwargs):
        """
        :param g:
        :param load_file:
        :param data: self.data 会尝试读取 data 中的复杂初始化数据, 需要 type 一致
        :param shuffle_data: bool; 是否随机打乱数据集顺序
            用于 generateDataset 方法的 shuffle=False 情况下的初始随机划分数据集
        :param kwargs:
        """
        super(GraphDistor, self).__init__(g, load_file, data, **kwargs)
        self.data.update({'type': self.__class__.__name__})
        nodes = self.data['nodes']
        if not load_file:
            get_data_f = lambda x: data[x] \
                if data and 'type' in data and x in data and data['type'] == self.data['type'] else None
            self.data.update({  # 数据集和距离矩阵 将占用较大存储空间 nodes_num^2
                'trainEdges': get_data_f('trainEdges'),  # numpy.ndarray
                'trainLables': get_data_f('trainLables'),  # numpy.ndarray
                'devEdges': get_data_f('devEdges'),  # numpy.ndarray
                'devLables': get_data_f('devLables'),  # numpy.ndarray
                'testEdges': get_data_f('testEdges'),  # numpy.ndarray
                'testLables': get_data_f('testLables'),  # numpy.ndarray
                'adj_train': get_data_f('adj_train'),  # sp.csr_matrix, 绝对连通才能计算最短距离
                'short_path_matrix': get_data_f('short_path_matrix'),  # numpy.ndarray
            })
            # 标准化邻接矩阵, 由于groundtruth, adj不需要像LP任务一样和train绑定
            if self.data['adj_train'] is None:
                adj_train = self.data['edgesCrs']
                adj_train = adj_train + adj_train.T + sp.eye(adj_train.shape[0])
                self.data['adj_train'] = self.normalize(adj_train)
            # 计算多源最短路径
            if self.data['short_path_matrix'] is None:
                self.data['short_path_matrix'] = multi_source_shortest_path(nodes, self.data['edges'])
        short_path_matrix = self.data['short_path_matrix']
        # 构建 edgesSBshort
        edgesSBshort = []  # [(小点编号, 大点编号, 最短路径长度),..], numpy
        edges_sp_D = {}  # {(小点编号,大点编号):最短路径长度,..}
        for i in range(short_path_matrix.shape[0] - 1):
            for j in range(i + 1, short_path_matrix.shape[1]):
                edgesSBshort.append((nodes[i], nodes[j], short_path_matrix[i, j]))
                edges_sp_D[(nodes[i], nodes[j])] = short_path_matrix[i, j]
        if shuffle_data:
            random.shuffle(edgesSBshort)
        # 警告
        assert len(edgesSBshort) == len(nodes) * (len(nodes) - 1) / 2, '多源路径数量错误! %d!=n(n-1)/2' % len(edgesSBshort)
        # 统计
        print('type:', self.data['type'], ', edgesSBshort-MB:', sys.getsizeof(edgesSBshort) / 1024 ** 2,
              ', short_path_matrix-MB', short_path_matrix.nbytes / 1024 ** 2)
        self.edgesSBshort = np.array(edgesSBshort)
        self.edges_sp_D = edges_sp_D  # 用于自定义数据集时提取路径

    def generateDataset(self, devRate, testRate, shuffle=False, dsMult=10, noTrainMinNum=None,
                        **kwargs):
        '''
        :param devRate: 验证集的比例
        :param testRate: 测试集的比例
        :param shuffle: 是否打乱数据集的顺序, 即每次使用不同的数据集划分
        :param dsMult: None or float; 边采样倍数, 共采样 dsMult * nodes_num 条边, train/test/dev集共享, None 表示全部取
        :param noTrainMinNum: None or int; 非训练集的最少数量, 数据不够会用训练集按顺序填充, 用于辅助用途(比如当rate=0时用)
        :param kwargs:
        :return:
        '''
        assert 1 > devRate + testRate >= 0, '比例错误!'
        w = (len(self.data['nodes']) - 1) / 2  # 最大边倍数
        if dsMult is None or dsMult <= 0 or dsMult > w:
            dsMult = w
        n = int(dsMult * len(self.data['nodes']))  # 边总数是 n(n-1)/2
        # 数据集大小
        devNum = int(n * devRate)
        testNum = int(n * testRate)
        trainNum = n - devNum - testNum
        # 边
        if shuffle or devNum == testNum == 0:
            if dsMult >= w:  # 最大值不必随机
                edgesSBshort = self.edgesSBshort
            else:
                edgesSBshort = np.random.choice(self.edgesSBshort, n, replace=False)
        else:
            if dsMult >= w:  # 最大值不必随机
                edgesSBshort = self.edgesSBshort[: - devNum - testNum]
            else:  # 训练集每次随机抽
                edgesSBshort = np.random.choice(self.edgesSBshort[: - devNum - testNum], trainNum, replace=False)
            edgesSBshort = np.vstack([edgesSBshort, self.edgesSBshort[- devNum - testNum:]])  # 非训练集固定
        # 数据
        s = 0
        self.data['trainEdges'] = edgesSBshort[s:s + trainNum, 0:2]
        self.data['trainLables'] = edgesSBshort[s:s + trainNum, 2]
        s += trainNum
        self.data['devEdges'] = edgesSBshort[s:s + devNum, 0:2]
        self.data['devLables'] = edgesSBshort[s:s + devNum, 2]
        s += devNum
        self.data['testEdges'] = edgesSBshort[s:s + testNum, 0:2]
        self.data['testLables'] = edgesSBshort[s:s + testNum, 2]
        # 补充非训练集
        if noTrainMinNum and noTrainMinNum > 0:
            concat = lambda x, y, n: np.concatenate((self.data[x], self.data[y][0:n]), axis=0) \
                if self.data[x].size > 0 else self.data[y][0:n]
            n = noTrainMinNum - len(self.data['devEdges'])
            if n > 0:
                self.data['devEdges'] = concat('devEdges', 'trainEdges', n)
                self.data['devLables'] = concat('devLables', 'trainLables', n)
            n = noTrainMinNum - len(self.data['testEdges'])
            if n > 0:
                self.data['testEdges'] = concat('testEdges', 'trainEdges', n)
                self.data['testLables'] = concat('testLables', 'trainLables', n)

    def getDataset(self):
        assert self.data['trainEdges'] is not None, '还未生成数据集!'
        out = {
            'train_edges': self.data['trainEdges'],
            'train_short_path': self.data['trainLables'],
            'dev_edges': self.data['devEdges'],
            'dev_short_path': self.data['devLables'],
            'test_edges': self.data['testEdges'],
            'test_short_path': self.data['testLables'],
            'adj_train': self.data['adj_train'],
            'features': self.getFeats(),
        }
        return out


class HypernymyRel(DataHelper):
    def __init__(self, g: 随机图 = None, load_file=None, data: dict = None, shuffle_data=False, **kwargs):
        """
        :param g:
        :param load_file:
        :param data: self.data 会尝试读取 data 中的复杂初始化数据, 需要 type 一致
        :param shuffle_data: bool; 是否随机打乱数据集顺序
            用于 generateDataset 方法的 shuffle=False 情况下的初始随机划分数据集
        :param kwargs:
        """
        super(HypernymyRel, self).__init__(g, load_file, data, **kwargs)
        self.data.update({'type': self.__class__.__name__})
        if not load_file:
            get_data_f = lambda x: data[x] \
                if data and 'type' in data and x in data and data['type'] == self.data['type'] else None
            self.data.update({
                'trainPos': get_data_f('trainPos'),  # numpy.ndarray, [[点l,点r],..]
                'trainNeg': get_data_f('trainNeg'),  # numpy.ndarray, 负例高一个维度, [[[点l,点r],..negNum],..]
                'devPos': get_data_f('devPos'),  # numpy.ndarray
                'devNeg': get_data_f('devNeg'),  # numpy.ndarray, 负例高一个维度
                'testPos': get_data_f('testPos'),  # numpy.ndarray
                'testNeg': get_data_f('testNeg'),  # numpy.ndarray, 负例高一个维度
                'adj_train': get_data_f('adj_train'),  # sp.csr_matrix
            })
            # 标准化邻接矩阵
            if self.data['adj_train'] is None:
                adj_train = self.data['edgesCrs']
                adj_train = adj_train + adj_train.T + sp.eye(adj_train.shape[0])
                self.data['adj_train'] = self.normalize(adj_train)
        # 负例边
        x, y = sp.triu(
            sp.csr_matrix(1. - self.data['edgesCrs'].toarray() - sp.eye(len(self.data['nodes'])))).nonzero()
        self.neg_edges = list(zip(x, y))
        # 正例边
        if shuffle_data:
            self.pos_edges = copy.deepcopy(self.data['edges'])
            random.shuffle(self.pos_edges)
        else:
            self.pos_edges = self.data['edges']
        print('type:', self.data['type'])

    def generateDataset(self, devRate, testRate, negNum: int = 10, shuffle=False, shuffle_neg=True, customTrainSet=None,
                        noTrainMinNum=None, **kwargs):
        '''
        :param devRate: 验证集的比例
        :param testRate: 测试集的比例
        :param negNum: int, 负例边是正例边的几倍
        :param shuffle: 是否打乱正例, 即每次都随机划分数据集
        :param shuffle_neg: 是否打乱负例
        :param customTrainSet: None or [(点1,点2),..]; 自定义训练集
            用参数 devRate 和 testRate 按比例获取非训练集, 训练集不受 shuffle 参数影响
        :param noTrainMinNum: None or int; 非训练集的最少数量, 数据不够会用训练集按顺序填充, 用于辅助用途(比如当rate=0时用)
        :return:
        '''
        assert 1 > devRate + testRate >= 0, '数据集比例错误!'
        assert negNum > 0, '不满足 negNum > 0 !'
        if customTrainSet is None:
            # 数据集大小
            devNum = int(len(self.pos_edges) * devRate)
            testNum = int(len(self.pos_edges) * testRate)
            trainNum = len(self.pos_edges) - devNum - testNum
            # 边
            if shuffle:
                pos_edges = copy.deepcopy(self.pos_edges)
                random.shuffle(pos_edges)
            else:
                pos_edges = self.pos_edges
        else:
            if not isinstance(customTrainSet, list):
                customTrainSet = customTrainSet.tolist()  # numpy 转 list
            train_set_S = set()  # 自定义边集合
            for i in customTrainSet:
                train_set_S.add((i[0], i[1]))
                train_set_S.add((i[1], i[0]))
            no_train_edges = [i for i in self.pos_edges if tuple(i) not in train_set_S]  # 非训练集边
            trainNum = len(customTrainSet)  # 训练集边数量
            # 测试集和验证集的边数量
            if devRate == testRate == 0:
                devNum = testNum = 0
            else:
                devNum = int(len(no_train_edges) * devRate / (devRate + testRate))
                testNum = len(no_train_edges) - devNum
            # 打乱 测试集和验证集
            if shuffle:
                random.shuffle(no_train_edges)
            pos_edges = customTrainSet + no_train_edges
        # 负例
        if shuffle_neg:
            neg_edges = random.sample(self.neg_edges, negNum * len(pos_edges))
        else:
            neg_edges = self.neg_edges[:negNum * len(pos_edges)]
        neg_edges = np.reshape(neg_edges, (-1, negNum, 2))
        # 数据
        s = 0
        self.data['trainPos'] = np.array(pos_edges[s:s + trainNum])
        self.data['trainNeg'] = np.array(neg_edges[s:s + trainNum])
        s += trainNum
        self.data['devPos'] = np.array(pos_edges[s:s + devNum])
        self.data['devNeg'] = np.array(neg_edges[s:s + devNum])
        s += devNum
        self.data['testPos'] = np.array(pos_edges[s:s + testNum])
        self.data['testNeg'] = np.array(neg_edges[s:s + testNum])
        # 补充非训练集
        if noTrainMinNum and noTrainMinNum > 0:
            concat = lambda x, y, n: np.concatenate((self.data[x], self.data[y][0:n]), axis=0) \
                if self.data[x].size > 0 else self.data[y][0:n]
            n = noTrainMinNum - len(self.data['devPos'])
            if n > 0:
                self.data['devPos'] = concat('devPos', 'trainPos', n)
                self.data['devNeg'] = concat('devNeg', 'trainNeg', n)
            n = noTrainMinNum - len(self.data['testPos'])
            if n > 0:
                self.data['testPos'] = concat('testPos', 'trainPos', n)
                self.data['testNeg'] = concat('testNeg', 'trainNeg', n)

    def getDataset(self):
        assert self.data['trainPos'] is not None, '还未生成数据集!'
        out = {
            'train_edges_pos': self.data['trainPos'],
            'train_edges_neg': self.data['trainNeg'],
            'dev_edges_pos': self.data['devPos'],
            'dev_edges_neg': self.data['devNeg'],
            'test_edges_pos': self.data['testPos'],
            'test_edges_neg': self.data['testNeg'],
            'adj_train': self.data['adj_train'],
            'features': self.getFeats(),
        }
        return out


class MixedDataset(DataHelper):
    def __init__(self, g: 随机图 = None, load_file=None, data: dict = None, shuffle_data=False,
                 main_task=Classification, regularization_task=(LinkPred,), **kwargs):
        """
        :param g:
        :param load_file: str; 使用后会忽略 main_task 和 regularization_task 参数
        :param data:
        :param shuffle_data: bool; 是否随机打乱数据集顺序
            用于 generateDataset 方法的 shuffle=False 情况下的初始随机划分数据集
        :param main_task: DataHelper; 主任务数据
        :param regularization_task: (DataHelper,..); 用于正则化项的辅助任务数据
        """
        self.data_mix = {'self': {}, 'main': {}, 'regular': []}  # 用于保存和读取数据, 加载需要 load_file 参数
        super(MixedDataset, self).__init__(g, load_file, data, **kwargs)
        self.data.update({'type': self.__class__.__name__})
        # 生成任务 DataHelper 实例, data 会有类似浅拷贝, dh之间不会直接共用 data
        if not load_file:
            dh_f = lambda dh: dh(g=g, data=self.data, shuffle_data=shuffle_data, print_stat=False, **kwargs)
            self._main_task_dh = dh_f(main_task)
            self._regular_task_dh_L = [dh_f(i) for i in regularization_task]
        else:
            dh_f = lambda dh, data: eval(dh)(data=data, shuffle_data=shuffle_data, print_stat=False, **kwargs)
            self._main_task_dh = dh_f(self.data_mix['main']['type'], self.data_mix['main'])
            self._regular_task_dh_L = [dh_f(i['type'], i) for i in self.data_mix['regular']]
        # generateDataset 方法的相关参数
        common_para = {'self', 'devRate', 'testRate', 'shuffle', 'noTrainMinNum', 'customTrainSet'}  # 常用参数
        main_task_para = set(  # 主任务生成数据需要参数
            inspect.getfullargspec(self._main_task_dh.generateDataset).args
        ) - common_para
        regular_task_para = set()  # 辅助任务生成数据需要参数
        for i in self._regular_task_dh_L:
            regular_task_para |= set(inspect.getfullargspec(i.generateDataset).args)
        regular_task_para -= common_para
        # 主任务辅助任务依赖的情况
        self._main_regular_cor_D = {  # {(dh1,dh2):getDataset返回参数中用于customTrainSet的参数,..}
            tuple(sorted([LinkPred.__name__, HypernymyRel.__name__])): 'train_edges_pos',
        }
        print(f'type: {self.data["type"]}; '
              f'generateDataset 特殊参数: '
              f'main[{self._main_task_dh.__class__.__name__}]{main_task_para}, '
              f'regular{[i.__class__.__name__ for i in self._regular_task_dh_L]}{regular_task_para}')

    def generateDataset(self, devRate, testRate, shuffle=False, noTrainMinNum=100, **kwargs):
        '''
        :param devRate: float; 验证集的比例
        :param testRate: float; 测试集的比例
        :param shuffle: bool; 是否打乱正例, 即每次都随机划分数据集
        :param noTrainMinNum: int; 辅助任务数据集中非训练集的最少数量, 用于帮助主任务跑通测试集和验证集. 考虑到停止策略问题, 数量不能太少
        :param kwargs: 除了以下参数其他参数需要斟酌, 防止主辅任务相同但不要同时用的参数
            等类别: 参见 Classification.generateDataset
            negNum: 参见 HypernymyRel.generateDataset
            dsMult: 参见 GraphDistor.generateDataset
            shuffle_neg: 参见 LinkPred.generateDataset / HypernymyRel.generateDataset
        :return:
        '''
        assert 1 > devRate + testRate >= 0, '数据集比例错误!'
        self._main_task_dh.generateDataset(devRate=devRate, testRate=testRate, shuffle=shuffle, **kwargs)
        for i in self._regular_task_dh_L:
            mn = tuple(sorted([self._main_task_dh.__class__.__name__, i.__class__.__name__]))  # 主辅任务名称对
            if mn in self._main_regular_cor_D:  # 是否有依赖关系, 如果打乱正例依赖关系就没有意义了
                customTrainSet = self._main_task_dh.getDataset()[self._main_regular_cor_D[mn]]
            else:
                customTrainSet = None
            i.generateDataset(devRate=0, testRate=0, shuffle=shuffle, customTrainSet=customTrainSet,
                              noTrainMinNum=noTrainMinNum, **kwargs)

    def getDataset(self):
        out = {'all_dh_name': []}  # 第一个数据集的名字是主任务数据集
        for dh in [self._main_task_dh, *self._regular_task_dh_L]:
            for k, v in dh.getDataset().items():
                out[f'{dh.data["type"]}_{k}'] = v
            out['all_dh_name'].append(dh.data["type"])
        # 防止一些非对应Decoder的方法抽取数据集失败, 比如 encoder.model(encoder.getData(dataset)). 20210318
        for k, v in self._main_task_dh.getDataset().items():
            out[k] = v
        return out

    @property
    def task_dh(self):
        """
        将所有任务对应的 data helper 实例化对象按顺序拿出来, 第一个是主任务. 尽量不要 shuffle=True, 否则破坏部分依赖的副任务训练的一致性(邻接矩阵)
        :return:
        """
        return [self._main_task_dh, *self._regular_task_dh_L]

    def 保存数据(self, path):
        '''
        :param path: str, 保存路径
        :return:
        '''
        self.data_mix['self'] = self.data
        self.data_mix['main'] = self._main_task_dh.data
        self.data_mix['regular'] = [i.data for i in self._regular_task_dh_L]
        buff = 10 ** 4
        bit = pkl.dumps(self.data_mix)
        with open(path, 'wb') as w:
            for i in range(0, len(bit), buff):
                w.write(bit[i:i + buff])

    def 读取数据(self, path):
        '''
        :param path: str, 保存路径
        :return:
        '''
        with open(path.encode('utf-8'), 'rb') as r:
            self.data_mix = pkl.load(r)
        self.data = self.data_mix['self']

    @staticmethod
    def test():
        文件路径 = 'ab_无标度网络;0;[];[];[];[];0;1023.pkl'
        图 = 随机图(文件路径)
        测试例子 = [
            [LinkPred, Classification, GraphDistor, HypernymyRel],
            [Classification, LinkPred, GraphDistor, HypernymyRel],
            [GraphDistor, Classification, LinkPred, HypernymyRel],
            [HypernymyRel, GraphDistor, Classification, LinkPred],
        ]
        for i, task in enumerate(测试例子):
            print('=' * 40, '测试', i + 1)
            print('-' * 20, '生成和保存数据:')
            dh = MixedDataset(图, shuffle_data=True, main_task=task[0], regularization_task=task[1:])
            dh.generateDataset(devRate=0.1, testRate=0.1, dsMult=10, 等类别=True, shuffle_neg=True, negNum=10)
            dh.保存数据(f'ac_{dh.__class__.__name__}.pkl')

            print('-' * 20, '从文件读取数据:')
            dh = MixedDataset(load_file=f'ac_{dh.__class__.__name__}.pkl')
            pprint({k: v.shape if isinstance(v, np.ndarray) else v for k, v in dh.getDataset().items()})
            pprint(dh.getDataset().keys())
            print()


def main():
    # 文件路径 = 'ab_不平衡多叉树;1;[1023];[23];[0.1706];[1.0021];1023;0.pkl'
    # 文件路径 = 'ab_完全多叉树;1;[1023];[10];[0.0];[1.0];1023;0.pkl'
    # 文件路径 = 'ab_高低多叉树;1;[1023];[7];[0.0226];[1.2875];1023;0.pkl'
    # 文件路径 = 'ab_低高多叉树;1;[1023];[9];[0.0034];[0.8981];1023;0.pkl'
    文件路径 = 'ab_混合树图;4;[511,511,511,511];[9,16,8,6];[0.0,0.1626,0.0028,0.0253];[1.0,1.0,0.8957,1.2588];1858;0.pkl'
    # 文件路径 = 'ab_无标度网络;0;[];[];[];[];0;1023.pkl'

    图 = 随机图(文件路径)
    dh = DataHelper
    # dh = LinkPred
    # dh = Classification
    # dh = GraphDistor
    # dh = HypernymyRel
    # dh = MixedDataset

    dataHelper = dh(图)
    dataHelper.generateDataset(devRate=0.1, testRate=0.1, shuffle=False)
    dataHelper.getDataset()
    dataHelper.保存数据(f'ac_{dh.__name__}.pkl')
    dataHelper.自动评估绘图(titleFrontText='$\\mathbb{R}$, ', 使用分类颜色=True, 使用树结构颜色=True, 使用层次颜色=True,
                      saveName=f'ac_{dataHelper.data["图统计信息"]["describe"]}.eps')


if __name__ == '__main__':
    # MixedDataset.test()
    main()
