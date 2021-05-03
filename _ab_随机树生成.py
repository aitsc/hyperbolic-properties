import math
from pprint import pprint
import pickle as pkl
from _ae_utils import *


def 快速均匀分割(n, array, acc=0.1, low=None, high=None):
    '''
    高效但可能不均衡. 一个分段大于中值的贪心策略可能导致靠后的节点数量较少
    :param n: int, 类别数量
    :param array: [x,..], 待分割数组
    :param acc: float, 分治法的二分精度
    :param low: 递归用参数
    :param high: 递归用参数
    :return: [[x,..],..], 分割结果结果
    '''
    assert 1 < n <= len(array), '划分数和总数不合适!'
    if not high:
        high = sum(array)
    if not low:
        low = max(array)
    if low > high:
        seg = []
        s = 0
        max_v = high + acc
        for no, i in enumerate(array):
            s += i
            if s > max_v or len(array) - no <= n - len(seg):  # 不能让类数减少
                s = i
                seg.append([i])
            else:
                if not seg:
                    seg.append([i])
                else:
                    seg[-1].append(i)
        return seg
    else:
        mid = (low + high) / 2
        seg = s = 0
        juge = True
        for no, i in enumerate(array):
            s += i
            if s > mid or len(array) - no <= n - seg - 1:
                s = i
                seg += 1
        if seg >= n:
            juge = False
        if juge:
            return 快速均匀分割(n, array, acc=acc, low=low, high=mid - acc)
        else:
            return 快速均匀分割(n, array, acc=acc, low=mid + acc, high=high)


def 准确均匀分割(n, array, *args):
    '''
    穷举速度慢, 但均衡.
    :param n: int, 类别数量
    :param array: [x,..], 待分割数组
    :param args:
    :return: [[x,..],..], 分割结果
    '''
    assert 1 < n <= len(array), '划分数和总数不合适!'
    nums = [i + 1 for i in range(n - 1)]
    nums_opt = nums
    σ_fake = sum(array)
    μ = σ_fake / n
    σ_fake *= n
    count = 0
    while True:
        count += 1
        D_ = 0
        for i in range(n):
            if i == 0:
                D_ += abs(sum(array[:nums[0]]) - μ)
            elif i == n - 1:
                D_ += abs(sum(array[nums[-1]:]) - μ)
            else:
                D_ += abs(sum(array[nums[i - 1]:nums[i]]) - μ)
        if D_ < σ_fake:
            σ_fake = D_
            nums_opt = [i for i in nums]
        add1pos = -1
        for i in range(n - 2, -1, -1):
            pos = nums[i]
            if pos + 1 + n - 2 - i < len(array):
                add1pos = i
                break
        if add1pos < 0:
            break
        nums[add1pos] += 1
        for i in range(add1pos + 1, n - 1):
            nums[i] = nums[i - 1] + 1
    array_L = []
    for i in range(n):
        if i == 0:
            array_L.append(array[:nums_opt[0]])
        elif i == n - 1:
            array_L.append(array[nums_opt[-1]:])
        else:
            array_L.append(array[nums_opt[i - 1]:nums_opt[i]])
    return array_L


均匀分割 = 准确均匀分割


class 随机图:
    class 类型:
        树 = '树'
        森林 = '森林'
        混合树图 = '混合树图'
        nx图 = 'nx图'

    def __init__(self, 文件路径=None):
        """
        增加类变量时注意修改保存和读取方法
        :param 文件路径: None or str; str则从文件中读取随机图
        """
        self.describe = None  # 描述
        self.type = None  # 随机图.类型
        # 一般节点编号从0开始(多树除外), 层数从1开始, 该树高度>=1
        # 如果存在大编号节点是小编号节点的父节点, 那么可能无法从返回排序的edges中确定root节点, 因此单树root节点尽量为0
        self.tree = []  # 所有树的列表, [[[节点编号,父,[子,..],层数,该树高度,该树节点数],..],..]=[[层次遍历节点1,..],..]=[树1,..]
        self.graph = None  # networkx的graph, 如果不为空则包含图所有信息
        self.triple = []  # 除了树中节点以外的三元组, [(节点1,节点2,权重),..]
        self.node_class_D = {}  # {节点:类别,..}, 一般类别是int从0开始不间断, 用于数据one-hot类别交叉熵
        self.node_alias_D = {}  # {节点:节点别名,..}, 可以用于绘图时候标记等
        if 文件路径:
            self.从文件读取(文件路径)

    @staticmethod
    def _树_节点坐标生成(tree, 缩放x=1., 缩放y=1., 平移x=0., 平移y=0.):
        '''
        横纵坐标基础长度属于 [0, len(tree)]
        :param tree: [[节点编号,父,[子,..],层数,该树高度,该树节点数],..]
        :param 缩放x: x轴缩放几倍
        :param 缩放y: y轴缩放几倍
        :param 平移x: x轴平移, 缩放之后
        :param 平移y: y轴平移, 缩放之后
        :return: [[点编号,x,y],..]
        '''
        点_序号D = {}
        for i, p in enumerate(tree):
            点_序号D[p[0]] = i
        点_xl_xr_x_y_L = [[i[0]] for i in tree]  # [[点编号,树左边界,树右边界,横坐标,纵坐标],..], 顺序与tree对应
        H = tree[0][4]  # 树总高度
        # 层次遍历递推节点坐标
        for i, p in enumerate(tree):
            if i == 0:
                点_xl_xr_x_y_L[0] = [p[0], 0, len(tree), len(tree) / 2, len(tree)]
            子树累计节点数L = []
            for sp in p[2]:
                sp = tree[点_序号D[sp]]
                if not 子树累计节点数L:
                    子树累计节点数L.append(sp[5])
                else:
                    子树累计节点数L.append(sp[5] + 子树累计节点数L[-1])
            for j, sp in enumerate(p[2]):  # 递推子节点坐标
                sp = tree[点_序号D[sp]]
                parent_xl = 点_xl_xr_x_y_L[点_序号D[p[0]]][1]
                parent_xr = 点_xl_xr_x_y_L[点_序号D[p[0]]][2]
                if j == 0:
                    xl = parent_xl
                else:
                    xl = 点_xl_xr_x_y_L[点_序号D[p[2][j - 1]]][2]
                xr = parent_xl + 子树累计节点数L[j] / 子树累计节点数L[-1] * (parent_xr - parent_xl)
                x = (xl + xr) / 2
                y = (H - sp[3]) / (H - 1) * len(tree)
                点_xl_xr_x_y_L[点_序号D[sp[0]]] = [sp[0], xl, xr, x, y]
        点_x_y_L = [[p[0], p[3] * 缩放x + 平移x, p[4] * 缩放y + 平移y] for p in 点_xl_xr_x_y_L]
        return 点_x_y_L

    def 从文件读取(self, 文件路径):
        with open(文件路径.encode('utf-8'), 'rb') as r:
            self.describe = pkl.load(r)
            self.type = pkl.load(r)
            self.tree = pkl.load(r)
            self.graph = pkl.load(r)
            self.triple = pkl.load(r)
            self.node_class_D = pkl.load(r)
            try:
                self.node_alias_D = pkl.load(r)
            except:
                ...

    def 保存到文件(self, 文件路径, 保留几位小数=4):
        dir = os.path.split(文件路径)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        后缀名 = os.path.splitext(文件路径)[1]
        if not 后缀名:  # 没有后缀名认为需要重命名, 路径记得最后加斜杠
            统计信息 = self.统计信息()
            name = self.describe
            name += ';' + str(统计信息['树数量'])
            name += ';' + str(统计信息['每棵树节点数量'])
            name += ';' + str(统计信息['每棵树的层数'])
            name += ';' + str([round(i, 保留几位小数) for i in 统计信息['每棵树不平衡程度']])
            name += ';' + str([round(i, 保留几位小数) for i in 统计信息['每棵树层次度分布的偏移度']])
            name += ';' + str(统计信息['树总节点数量'])
            name += ';' + str(统计信息['nx图节点数量'])
            name = name.replace(' ', '') + '.pkl'
            文件路径 += name
        with open(文件路径.encode('utf-8'), 'wb') as w:
            w.write(pkl.dumps(self.describe))
            w.write(pkl.dumps(self.type))
            w.write(pkl.dumps(self.tree))
            w.write(pkl.dumps(self.graph))
            w.write(pkl.dumps(self.triple))
            w.write(pkl.dumps(self.node_class_D))
            w.write(pkl.dumps(self.node_alias_D))
        return 文件路径

    def 树_计算子树高度与节点数(self, 强制=False):
        '''
        :param 强制: 是否重新计算
        :return: bool
        '''
        # 对于每一棵树
        for tree in self.tree:
            if not 强制:
                # 如果树没有节点
                if not tree:
                    continue
                # 如果节点信息够长, 但是中间有数值
                if len(tree[0]) >= 6 and tree[0][4] and tree[0][5]:
                    continue
            点_序号D = {}
            for i, p in enumerate(tree):
                点_序号D[p[0]] = i
            # 对于每一棵树的节点
            for i in range(len(tree) - 1, -1, -1):
                p = tree[i]
                树高度 = 0
                树节点数 = 1
                for j in p[2]:  # 回朔子节点
                    子树高度 = tree[点_序号D[j]][4]
                    子树节点数 = tree[点_序号D[j]][5]
                    树高度 = max(树高度, 子树高度)
                    树节点数 += 子树节点数
                树高度 += 1
                if len(p) == 4:
                    p += [树高度, 树节点数]
                elif len(p) == 5:
                    p[4] = 树高度
                    p.append(树节点数)
                elif len(p) >= 6:
                    p[4] = 树高度
                    p[5] = 树节点数
                else:
                    raise NameError('tree数据结构错误!', i)
        return True

    def 树_计算不平衡程度(self):
        '''
        :return: [IB,..]
        '''
        IB_L = []  # 每棵树的不平衡程度
        for tree in self.tree:
            点_序号D = {}
            for i, p in enumerate(tree):
                点_序号D[p[0]] = i
            IB = 0
            for p in tree:
                # 没有子节点或只有一个子节点则方差为0
                if len(p[2]) <= 1:
                    continue
                # 计算子节点高度均值
                μ = 0
                for sp in p[2]:
                    μ += tree[点_序号D[sp]][4]
                μ /= len(p[2])
                # 计算子节点高度方差
                D = 0
                for sp in p[2]:
                    D += (tree[点_序号D[sp]][4] - μ) ** 2
                D /= len(p[2])
                IB += D ** 0.5
            IB /= len(tree)
            IB = 2 / (1 + math.e ** -IB) - 1
            IB_L.append(IB)
        return IB_L

    # def 树_计算层次度分布的偏移度(self):
    #     '''
    #     :return: [ID,..]
    #     '''
    #     ID_L = []  # 每棵树的层次度分布的均匀度
    #     for tree in self.tree:
    #         H = tree[-1][3]
    #         # 计算每层 所有子节点数大于0的节点的平均子节点数
    #         层_dv = [0] * (H - 1)  # [dv1,..], dv表示第l-1层所有子节点数大于0的节点的平均子节点数
    #         层_节点数 = [0] * (H - 1)
    #         for p in tree:
    #             if len(p[2]) > 0:
    #                 层_dv[p[3] - 1] += len(p[2])
    #                 层_节点数[p[3] - 1] += 1
    #         层_dv = [层_dv[i] / 层_节点数[i] for i in range(H - 1)]
    #         ID = H * sum(层_dv) / sum([(i + 1) * j for i, j in enumerate(层_dv)]) / 2
    #         ID_L.append(ID)
    #     return ID_L

    def 树_计算层次度分布的偏移度(self):
        '''
        :return: [ID,..]
        '''
        ID_L = []  # 每棵树的层次度分布的均匀度
        for tree in self.tree:
            H = tree[-1][3]
            # 计算每层 所有子节点数大于0的节点的平均子节点数
            层_dv = [0] * (H - 1)  # [dv1,..], dv表示第l-1层所有子节点数大于0的节点的平均子节点数
            层_节点数 = [0] * (H - 1)
            for p in tree:
                if len(p[2]) > 0:
                    层_dv[p[3] - 1] += len(p[2])
                    层_节点数[p[3] - 1] += 1
            层_dv = [层_dv[i] / 层_节点数[i] for i in range(H - 1)]
            avg_dv = sum(层_dv) / len(层_dv)
            ID = (sum([(dv - avg_dv) ** 2 for dv in 层_dv]) / len(层_dv))  # 先是方差
            # ID = 2 / (1 + math.e ** - ID) - 1
            DCG = iDCG = bDCG = 0
            for i, (dv, dv_o, dv_r) in enumerate(zip(层_dv, sorted(层_dv), sorted(层_dv, reverse=True))):
                l = i + 1
                DCG += (2 ** dv - 1) / math.log(l + 1, 2)
                iDCG += (2 ** dv_r - 1) / math.log(l + 1, 2)
                bDCG += (2 ** dv_o - 1) / math.log(l + 1, 2)
            if iDCG - bDCG == 0:
                ID = 0
            else:
                ID *= ((DCG - bDCG) / (iDCG - bDCG) - 0.5)
            ID = 1 / (1 + math.e ** -ID)
            ID_L.append(ID)
        return ID_L

    def 树_统计所有点和边(self):
        '''
        树序号 从0开始
        :return: ([点编号,..],[(小编号,大编号,树重合数),..],{点编号:[树序号,..],..})
        '''
        点编号_树D = {}
        边三元组D = {}  # {(小编号,大编号): 重合数,..}
        for i, t in enumerate(self.tree):
            for p in t:
                if p[0] in 点编号_树D:
                    点编号_树D[p[0]].append(i)
                else:
                    点编号_树D[p[0]] = [i]
                for sp in p[2]:
                    edge = tuple(sorted([p[0], sp]))
                    if edge in 边三元组D:
                        边三元组D[edge] += 1
                    else:
                        边三元组D[edge] = 1
        点编号L = [i for i, j in 点编号_树D.items()]
        点编号L.sort()
        边三元组L = [(k[0], k[1], v) for k, v in 边三元组D.items()]
        return 点编号L, 边三元组L, 点编号_树D

    def 树_获取有向边(self):
        '''
        获得所有树的有向边
        :return: [[(父节点编号,子节点编号),..],..]=[树1,..]
        '''
        树_边二元组 = []  # [[(父节点编号,子节点编号),..],..]
        for i, t in enumerate(self.tree):
            树_边二元组.append([])
            for p in t:
                树_边二元组[-1] += [(p[0], sp) for sp in p[2]]
        return 树_边二元组

    def 树_融合树(self, tree, 节点重叠率μ=0.8, 节点重叠率σ=0.2):
        '''
        融合后属于混合树图
        :param tree: [[节点编号,父,[子,..],层数],..]
        :param 节点重叠率μ: 正态分布均值
        :param 节点重叠率σ: 标准差
        :return: {节点编号:新编号,..}
        '''
        assert 0 <= 节点重叠率μ + 节点重叠率σ <= 1, '不满足 0 <= 节点重叠率μ + 节点重叠率σ <= 1 !'
        节点重叠率 = np.random.normal(节点重叠率μ, 节点重叠率σ)
        while 节点重叠率 < 0 or 节点重叠率 > 1:
            节点重叠率 = np.random.normal(节点重叠率μ, 节点重叠率σ)
        # 已有节点编号
        已有节点编号S = set()
        for t in self.tree:
            for p in t:
                已有节点编号S.add(p[0])
        已有最大节点编号 = max(已有节点编号S) if 已有节点编号S else -1
        已有节点编号L = list(已有节点编号S)
        np.random.shuffle(已有节点编号L)  # 已有节点编号L 随机
        # 新节点编号
        新节点编号L = [p[0] for p in tree]
        np.random.shuffle(新节点编号L)  # 打乱加入树的节点编号, 和原来树的打乱的节点匹配, 双向随机
        节点重叠数 = int(min(len(已有节点编号L), len(tree)) * 节点重叠率)
        新节点编号_重叠S = set(新节点编号L[:节点重叠数])
        # 开始重新编号
        点_新编号D = {}
        for p in tree:
            if p[0] in 新节点编号_重叠S:
                编号 = 已有节点编号L.pop(0)  # 已有节点编号L 最后接近空
            else:
                编号 = 已有最大节点编号 + 1
                已有最大节点编号 += 1
            点_新编号D[p[0]] = 编号
        # 构建新树
        tree_重编号 = []  # [[节点编号,父,[子,..],层数],..]
        for p in tree:
            新编号 = 点_新编号D[p[0]]
            父编号 = 点_新编号D[p[1]] if p[1] is not None else None
            子编号L = [点_新编号D[i] for i in p[2]]
            tree_重编号.append([新编号, 父编号, 子编号L] + p[3:])
        self.tree.append(tree_重编号)
        return 点_新编号D

    def 节点类别分配(self, n, 分配方式='branch', 允许不同层次=False, 强制=False, 最大σμ比=None, 输出提示=False, 使用快速均匀分割=80):
        '''
        混合树图branch分配中, 根节点可能也会分配类别, 因为其可能恰好是别树子节点. 类别从0开始
        :param n: 类别数, >1
        :param 分配方式: 随机 random 或 按分支 branch, 选择第一个. nx图强制随机
        :param 允许不同层次: 允许非同一棵子树上的节点不同层次
        :param 强制: 如果已经分配过分类, 是否重新分配
        :param 最大σμ比: float, 需要 允许不同层次=True, 用于限制分类后节点数量的方差大小, 0.2左右比较适合
        :param 输出提示: bool, 是否输出中间的提示
        :param 使用快速均匀分割: int, len(层次节点编号_树序号L) > 使用快速均匀分割 时使用快速均匀分割,防止分割速度过慢
        :return:
        '''
        if self.type == self.类型.nx图:  # nx图只有随机
            nodes = self.graph.nodes
            if not 强制 and len(self.node_class_D) == len(nodes):  # 已经分配过
                return False
            node_class_D = {i: int(random.random() * n) for i in nodes}  # 类别从0开始
            self.node_class_D = node_class_D
            return True
        点编号S = set()
        for t in self.tree:
            for p in t:
                点编号S.add(p[0])
        if not 强制 and len(self.node_class_D) == len(点编号S):  # 已经分配过
            return False
        if 分配方式 == 'random':
            node_class_D = {i: int(random.random() * n) for i in 点编号S}  # 类别从0开始
            self.node_class_D = node_class_D
            return True
        if 分配方式 == 'branch':
            点_序号D_L = []  # 用于子节点追溯
            for tree in self.tree:
                点_序号D = {}
                for i, p in enumerate(tree):
                    点_序号D[p[0]] = i
                点_序号D_L.append(点_序号D)
            # 层次统计
            self.树_计算子树高度与节点数()
            层次节点编号_树序号L = [[tree[0][0], i, tree[0][5]] for i, tree in
                           enumerate(self.tree)]  # [(节点编号,树序号,该树节点数),..], 树序号从0开始
            if 允许不同层次 and 最大σμ比 and 0 < 最大σμ比:
                σμ比 = 最大σμ比 + 1
                while 最大σμ比 < σμ比:
                    max_nodes_num = 0  # 最大子树节点数
                    max_nodes_i = 0  # 位置
                    for k, (p, i, h) in enumerate(层次节点编号_树序号L):
                        if h > max_nodes_num:
                            max_nodes_num = h
                            max_nodes_i = k
                    assert max_nodes_num > 1, '子树不可再分!'
                    # 插入对应子节点
                    p, i, h = 层次节点编号_树序号L[max_nodes_i]
                    del 层次节点编号_树序号L[max_nodes_i]
                    sp = self.tree[i][点_序号D_L[i][p]][2]
                    x = []
                    for spi in sp:
                        x.append([spi, i, self.tree[i][点_序号D_L[i][spi]][5]])
                    层次节点编号_树序号L = 层次节点编号_树序号L[:max_nodes_i] + x + 层次节点编号_树序号L[max_nodes_i:]
                    # 计算 σμ比
                    if len(层次节点编号_树序号L) < n:
                        continue
                    else:
                        seg = 快速均匀分割(n, [i[2] for i in 层次节点编号_树序号L])
                        seg_num = [sum(i) for i in seg]
                        μ = sum(seg_num) / len(seg_num)
                        σ = (sum([(i - μ) ** 2 for i in seg_num]) / len(seg_num)) ** 0.5
                        σμ比 = σ / μ
                    if 输出提示:
                        print('\t计算 合适的σμ比...', σμ比)
            else:
                while len(层次节点编号_树序号L) < n:
                    new_层次节点编号_树序号L = []
                    for p, i, h in 层次节点编号_树序号L:
                        sp = self.tree[i][点_序号D_L[i][p]][2]
                        if not sp:
                            if 允许不同层次:
                                new_层次节点编号_树序号L.append([p, i, h])
                            continue
                        for spi in sp:
                            new_层次节点编号_树序号L.append([spi, i, self.tree[i][点_序号D_L[i][spi]][5]])
                    层次节点编号_树序号L = new_层次节点编号_树序号L
            assert len(层次节点编号_树序号L) >= n, '类别比一层的节点多!'
            # 划分节点类别
            if 输出提示:
                print('\t划分节点类别... 总划分段数:', len(层次节点编号_树序号L))
            if len(层次节点编号_树序号L) > 使用快速均匀分割:
                seg = 快速均匀分割(n, [i[2] for i in 层次节点编号_树序号L])
            else:
                seg = 均匀分割(n, [i[2] for i in 层次节点编号_树序号L])
            # 点编号
            if 输出提示:
                print('\t合成 node_class_D...')
            node_class_D = {i: [] for i in 点编号S}  # 最上面节点无类别为 None
            累计节点数 = 0
            for c, s in enumerate([len(i) for i in seg]):
                for i in range(累计节点数, 累计节点数 + s):
                    点编号 = 层次节点编号_树序号L[i][0]
                    树序号 = 层次节点编号_树序号L[i][1]
                    p = self.tree[树序号][点_序号D_L[树序号][点编号]][0]
                    p_队列 = [p]
                    while p_队列:
                        p = p_队列.pop(0)
                        node_class_D[p].append(c)
                        sp = self.tree[树序号][点_序号D_L[树序号][p]][2]
                        p_队列 += sp
                累计节点数 += s
            # 多分类节点不分配类别
            self.node_class_D = {node: c[0] if len(c) == 1 else None for node, c in node_class_D.items()}
            return True
        return False

    def _树_绘图(self, ns=2, nc='#1E78B3', 点编号_颜色D=None):
        '''
        :param ns: 点大小
        :param nc: str or list, 点的颜色
        :param 点编号_颜色D: {点编号:颜色,..}
        :return:
        '''
        plt.figure(100)
        self.树_计算子树高度与节点数()  # 用于 树_节点坐标生成
        tree_num = len(self.tree)
        for i, tree in enumerate(self.tree):
            长 = int(tree_num ** 0.5)
            plt.subplot(长, math.ceil(tree_num / 长), i + 1)
            plt.xticks([])  # 去掉横坐标值
            plt.yticks([])  # 去掉纵坐标值
            plt.axis('off')  # 关闭坐标
            # 点
            点_x_y_L = self._树_节点坐标生成(tree)
            点_x_y_D = {i[0]: i[1:] for i in 点_x_y_L}
            # 颜色
            if 点编号_颜色D:
                colors = []
                for p in 点_x_y_L:
                    colors.append(点编号_颜色D[p[0]])
            else:
                colors = nc
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
        plt.show()

    def 绘图(self, length=10, width=10, ns=2, nc='#1E78B3', 按类别颜色画点=False):
        '''
        :param length: 画布高度, 单位100像素
        :param width: 画布宽度, 单位100像素
        :param ns: 点大小
        :param nc: str or list, 点的颜色
        :param 按类别颜色画点: bool, 否则混合树图按树颜色画点
        :return:
        '''
        if not self.type:
            print('没有图数据!')
        plt.rcParams['figure.figsize'] = (length, width)
        if not self.node_class_D:
            按类别颜色画点 = False
        if 按类别颜色画点:
            class_s = set(i[1] for i in self.node_class_D.items())  # 如果有则包含无分类节点, 分类从0开始
            class_num = len(class_s)
            print(('类别数量: %d ' % class_num) + ('(包含None)' if None in class_s else '(无None)'))
            if isinstance(nc, list) and len(nc) == class_num:
                cs = nc  # 所有颜色
            else:
                cs = ncolors(class_num)
            点编号_颜色D = {}
            for n, c in self.node_class_D.items():
                if c is None:
                    点编号_颜色D[n] = cs[0]
                else:
                    if None in class_s:
                        点编号_颜色D[n] = cs[c + 1]
                    else:  # 如果只有一棵树并没有空分类
                        点编号_颜色D[n] = cs[c]
        # 单棵树: 完全多叉树, 不平衡多叉树, 低高叉树, 高低叉树
        if self.type == self.类型.树:
            if 按类别颜色画点:
                self._树_绘图(ns, 点编号_颜色D=点编号_颜色D)
            else:
                self._树_绘图(ns, nc)
        # 混合树图
        elif self.type == self.类型.混合树图:
            点编号L, 边三元组L, 点编号_树D = self.树_统计所有点和边()
            tree_num = len(self.tree)
            node_c = []
            if 按类别颜色画点:
                for p in 点编号L:
                    node_c.append(点编号_颜色D[p])
            else:  # 按树颜色画点
                if isinstance(nc, list) and len(nc) == tree_num + 1:
                    cs = nc  # 所有颜色
                else:
                    cs = ncolors(tree_num + 1)
                for p in 点编号L:
                    if len(点编号_树D[p]) == 1:
                        node_c.append(cs[点编号_树D[p][0] + 1])
                    else:
                        node_c.append(cs[0])
                点编号_颜色D = {点编号L[i]: node_c[i] for i in range(len(node_c))}
            # 绘制所有单个树
            self._树_绘图(ns, 点编号_颜色D=点编号_颜色D)
            # 绘图
            plt.figure(1)
            g = nx.Graph()
            g.add_nodes_from(点编号L)
            g.add_weighted_edges_from(边三元组L)
            pos = nx.spring_layout(g)
            nx.draw(g, pos, node_size=ns, linewidths=1, node_color=node_c)
        # 图: 无标度网 等
        elif self.type == self.类型.nx图:
            if 按类别颜色画点:
                node_c = []
                for p in self.graph.nodes:
                    node_c.append(点编号_颜色D[p])
            else:
                node_c = nc
            # 绘图
            plt.figure(1)
            if self.describe == '小世界网络':
                pos = nx.circular_layout(self.graph)
            elif self.describe == 'ER随机图':
                pos = nx.kamada_kawai_layout(self.graph)
            elif self.describe == '规则图':
                pos = nx.kamada_kawai_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)
            nx.draw(self.graph, pos, node_size=ns, linewidths=1, node_color=node_c)
        plt.show()
        plt.close()  # 防止图片叠加变大, 无界面的Linux

    def 统计信息(self):
        out = {'describe': self.describe}
        out['树数量'] = len(self.tree)
        out['每棵树节点数量'] = [len(tree) for tree in self.tree]
        out['每棵树的层数'] = [tree[-1][3] for tree in self.tree]
        out['每棵树不平衡程度'] = self.树_计算不平衡程度()
        out['每棵树层次度分布的偏移度'] = self.树_计算层次度分布的偏移度()
        点编号S = set()
        for t in self.tree:
            for p in t:
                点编号S.add(p[0])
        树总节点数量 = len(点编号S)
        out['树总节点数量'] = 树总节点数量
        if self.graph:
            nodes = self.graph.nodes
            edges = self.graph.edges
            out['nx图节点数量'] = len(nodes)
            out['nx图边数量'] = len(edges)
            g = self.graph
        else:
            out['nx图节点数量'] = 0
            out['nx图边数量'] = 0
            g = nx.Graph()
            g.add_weighted_edges_from(self.树_统计所有点和边()[1] + list(self.triple))
        # 计算度信息
        degree = [i[1] for i in g.degree]
        degree_avg = sum(degree) / len(degree)
        out['度均值'] = degree_avg
        out['度最大值'] = max(degree)
        out['度最小值'] = min(degree)
        out['度标准差'] = (sum([(i - degree_avg) ** 2 for i in degree]) / len(degree)) ** 0.5
        return out


class 生成随机图:
    @staticmethod
    def 完全多叉树(n, m=2):
        '''
        :param n: 节点数量
        :param m: 分叉数
        :return:
        '''
        RG = 随机图()
        RG.type = RG.类型.树
        RG.describe = '完全多叉树'
        点生成队列 = [[0, None, [], 1]]
        当前节点编号 = 0
        tree = [点生成队列[0]]  # [[节点编号,父,[子,..],层数],..]
        while len(tree) < n:
            x = 点生成队列.pop(0)  # 第一个点
            for i in range(min(m, n - len(tree))):
                当前节点编号 += 1
                x[2].append(当前节点编号)
                y = [当前节点编号, x[0], [], x[3] + 1]
                点生成队列.append(y)
                tree.append(y)
        RG.tree.append(tree)
        RG.树_计算子树高度与节点数()
        return RG

    @staticmethod
    def 不平衡高低树(n, αr=1., αt=1., βμs=2, βμe=2, βtμ=1., βσs=0., βσe=0., βtσ=1., describe='不平衡高低树'):
        '''
        :param n: 节点数量
        :param αr: 越接近0表示生成的树越不平衡
        :param αt: =1表示不平衡线性衰减, >1表示左侧不平衡衰减慢, <1表示左侧不平衡衰减快
        :param βμs: 根节点生成子节点数量的均值
        :param βμe: 最后一个非叶子节点生成子节点数量的均值
        :param βtμ: =1表示均值线性变化, >1表示上方均值变化慢, <1表示上方均值变化快
        :param βσs: 根节点生成子节点数量的标准差
        :param βσe: 最后一个非叶子节点生成子节点数量的标准差
        :param βtσ: =1表示标准差线性变化, >1表示上方标准差变化慢, <1表示上方标准差变化快
        :param describe: 描述类型, 比如 不平衡高低树/低高多叉树/高低多叉树
        :return:
        '''
        RG = 随机图()
        RG.type = RG.类型.树
        RG.describe = describe
        当前节点编号 = 0
        遍历点编号 = 0
        tree = [[0, None, [], 1, 1]]  # [[节点编号,父,[子,..],层数,子节点保留概率],..], 层次遍历
        while 当前节点编号 < n - 1:
            x = tree[遍历点编号]  # 第一个点
            遍历点编号 += 1
            if x[4] < random.random():  # 不生成子节点
                continue
            # 计算子节点生成数量
            μ = βμs + ((len(tree) - 1) / (n - 1)) ** βtμ * (βμe - βμs)
            σ = βσs + ((len(tree) - 1) / (n - 1)) ** βtσ * (βσe - βσs)
            m = np.random.normal(μ, σ) + 0.5
            while m < 0 <= μ:  # 必须大于0
                m = np.random.normal(μ, σ) + 0.5
            m = int(m)  # 下取整
            for i in range(min(m, n - len(tree))):
                # 计算子节点保留概率
                if m <= 1:
                    Pr = x[4]
                else:
                    Pr = (αr + ((m - i - 1) / (m - 1)) ** αt * (1 - αr)) * x[4]
                当前节点编号 += 1
                x[2].append(当前节点编号)
                y = [当前节点编号, x[0], [], x[3] + 1, Pr]
                tree.append(y)
        RG.tree.append([i[:4] for i in tree])
        RG.树_计算子树高度与节点数()
        return RG

    @staticmethod
    def 混合树图(RG_L, μ=0.1, σ=0.1):
        '''
        :param RG_L: [随机图,..]
        :param μ: 节点重叠率的正态分布均值
        :param σ: 节点重叠率的标准差
        :return:
        '''
        RG = 随机图()
        RG.type = RG.类型.混合树图
        RG.describe = '混合树图'
        for i, 图 in enumerate(RG_L):
            assert len(图.tree) == 1, '随机图必须是一棵树!, %d' % i
            tree = 图.tree[0]
            RG.树_融合树(tree, μ, σ)
        return RG

    @staticmethod
    def 无标度图(n, m, p, q):
        '''
        :param n: Number of nodes
        :param m: Number of edges with which a new node attaches to existing nodes
        :param p: Probability value for adding an edge between existing nodes. p + q < 1
        :param q: Probability value of rewiring of existing edges. p + q < 1
        :return:
        '''
        RG = 随机图()
        RG.type = RG.类型.nx图
        RG.describe = '无标度网络'
        g = nx.generators.random_graphs.extended_barabasi_albert_graph(n, m, p, q)
        # 移除孤立节点
        # g.remove_nodes_from(list(nx.isolates(g)))
        # 只取最大连通子图
        g = g.subgraph(max(nx.connected_components(g), key=len))
        # 重新编号节点, 从0开始
        g = nx.relabel_nodes(g, {o: i for i, o in enumerate(sorted(g.nodes()))})
        RG.graph = g
        return RG

    @staticmethod
    def 小世界网络(n, k, p):
        RG = 随机图()
        RG.type = RG.类型.nx图
        RG.describe = '小世界网络'
        g = nx.generators.random_graphs.watts_strogatz_graph(n, k, p)
        # 只取最大连通子图
        g = g.subgraph(max(nx.connected_components(g), key=len))
        # 重新编号节点, 从0开始
        g = nx.relabel_nodes(g, {o: i for i, o in enumerate(sorted(g.nodes()))})
        RG.graph = g
        return RG

    @staticmethod
    def ER随机图(n, p):
        RG = 随机图()
        RG.type = RG.类型.nx图
        RG.describe = 'ER随机图'
        g = nx.generators.random_graphs.erdos_renyi_graph(n, p)
        # 只取最大连通子图
        g = g.subgraph(max(nx.connected_components(g), key=len))
        # 重新编号节点, 从0开始
        g = nx.relabel_nodes(g, {o: i for i, o in enumerate(sorted(g.nodes()))})
        RG.graph = g
        return RG

    @staticmethod
    def 规则图(d, n):
        if (n * d) % 2 != 0:
            n += 1
        RG = 随机图()
        RG.type = RG.类型.nx图
        RG.describe = '规则图'
        g = nx.generators.random_graphs.random_regular_graph(d, n)
        # 只取最大连通子图
        g = g.subgraph(max(nx.connected_components(g), key=len))
        # 重新编号节点, 从0开始
        g = nx.relabel_nodes(g, {o: i for i, o in enumerate(sorted(g.nodes()))})
        RG.graph = g
        return RG


if __name__ == '__main__':
    保存 = True

    if 保存:
        n = 3280

        # 图 = 生成随机图.无标度图(n, 2, 0.2, 0.2)
        # 图 = 生成随机图.小世界网络(n, 3, 0.4)
        # 图 = 生成随机图.ER随机图(n, 2/n)
        # 图 = 生成随机图.规则图(3, n)
        # 图 = 生成随机图.完全多叉树(n, 3)
        # 图 = 生成随机图.不平衡高低树(n, αr=0.2, αt=2, βμs=5, βμe=5, describe='不平衡树')
        图 = 生成随机图.不平衡高低树(n, βμs=2, βμe=7, βtμ=1., βσs=0.4, βσe=1.5, βtσ=1., describe='低高多叉树')
        # 图 = 生成随机图.不平衡高低树(n, βμs=6, βμe=1, βtμ=0.3, βσs=0.1, βσe=1, βtσ=1., describe='高低多叉树')

        # n = int(n / 2)
        # 图1 = 生成随机图.完全多叉树(n, 2)
        # 图2 = 生成随机图.不平衡高低树(n, αr=0.2, αt=2, βμs=5, βμe=5, describe='不平衡多叉树')
        # 图3 = 生成随机图.不平衡高低树(n, βμs=2, βμe=5, βtμ=1., βσs=0.4, βσe=2, βtσ=1., describe='低高多叉树')
        # 图4 = 生成随机图.不平衡高低树(n, βμs=6, βμe=1, βtμ=0.3, βσs=0.1, βσe=1, βtσ=1., describe='高低多叉树')
        # 图 = 生成随机图.混合树图([图1, 图2, 图3, 图4], μ=0.1, σ=0.1)

        # 图.节点类别分配(n=6, 分配方式='branch', 允许不同层次=True, 最大σμ比=0.2)
        图.绘图(按类别颜色画点=True)
        pprint(图.统计信息())
        # print('保存路径:', 图.保存到文件('ab_test/ab_'))

        # 以下用于调参, 获得合适的IB/ID
        # p = (1, 1, 2, 15)
        # IB_L = []
        # ID_L = []
        # t = time.time()
        # for i in range(100):
        #     图 = 生成随机图.不平衡高低树(n, αr=p[0], αt=p[1], βμs=p[2], βμe=p[3], describe='不平衡高低树')
        #     IB_L.append(图.树_计算不平衡程度()[0])
        #     ID_L.append(图.树_计算层次度分布的偏移度()[0])
        # print(f'{str(p)},  # IB:', round(sum(IB_L) / len(IB_L), 4), '; ID:', round(sum(ID_L) / len(ID_L), 4))
        # print(time.time() - t)
    else:
        文件路径 = 'ab_不平衡多叉树;1;[1023];[23];[0.1706];[1.0021];1023;0.pkl'
        图 = 随机图(文件路径)
        图.绘图(按类别颜色画点=True)
        pprint(图.统计信息())
