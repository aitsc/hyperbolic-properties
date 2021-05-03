from _ac_data_helper import *
import pandas
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from networkx.drawing.nx_agraph import graphviz_layout


def isSubsequence(s, t) -> bool:
    """
    判断 s 是否为 t 的子序列
    :param s: str or list or tuple
    :param t: str or list or tuple
    :return: bool
    """
    index = 0
    for value in s:
        if isinstance(t, str):
            index = t.find(value, index)
        else:
            try:
                index = t.index(value, index)
            except:
                index = -1
        if index == -1:
            return False
        index += 1
    return True


def 获取wordnet所有名词():
    """
    :return: pandas.DataFrame; [['id1', 'id2'],[下位词,上位词],..], 不包括 .n. 后缀, 后缀有 .01 会自动去掉
    """

    def 名字清洗(t: str):
        t = t.replace('.n.01', '')
        t = t.replace('.n.', '.')
        return t

    edges = set()
    for synset in tqdm(wn.all_synsets(pos='n'), '获取wordnet所有名词'):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))

        # also write transitive closure for all instances of a synset
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))
    edges = set((名字清洗(i), 名字清洗(j)) for i, j in edges)
    nouns = pandas.DataFrame(list(edges), columns=['id1', 'id2'])
    return nouns


def 提取子集(nouns, sub_root='animal'):
    """
    :param nouns:
    :param sub_root: str
    :return: {(上位词,下位词),..}
    """
    print('提取子集...', sub_root)
    sub_set = set(nouns[nouns.id2 == sub_root].id1.unique())
    sub_set.add(sub_root)  # 用于 nouns.id2.isin
    sub = nouns[nouns.id1.isin(sub_set) & nouns.id2.isin(sub_set)]
    # {(上位词,下位词),..}
    sub = {(i[1]['id2'], i[1]['id1']) for i in sub.iterrows()}
    return sub


def 构建树形图(上位词_下位词_L, root='mammal', 加速=True):
    """
    只保留根节点到某节点的最长简单路径
    :param 上位词_下位词_L: [(上位词,下位词),..] or {(上位词,下位词),..}; 保证是联通的
    :param root: str; 根节点名称, 根节点不能有入度
    :param 加速: bool; 使用加速将不会计算入度超过1的点和路径,以及路径总数
    :return:
    """
    print('构建树形图...')
    graph = nx.DiGraph()
    graph.add_edges_from(上位词_下位词_L)
    print('初始点数量:', len(list(graph)), '初始边数量:', len(list(graph.edges)))
    if 加速:
        # 入度排序的点计算
        in_degree = sorted(graph.in_degree(), key=lambda t: t[1], reverse=True)
        point = [i[0] for i in in_degree if i[0] != root and i[1] > 1]
        point_del = set()
        for i in tqdm(point, '删除非最长简单路径上点的其他入度'):
            if i in point_del:
                continue
            paths = list(nx.algorithms.all_simple_edge_paths(graph, root, i))  # 所有路径
            longest_path = set(max(paths, key=lambda t: len(t)))  # 最长路径
            lpp = set(sum(longest_path, ()))  # 最长路径上的点
            father_points = set(j for j in sum(paths, []) if j[1] in lpp) - longest_path
            graph.remove_edges_from(father_points)
            point_del |= lpp
        # points = set(graph) - {root}
        # for _ in tqdm(graph, '删除非最长简单路径上点的其他入度'):
        #     i, in_degree = max(graph.in_degree(points), key=lambda t: t[1])
        #     if in_degree <= 1:
        #         break
        #     paths = list(nx.algorithms.all_simple_edge_paths(graph, root, i))  # 所有路径
        #     longest_path = set(max(paths, key=lambda t: len(t)))  # 最长路径
        #     lpp = set(sum(longest_path, ()))  # 最长路径上的点
        #     father_points = set(j for j in sum(paths, []) if j[1] in lpp) - longest_path
        #     graph.remove_edges_from(father_points)
        #     points -= lpp
    else:
        # 获取标准路径
        all_path = set()
        路径总数 = 0
        入度超过1的点和路径 = []  # [[点,[(路径),..]],..]
        for i in tqdm(set(graph) - {root}, '计算每个点最合适的简单路径'):
            path = []  # [(路径),..]
            for p in nx.algorithms.all_simple_paths(graph, root, i):
                路径总数 += 1
                remove = False
                path_ = []
                for k in path:
                    if isSubsequence(p, k):
                        remove = True
                        break
                    if not isSubsequence(k, p):
                        path_.append(tuple(k))
                if remove:
                    continue
                path_.append(tuple(p))
                path = path_
            path = sorted(path, key=lambda t: len(t), reverse=True)
            if len(path) > 1:
                父点 = set(j[-2] for j in path)
                if len(父点) > 1:
                    入度超过1的点和路径.append([i, path])
            path = path[0:1]  # 删除大于1条的路径
            all_path |= set(path)
        # 入度超过1的点和路径
        print('路径总数:', 路径总数, '有效路径:', len(all_path), '入度超过1的点和路径:', len(入度超过1的点和路径))
        for i in 入度超过1的点和路径:
            print(i[0], len(i[1]), i[1])
        # 构建新图
        graph = nx.DiGraph()
        for path in all_path:
            for i in range(1, len(path)):
                graph.add_edge(path[i - 1], path[i])
    print('点数量:', len(list(graph)), '边数量:', len(list(graph.edges)))
    return graph


def 生成树与数据集(nouns, sub_root, class_num=6):
    """
    :param nouns:
    :param sub_root: str; 根节点名称
    :param class_num: int; 划分的类别数量
    :return:
    """
    根节点名 = sub_root
    graph = 构建树形图(提取子集(nouns, sub_root), root=根节点名)
    print('graphviz_layout绘图...')
    # 颜色
    node_color = ['#1f78b4'] * len(graph)
    node_color[list(graph).index(根节点名)] = 'r'
    # 节点大小
    node_size = [1] * len(graph)
    node_size[list(graph).index(根节点名)] = 2
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.set_title(f'WordNet.{根节点名}, {len(graph)} points')
    nx.draw(graph, node_size=node_size, with_labels=True, pos=graphviz_layout(graph, prog='twopi'), width=0.2,
            arrowsize=1, font_size=1, edge_color='gray', node_color=node_color, ax=ax)
    plt.tight_layout()
    plt.savefig(f'aj_{根节点名}_twopi.eps', format='eps')
    plt.show()
    plt.close()

    node_alias_D = {0: 根节点名}  # {节点:节点别名,..}
    alias_node_D = {根节点名: 0}  # {节点别名:节点,..}
    for i, p in enumerate(set(graph) - {根节点名}):
        node_alias_D[i + 1] = p
        alias_node_D[p] = i + 1

    点_子点D = {}  # {点编号:[子点编号,..],..}
    点编号S = set(range(len(node_alias_D)))
    for x in graph.edges:
        i, j = x[:2]
        i, j = alias_node_D[i], alias_node_D[j]
        点编号S.discard(j)
        if i in 点_子点D:
            点_子点D[i].append(j)
        else:
            点_子点D[i] = [j]
    assert len(点编号S) == 1, '根节点错误, 可能不是树! ' + str(点编号S)
    根节点 = 点编号S.pop()

    print('树生成...')
    RG = 随机图()
    RG.type = RG.类型.树
    RG.describe = 'wordnet.' + 根节点名
    点生成队列 = [[根节点, None, 点_子点D[根节点], 1]]
    tree = [点生成队列[0]]  # [[节点编号,父,[子,..],层数],..]
    while len(点生成队列) > 0:
        x = 点生成队列.pop(0)  # 第一个点
        for p in x[2]:
            y = [p, x[0], 点_子点D[p] if p in 点_子点D else [], x[3] + 1]
            点生成队列.append(y)
            tree.append(y)
    # 重新按层次遍历编号
    on_nn_D = {p[0]: i for i, p in enumerate(tree)}  # {旧节点编号:新节点编号,..}
    on_nn_D[None] = None
    RG.node_alias_D = {}
    for p in tree:
        i = on_nn_D[p[0]]  # 新节点编号
        RG.node_alias_D[i] = node_alias_D[p[0]]
        p[0] = i
        p[1] = on_nn_D[p[1]]
        p[2] = [on_nn_D[j] for j in p[2]]
    RG.tree.append(tree)
    RG.树_计算子树高度与节点数()

    print('类别构建...')
    RG.节点类别分配(n=class_num, 分配方式='branch', 允许不同层次=True, 最大σμ比=0.2, 输出提示=True)

    print('生成数据...')
    dataHelper = DataHelper(RG)
    return RG, dataHelper


if __name__ == '__main__':
    nouns = 获取wordnet所有名词()
    for describe in ['mammal', 'animal', 'living_thing', 'entity']:
        print('-----describe:', describe)
        RG, dataHelper = 生成树与数据集(nouns, sub_root=describe)
        RG.保存到文件('aj_')
        dataHelper.自动评估绘图(titleFrontText=f'{describe}=', saveName=f'aj_{describe}.eps')
        dataHelper.保存数据(f'aj_{describe}-dh.pkl')  # 将保留 short_path_matrix
        print()
