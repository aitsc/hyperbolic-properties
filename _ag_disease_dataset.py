from _ac_data_helper import *
import scipy.sparse as sp


def csvDraw(path, draw=False):
    点编号S = set()
    边三元组L = []
    with open(path, 'r') as r:
        for line in r:
            line = line.strip()
            l, r = line.split(',')
            l, r = int(l), int(r)
            边三元组L.append((l, r, 1))
            点编号S.add(l)
            点编号S.add(r)
    点编号L = sorted(点编号S)

    g = nx.Graph()
    g.add_nodes_from(点编号L)
    g.add_weighted_edges_from(边三元组L)
    是否连通 = nx.is_connected(g)
    print('是否连通:', 是否连通)
    print('边数量:', len(边三元组L))
    print('点数量:', len(点编号S))
    if draw:
        pos = nx.spring_layout(g)
        nx.draw(g, pos, node_size=2, linewidths=1)
        plt.show()
        plt.close()
    return 点编号L, 边三元组L, 是否连通


def 生成树与数据集(edges_path, feats_path, labels_path):
    print('读取文件...')
    点编号L, 边三元组L, 是否连通 = csvDraw(edges_path)
    assert 是否连通, '非连通图!'
    assert len(点编号L) == max(点编号L) + 1, '点编号不是按顺序编码!'
    feats = sp.load_npz(feats_path)
    labels = np.load(labels_path)

    点_子点D = {}  # {点编号:[子点编号,..],..}
    点编号S = set(点编号L)
    for i, j, _ in 边三元组L:
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
    RG.describe = '疾病树'
    点生成队列 = [[根节点, None, 点_子点D[根节点], 1]]
    tree = [点生成队列[0]]  # [[节点编号,父,[子,..],层数],..]
    while len(点生成队列) > 0:
        x = 点生成队列.pop(0)  # 第一个点
        for p in x[2]:
            y = [p, x[0], 点_子点D[p] if p in 点_子点D else [], x[3] + 1]
            点生成队列.append(y)
            tree.append(y)
    RG.tree.append(tree)
    RG.树_计算子树高度与节点数()

    print('类别构建...')
    all_class = set()
    node_class_D = {}  # {节点:类别,..}
    for i, v in enumerate(labels):
        c = int(v)
        all_class.add(c)
        node_class_D[i] = c  # 节点从0开始编号不间断
    RG.node_class_D = node_class_D

    print('生成数据...')
    dataHelper = DataHelper(RG, feats=feats)
    return RG, dataHelper


if __name__ == '__main__':
    for describe in ['disease_nc', 'disease_lp']:
        print('-----describe:', describe)
        path = f'/Users/tanshicheng/git/paper/hgcn/data/{describe}/{describe}.'
        RG, dataHelper = 生成树与数据集(
            edges_path=path + 'edges.csv',
            feats_path=path + 'feats.npz',
            labels_path=path + 'labels.npy',
        )
        RG.describe = describe
        RG.保存到文件('ag_')
        dataHelper.自动评估绘图(titleFrontText=f'{describe}=', saveName=f'ag_{describe}.eps')
        dataHelper.保存数据(f'ag_{describe}-dh.pkl')  # 将保留 short_path_matrix
