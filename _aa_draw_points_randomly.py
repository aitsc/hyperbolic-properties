import matplotlib.pyplot as plt
import numpy as np


class 随机采样点:
    def __init__(self, 圆形=True, 数量=3000, 半径=4):
        所有生成点 = []  # [(x,y),..]
        while len(所有生成点) < 数量:
            x = (np.random.rand() - 0.5) * 2 * 半径
            y = (np.random.rand() - 0.5) * 2 * 半径
            if 圆形 and x ** 2 + y ** 2 > 半径 ** 2:  # 圆形
                continue
            else:
                所有生成点.append((x, y))
        self.所有生成点 = 所有生成点


class 随机树嵌入:
    def __init__(self, 圆形=True, 半径=4, 层级半径=1, 最小边距=0.5, 最大分枝数=3, 最小分支数=1, 最多点数量=300, 根点数=3):
        三元组 = []  # [[(x1, y1), (x2, y2), 父节点层级],..]
        所有生成点 = [[(0, 0), 1]]  # [[节点, 第几层],..]
        点生成队列 = [[(0, 0), (0, 0), 1]]  # [[节点, 父节点, 第几层],..]

        while 点生成队列 and len(所有生成点) < 最多点数量:
            参考点, 祖点, 层数 = 点生成队列.pop(0)
            # 靠近边界的点将不再生成子节点
            if abs(参考点[0]) > 半径 - 最小边距 or abs(参考点[1]) > 半径 - 最小边距:  # 方形
                continue
            if 圆形 and 参考点[0] ** 2 + 参考点[1] ** 2 > (半径 - 最小边距) ** 2:  # 圆形
                continue
            # 生成子节点
            if 层数 <= 1:
                分支数 = 根点数
            else:
                分支数 = np.random.randint(最小分支数, 最大分枝数 + 1)
            分支数 = min(最多点数量 - len(所有生成点), 分支数)  # 生成分支不能导致超过 最多点数量
            分支点_L = []
            while len(分支点_L) < 分支数:
                # 方形随机点
                x = (np.random.rand() - 0.5) * 2 * 层级半径
                y = (np.random.rand() - 0.5) * 2 * 层级半径
                if x ** 2 + y ** 2 > 层级半径 ** 2:  # 圆形约束
                    continue
                x += 参考点[0]
                y += 参考点[1]
                if abs(x) > 半径 or abs(y) > 半径:  # 不能超过方形边界
                    continue
                if 圆形 and x ** 2 + y ** 2 > 半径 ** 2:  # 不能超过圆形边界
                    continue
                if x ** 2 + y ** 2 <= 参考点[0] ** 2 + 参考点[1] ** 2:  # 不能比参考点离原点更近
                    continue
                if (x - 祖点[0]) ** 2 + (y - 祖点[1]) ** 2 <= (参考点[0] - 祖点[0]) ** 2 + (
                        参考点[1] - 祖点[1]) ** 2:  # 不能比参考点离祖父节点更近
                    continue
                分支点_L.append((x, y))
            点生成队列 += [[i, 参考点, 层数 + 1] for i in 分支点_L]
            所有生成点 += [[i, 层数 + 1] for i in 分支点_L]
            # s = np.sqrt(参考点[0] ** 2 + 参考点[1] ** 2)
            for (x, y) in 分支点_L:
                三元组.append([参考点, (x, y), 层数])
        self.三元组 = 三元组
        self.所有生成点 = 所有生成点


exp_L = [
    (lambda norm, c=1: 1, 'Euclidean'),
    (lambda norm, c=1: np.tanh(c ** 0.5 * norm) / norm / c ** 0.5, 'Poincaré'),
    (lambda norm, c=1: np.sinh(norm * c ** 0.5) / norm / c ** 0.5, 'Hyperboloid'),
]
形状L = [True, False]
# 曲率L = [1, 0.5]
曲率L = [1]
plt.rcParams['figure.figsize'] = (len(exp_L) * 10, len(形状L) * len(曲率L) * 10)

for j, 圆形 in enumerate(形状L):  # 圆形还是方形
    随机生成点 = 随机采样点(圆形=圆形).所有生成点
    随机树嵌入_obj = 随机树嵌入(圆形=圆形)
    三元组 = 随机树嵌入_obj.三元组
    xy_L = 随机树嵌入_obj.所有生成点
    print('圆形:', 圆形, ' 三元组数量:', len(三元组), ' 节点数量:', len(xy_L))
    for k, c in enumerate(曲率L):  # 曲率
        for i, w_obj in enumerate(exp_L):  # 流形
            第几个图 = i + 1 + k * len(exp_L) + j * len(exp_L) * len(曲率L)
            plt.subplot(len(形状L) * len(曲率L), len(exp_L), 第几个图)
            if 第几个图 <= len(exp_L):
                plt.title(w_obj[1], fontsize=32)
            # plt.xlabel('$x_1$')
            # plt.ylabel('$x_2$')
            plt.axis('equal')

            xValue = []
            yValue = []
            for x, y in 随机生成点:
                norm = np.sqrt(x ** 2 + y ** 2)
                w = w_obj[0](norm, c)
                xValue.append(x * w)
                yValue.append(y * w)
            plt.scatter(xValue, yValue, s=3, c="#53B33D", marker='o')

            for (x, y), 层数 in xy_L:
                if x == y == 0:
                    w = 1
                else:
                    norm = np.sqrt(x ** 2 + y ** 2)
                    w = w_obj[0](norm, c)
                plt.scatter(x * w, y * w, color='r', s=(1 / 层数) * 60, zorder=10)

            for xy1, xy2, 层数 in 三元组:
                if xy1[0] == xy1[1] == 0:
                    w1 = 1
                else:
                    norm = np.sqrt(xy1[0] ** 2 + xy1[1] ** 2)
                    w1 = w_obj[0](norm, c)
                if xy2[0] == xy2[1] == 0:
                    w2 = 1
                else:
                    norm = np.sqrt(xy2[0] ** 2 + xy2[1] ** 2)
                    w2 = w_obj[0](norm, c)
                plt.plot((xy1[0] * w1, xy2[0] * w2), (xy1[1] * w1, xy2[1] * w2), linewidth=1 / 层数 * 4, zorder=1,
                         color='b')

# plt.xticks([])  #去掉横坐标值
# plt.yticks([])  #去掉纵坐标值
# plt.axis('off')  # 关闭坐标
plt.tight_layout()
plt.savefig('aa_show.pdf', format='pdf')
plt.show()
