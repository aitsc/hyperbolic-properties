import networkx as nx
import matplotlib.pyplot as plt
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import plot_model
from colorama import Fore, Back, Style
from matplotlib.text import Text
import colorsys
import random
import re
import os
import time
import sys
import traceback


def multi_source_shortest_path(nodes, edges, force_use=None):
    """
    计算多源最短路径
    :param nodes: [点1,..]
    :param edges: [(点1,点2),..] or [(点1,点2,距离),..]; 必须无向图, 加速计算树多源最短路径距离一律为1
    :param force_use: None or int; 是否强制使用, 1=nx弗洛伊德算法, 2=tf弗洛伊德算法, 3=加速树算法
        None或者其他就默认, 有gpu使用tf, 有树使用加速, 最后考虑nx
    :return:
    """
    assert len(nodes) == len(set(nodes)), '点不能重复!'
    # 边改为距离三元组
    if len(edges[0]) == 2:
        edges = [(i, j, 1.) for i, j in edges]
    # 去除重复边/有向边/自环
    edges = sorted(set((*sorted((i, j)), k) for i, j, k in edges if i != j))
    assert len(set((i, j) for i, j, k in edges)) == len(edges), '不是无向图!'
    # 使用 tensorflow 版弗洛伊德算法
    if force_use == 2 or (tf.test.is_gpu_available() and force_use not in {1, 3}):
        node_i_D = {node: i for i, node in enumerate(nodes)}  # {节点编号:序号,..}
        mat = np.ones([len(nodes), len(nodes)]) * float("inf")  # 节点距离矩阵
        for i in range(len(nodes)):
            mat[i][i] = 0.  # 自己距离是0
        for i, j, k in edges:
            i = node_i_D[i]
            j = node_i_D[j]
            mat[i][j] = mat[j][i] = k
        floyd_warshall_tf = lambda m: tf.while_loop(
            lambda m, n, i: n > i,
            lambda m, n, i: (tf.reduce_min([m, m[i: i + 1, :] + m[:, i: i + 1]], axis=0), n, i + 1),
            [m, m.shape[0], 0] if isinstance(m, tf.Tensor) else [tf.constant(m), len(m), 0]
        )[0]
        return floyd_warshall_tf(mat).numpy()
    # 构建 nx 图
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    # 如果是一棵树
    nodes_S = set(sum([list(i[:2]) for i in edges], []))  # 边上的点
    root_node = min(nodes_S)
    if force_use == 3 or (nx.is_connected(g) and len(nodes_S) == len(edges) + 1 and force_use not in {1, 2}):
        # 获得 节点对应子节点, 根节点
        p_sp_D = {}  # {点编号:[子点编号,..],..}, 点编号 不含叶子节点
        for i, j, _ in edges:
            nodes_S.discard(j)
            if i in p_sp_D:
                p_sp_D[i].append(j)
            else:
                p_sp_D[i] = [j]
        if len(nodes_S) != 1:
            print('multi_source_shortest_path: 无法直接获取根节点, 该树可能存在大编号节点是小编号节点的父节点, 导致边排序后破坏了上下位关系')
            print('\t设置根节点为最小编号:', root_node)
        else:
            root_node = nodes_S.pop()
        # 获得每个节点的路径集合
        p_path_D = {root_node: {root_node}}  # {点编号:{路径上点1,..},..}
        node_stack = [root_node]  # 根节点
        while len(node_stack) != 0:
            x = node_stack.pop(0)  # 第一个点
            if x in p_sp_D:  # 非叶子节点向下扫描
                for y in p_sp_D[x]:
                    node_stack.append(y)
                    p_path_D[y] = p_path_D[x] | {y}
        # 距离计算
        short_path_matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for i in range((len(nodes) - 1)):
            x = nodes[i]  # 节点编号
            try:
                xs = p_path_D[x]  # 节点路径 set
            except:
                xs = None
            for j in range(i + 1, len(nodes)):
                y = nodes[j]
                try:
                    if xs:
                        ys = p_path_D[y]
                        short_path_matrix[i, j] = len(xs) + len(ys) - 2 * len(xs & ys)
                except:
                    short_path_matrix[i, j] = float('inf')
        short_path_matrix += short_path_matrix.T
    else:
        short_path_matrix = nx.floyd_warshall_numpy(g, nodelist=nodes).astype(np.float32)
    return short_path_matrix


def ncolors(n=1, rand=False):
    '''
    生成区分度比较大的几个颜色
    :param n: 生成几个颜色
    :param rand: 是否随机
    :return:
    '''
    if n < 1:
        return []
    if n == 1:
        return ['#1E78B3']
    rgb_colors = []
    # get_n_hls_colors
    hls_colors = []
    i = 0
    step = 360.0 / n
    while i < 360:
        h = i
        if rand:
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
        else:
            s = 95
            l = 55
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    # hlsc to rgb
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        rgb = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb = [('0' + hex(i)[2:])[-2:] for i in rgb]
        rgb_colors.append('#' + ''.join(rgb))
    return rgb_colors


def create_gif(image_list, gif_name, duration=0.5):
    '''
    用于生成gif图片, 用eps矢量图生成gif会模糊
    :param image_list: [路径1,..]
    :param gif_name: str, gif图片存储位置
    :param duration: 每帧间隔
    :return: None
    '''
    frames = []
    for image_name in image_list:
        try:
            frames.append(imageio.imread(image_name))
        except ValueError:  # 其他格式图片不行, 比如pdf
            return False
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return True


def get_stat(tensor, text='', get_out=0):
    '''
    计算张量的统计信息
    :param tensor: np.array or list or tf.IndexedSlices or tf.Tensor, 待统计张量
    :param text: 描述
    :param get_out: 0=返回统计结果, 1=t就是统计结果并返回描述
    :return:
    '''
    inform = [text]
    if tensor is None:
        return inform
    if get_out != 1:
        if isinstance(tensor, tf.IndexedSlices):
            t = tensor.values
        else:
            t = tensor
        try:
            t = t.numpy()
        except:
            t = np.array(t)
        norm2 = np.sum(t ** 2, -1)
        inform.append(t.max())
        inform.append(t.min())
        inform.append(np.abs(t).min())
        inform.append(t.mean())
        inform.append(t.std())
        inform.append(norm2.max())
        inform.append(norm2.min())
        all_num = np.prod(t.shape)  # 所有值数量
        x = np.count_nonzero(np.isinf(t))  # 无穷值的数量
        inform += [x, x * 100 / all_num]
        x = np.count_nonzero(np.isnan(t))  # 无效值的数量
        inform += [x, x * 100 / all_num]
        x = all_num - np.count_nonzero(t)  # 零值的数量
        inform += [x, x * 100 / all_num]
        x = np.count_nonzero(t < 0)  # 负值的数量
        inform += [x, x * 100 / all_num]
        inform.append(str(t.shape))  # 形状
    if get_out == 0:  # 统计
        return inform
    if get_out == 1:  # 当作统计好的只输出
        inform = tensor
    if len(inform) > 1:
        o = ': max(%e), min(%e), min_abs(%e), mean(%e), std(%e), norm2_max(%e), norm2_min(%e), inf_num(%d-%.2f%%), ' \
            'nan_num(%d-%.2f%%), zero_num(%d-%.2f%%), neg_num(%d-%.2f%%), shape%s' % (
                *inform[1:],)
    else:
        o = ': None'
    o = inform[0] + Fore.GREEN + o + Style.RESET_ALL
    return o


def get_out_grad(model, data, y_index: int, outScreen=True):
    '''
    :param model: tf模型
    :param data: np.array, 训练数据
    :param y_index: int, model.outputs[y_index] 为被求导因变量
    :param outScreen: 是否直接print输出
    :return:
    '''
    # 每层的输出张量/描述
    l_tensor, l_name = get_layers_tensors(model, clip=True)
    # 参数的输出/名字
    w_out = model.trainable_weights
    if not isinstance(w_out, list):
        w_out = [w_out]
    w_name = [i.name for i in w_out]
    # 被求导因变量
    y_tensor = model.outputs[y_index] if y_index is not None else model.outputs[0]
    # 构建新模型
    functors = Model(model.input, l_tensor + [y_tensor])
    # 计算每层输出
    with tf.GradientTape() as tape:
        out = functors(data)
        l_out = out[:-1]
        y_out = out[-1]
    # 计算梯度
    output_grad = tape.gradient(y_out, w_out + l_out)

    stat = []
    for i, out in enumerate(w_out + l_out):
        if i < len(w_out):
            text = '%d-weight_outs (%s)' % (i + 1, w_name[i])
        else:
            text = '%d-layer_outs (%s)' % (i - len(w_out) + 1, l_name[i - len(w_out)])
        stat.append(get_stat(out, text))
    for i, grad in enumerate(output_grad):
        if i < len(w_out):
            text = '%d-weight_grad (%s)' % (i + 1, w_name[i])
        else:
            text = '%d-layer_grad (%s)' % (i - len(w_out) + 1, l_name[i - len(w_out)])
        stat.append(get_stat(grad, text))
    if outScreen:
        for i, j in enumerate(stat):
            if i > 0 and j[0][0:2] == '1-':
                print('-' * 20)
            print(get_stat(j, get_out=1))
    return stat


def printModelG(model, path):
    '''
    绘制模型图并输出图片
    :param model: tensorflow.keras.Model
    :param path: 输出文件路径
    :return:
    '''
    model._layers = [layer for layer in model._layers if isinstance(layer, Layer)]
    try:
        plot_model(model, to_file=path, show_shapes=True, show_layer_names=True, expand_nested=True)
    except ImportError:
        print(f"模型输出失败: {path}")
        print(traceback.format_exc())


def get_layers_tensors(model, clip=False):
    '''
    获得模型每一层的多个输出张量, 以及层/输入张量/输入张量的名字. 处理不了一个操作返回多张量, 比如 TopKV2 = tf.nn.top_k([10,2,3], 1)
    :param model: tensorflow.keras.Model
    :param clip: bool, 是否 修剪张量名称的编号等信息, 比如 /xxxxx:0 会被去掉
    :return: 每层张量 list, 对应名称描述 list
    '''
    l_outputs = []
    l_names = []
    l_inputs_names = []
    if clip:
        clip_tn = lambda x: re.sub('/[^:]+?:[0-9]+|:[0-9]+', '', x)  # 修剪张量名称的编号等信息
    else:
        clip_tn = lambda x: x
    for l in model.layers:
        l_name = l.name
        l_tensors = []
        for t in l.inbound_nodes:  # 同一层可能被多次使用
            if isinstance(t.input_tensors, list):
                l_inputs_n = ','.join([clip_tn(i.name) for i in t.input_tensors])
            else:
                l_inputs_n = clip_tn(t.input_tensors.name)
            t = t.output_tensors
            if isinstance(t, list):
                l_tensors += [i for i in t]
                l_inputs_names += [l_inputs_n] * len(t)
            else:
                l_tensors.append(t)
                l_inputs_names += [l_inputs_n]
        l_outputs += l_tensors
        l_names += [l_name] * len(l_tensors)  # 不同张量同一个层名字
    l_outputs_names = [clip_tn(i.name) for i in l_outputs]
    all_names = ['%s::%s → %s' % (i, j, k) for i, j, k in zip(l_names, l_inputs_names, l_outputs_names)]
    return l_outputs, all_names


def draw_line_chart(x, xaxis: str = None, xlim: list = None, title: str = None, ylabel_display=True,
                    y_left: list = None, yaxis_left: str = None, ylim_left: list = None, ylabel_left: list = None,
                    y_right: list = None, yaxis_right: str = None, ylim_right: list = None, ylabel_right: list = None,
                    y_left_ls='-', y_right_ls='--',
                    grid_ls_x='', grid_axis_yl=False, grid_axis_yr=False, grid_alpha=0.4,
                    annotate=True, annotate_in_left=True, annotate_color=True, legend_right=True,
                    xl_text_margin=0.1, xl_arrow_len=0.3, xr_text_margin=0.05, xr_arrow_len=0.3, custom_dpi=90,
                    length=10, width=10, save_path: str = None, lw=1, ytext_min_gap=0.25, markersize=4):
    '''
    绘制折线图, 可以标记/双y轴, 所有折线共用x轴.
    代码修改建议:
        - 点的形状/中心颜色可以修改
        - annotate 箭头默认颜色可修改
        - 空出图例的位置 可能需要修改 标签的长度/label_len
    :param x: [[..],..]; x轴坐标, 二维列表, 必选
    :param xaxis: str or None; x轴的名字
    :param xlim: [left,right] or None; x轴的左右限制
    :param title:  str or None; 标题

    :param y_left: [[..],..] or None; 左y轴坐标, 二维列表, 必选
    :param yaxis_left: str or None; 左y轴的名字
    :param ylim_left: [left,right] or None; 左y轴的左右限制. left,right 可为None自动限制y最大值. 设置太小会自动放大
    :param ylabel_left: [str,..] or None; 左y轴每条折线的标签
    :param y_right: [[..],..] or None; 右y轴坐标, 二维列表
    :param yaxis_right: str or None; 右y轴的名字
    :param ylim_right: [left,right] or None; 右y轴的左右限制. left,right 可为None自动限制y最大值. 设置太小会自动放大
    :param ylabel_right: [str,..] or None; 右y轴每条折线的标签
    :param y_left_ls: str; 左y轴/左标记箭头/左y轴网格 的折线风格 ['-','--','-.',':']
    :param y_right_ls: str; 右y轴/右标记箭头/右y轴网格 的折线风格
    :param ylabel_display: bool, 是否显示每条折线的标签(图例,legend)

    :param grid_ls_x: str or None; 网格x轴的折线风格, 空则不绘制x轴网格
    :param grid_axis_yl: bool; 是否绘制左y轴的网格
    :param grid_axis_yr: bool; 是否绘制右y轴的网格
    :param grid_alpha: float; 网格的不透明度

    :param annotate: bool; 是否标记. 标记则 xlim 无效
    :param annotate_in_left: bool; 左y轴是否标记在左侧, 否则右y轴标记在左侧
    :param annotate_color: bool; 标记是否使用折线标签的颜色
    :param legend_right: bool; 是否在右侧自动空出图例的位置, 需要 annotate,ylabel_display=True. 可通过调 xr_text_margin=1 替代

    和字体大小有关, 字体大以下值可能也要大:
    :param xl_text_margin: float; 左侧标记离左y轴的距离, 单位基本与 length 相同
    :param xl_arrow_len: float; 左标记箭头长度, 单位基本与 length 相同
    :param xr_text_margin: float; 右侧标记离右y轴的距离, 单位基本与 length 相同, 小心和 legend_right 重复拉大边距
    :param xr_arrow_len: float; 右标记箭头长度, 基本与 length 相同
    :param custom_dpi: float; 基本与 length 相同的单位, 稍稀疏一些. 如果标记文本过长则可以适当减小这个数值
    :param ytext_min_gap: float; 上下标记之间的最小中心(非边缘)距离, 单位基本与 length 相同

    :param length: float; 图片的长度, 单位一般是100像素
    :param width: float; 图片的宽度/高度, 单位一般是100像素
    :param save_path: str or None; 图片保存路径
    :param lw: float; 折线的线条宽度
    :param markersize: float; 点大小
    :return:
    '''
    assert (y_left is not None) and (len(y_left) > 0), 'y_left 不能为空!'
    assert len(x) == len(y_left[0]), 'x 与 y_left[0] 长度不相等! %d!=%d' % (len(x), len(y_left[0]))
    assert len(y_left[0]) == sum([len(i) for i in y_left]) / len(y_left), 'y_left 每行长度不相等!'
    assert not isinstance(y_left[0][0], list), 'y_left 必须是二维矩阵! ' + str(y_left)
    # 允许的点的形状, 按顺序取的, 可修改
    markers = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '']
    mfc = ['None'] * len(markers)  # 点的中间颜色, 数量与markers相等
    # 全图配置
    plt.rcParams['figure.figsize'] = (length, width)
    fig = plt.figure(1)
    text_obj = Text()
    get_text_w = lambda t: get_text_width(t, fig, text_obj)  # 获取字符像素宽度
    ax_left = fig.add_subplot(111)
    all_polt = []  # 所有图, 用于绘制图例
    # 左右坐标 配置信息
    colors = ncolors((len(y_left) if y_left else 0) + (len(y_right) if y_right else 0))  # 颜色
    colors_left = colors[:len(y_left) if y_left else 0]  # 左 颜色
    colors_right = colors[len(colors_left):]  # 右 颜色
    assert len(colors) <= len(markers), '线条数量(%d)不能超过点的形状数量(%d)!' % (len(colors), len(markers))
    markers_left = markers[:len(colors_left)]  # 左 点形状
    mfc_left = mfc[:len(colors_left)]
    markers_right = markers[len(colors_left): len(colors)]  # 右 点形状
    mfc_right = mfc[len(colors_left): len(colors)]
    # y_left 绘制
    if ylabel_left is not None and len(ylabel_left) > 0:  # 处理标签
        assert len(ylabel_left) == len(y_left), 'ylabel_left 和 y_left 长度不相等! %d!=%d' % (len(ylabel_left), len(y_left))
    else:
        ylabel_left = None
    for i, y in enumerate(y_left):  # 开始绘制
        if ylabel_left:
            label = ylabel_left[i]
        else:
            label = None
        lns = ax_left.plot(x, y, c=colors_left[i], label=label, markersize=markersize,
                           marker=markers_left[i], mfc=mfc_left[i], lw=lw, ls=y_left_ls)
        if label:  # 如果有标签
            all_polt += lns
    if yaxis_left:  # 坐标轴名称
        ax_left.set_ylabel(yaxis_left)
    if ylim_left:  # 坐标轴约束
        assert len(ylim_left) == 2, 'ylim_left 数量错误! %s' % str(ylim_left)
        yy = sum(y_left, [])
        y_min, y_max = min(yy), max(yy)
        if ylim_left[0] is None or ylim_left[0] > y_min:
            ylim_left[0] = y_min
        if ylim_left[1] is None or ylim_left[1] < y_max:
            ylim_left[1] = y_max
        assert ylim_left[0] < ylim_left[1], 'ylim_left 大小错误! %s' % str(ylim_left)
        ax_left.set_ylim(*ylim_left)
    # y_right 绘制
    ax_right = None
    if y_right is not None and len(y_right) > 0:
        assert len(x) == len(y_right[0]), 'x 与 y_right[0] 长度不相等! %d!=%d' % (len(x), len(y_right[0]))
        assert len(y_right[0]) == sum([len(i) for i in y_right]) / len(y_right), 'y_right 每行长度不相等!'
        assert not isinstance(y_right[0][0], list), 'y_right 必须是二维矩阵! ' + str(y_right)
        if ylabel_right is not None and len(ylabel_right) > 0:  # 处理标签
            assert len(ylabel_right) == len(y_right), \
                'ylabel_right 和 y_right 长度不相等! %d!=%d' % (len(ylabel_right), len(y_right))
        else:
            ylabel_right = None
        ax_right = ax_left.twinx()
        for i, y in enumerate(y_right):  # 开始绘制
            if ylabel_right:
                label = ylabel_right[i]
            else:
                label = None
            lns = ax_right.plot(x, y, c=colors_right[i], label=label, markersize=markersize,
                                marker=markers_right[i], mfc=mfc_right[i], lw=lw, ls=y_right_ls)
            if label:  # 如果有标签
                all_polt += lns
        if yaxis_right:  # 坐标轴名称
            ax_right.set_ylabel(yaxis_right)
        if ylim_right:  # 坐标轴约束
            assert len(ylim_right) == 2, 'ylim_right 数量错误! %s' % str(ylim_right)
            yy = sum(y_right, [])
            y_min, y_max = min(yy), max(yy)
            if ylim_right[0] is None or ylim_right[0] > y_min:
                ylim_right[0] = y_min
            if ylim_right[1] is None or ylim_right[1] < y_max:
                ylim_right[1] = y_max
            assert ylim_right[0] < ylim_right[1], 'ylim_right 大小错误! %s' % str(ylim_right)
            ax_right.set_ylim(*ylim_right)
    # 标记
    if annotate:
        if annotate_in_left:
            yl_annotate_left = True  # y_left 是否标记在左边
            yl_annotate_right = False  # y_left 是否标记在右边
            yr_annotate_left = False  # y_right 是否标记在左边
            yr_annotate_right = True  # y_right 是否标记在右边
        else:  # 不可能两边都有同一条线的标记
            yl_annotate_left = False
            yl_annotate_right = True
            yr_annotate_left = True
            yr_annotate_right = False
    else:
        yl_annotate_left = yl_annotate_right = yr_annotate_left = yr_annotate_right = False
    if yl_annotate_left or yl_annotate_right:
        assert r'\n' not in str(ylabel_left), '标记时 ylabel_left 不允许存在换行符号!'  # 防止字体宽度和间隔判断错误
    if yr_annotate_left or yr_annotate_right:
        assert r'\n' not in str(ylabel_right), '标记时 ylabel_right 不允许存在换行符号!'
    # 计算左右标记和双y轴的关系
    if (yl_annotate_left and ylabel_left) or (yr_annotate_left and y_right and ylabel_right):
        annotate_left = True  # 左侧标记
    else:
        annotate_left = False
    if (yl_annotate_right and ylabel_left) or (yr_annotate_right and y_right and ylabel_right):
        annotate_right = True  # 右侧标记
    else:
        annotate_right = False
    x_min, x_max = min(x), max(x)
    # 计算y轴标记目标点位置/颜色
    name_y_left = []  # [[标记名称,目标点y轴坐标,标记文本位置,颜色],..], 左侧标记
    name_y_right = []  # [[标记名称,目标点y轴坐标,标记文本位置,颜色],..], 右侧标记
    i_min = x.index(x_min)  # x轴最左边
    i_max = x.index(x_max)  # x轴最右边
    ltext_min_gap = rtext_min_gap = ltext_top = ltext_bottom = rtext_top = rtext_bottom = 0  # 左右文本的最小间隙和ylim, 单位与坐标轴相同
    ax_annotate_left, ax_annotate_right = None, None  # 左右标记对应的坐标轴
    if annotate_left:  # 如果左侧有标记
        if yl_annotate_left:
            y, ylabel, yc, ylim = y_left, ylabel_left, colors_left, ylim_left
            ax_annotate_left = ax_left
        else:
            y, ylabel, yc, ylim = y_right, ylabel_right, colors_right, ylim_right
            ax_annotate_left = ax_right
        # 计算y轴约束
        yy = sum(y, [])
        ltext_bottom = min(yy + ([ylim[0]] if ylim else []))
        ltext_top = max(yy + ([ylim[1]] if ylim else []))
        if ylim:
            ltext_min_gap = ytext_min_gap / (width / (ylim[1] - ylim[0]))
        else:
            ltext_min_gap = ytext_min_gap / (width / (ltext_top - ltext_bottom))
        # 开始赋值
        for v, name, c in zip(y, ylabel, yc):
            v = v[i_min]
            name_y_left.append([name, v, v, c])
    if annotate_right:  # 如果右侧有标记
        if yl_annotate_right:
            ax_annotate_right = ax_left
            y, ylabel, yc, ylim = y_left, ylabel_left, colors_left, ylim_left
        else:
            ax_annotate_right = ax_right
            y, ylabel, yc, ylim = y_right, ylabel_right, colors_right, ylim_right
        # 计算y轴约束
        yy = sum(y, [])
        rtext_bottom = min(yy + ([ylim[0]] if ylim else []))
        rtext_top = max(yy + ([ylim[1]] if ylim else []))
        if ylim:
            rtext_min_gap = ytext_min_gap / (width / (ylim[1] - ylim[0]))
        else:
            rtext_min_gap = ytext_min_gap / (width / (rtext_top - rtext_bottom))
        # 开始赋值
        for v, name, c in zip(y, ylabel, yc):
            v = v[i_max]
            name_y_right.append([name, v, v, c])
    # 计算y轴标记文本位置
    all = []
    if name_y_left:
        all.append([name_y_left, ltext_top, ltext_bottom, ltext_min_gap])
    if name_y_right:
        all.append([name_y_right, rtext_top, rtext_bottom, rtext_min_gap])
    for name_y_, text_top, text_bottom, text_min_gap in all:
        # 正序排序, (目标点y轴坐标,标记名称)
        name_y_ = sorted(name_y_, key=lambda t: (t[1], t[0]))
        # 均匀分割
        gap = (text_top - text_bottom) / (len(name_y_) + 1)
        p_L = [text_bottom + i * gap for i in range(1, len(name_y_) + 1)]
        for i, p in enumerate(p_L):
            name_y_[i][2] = p
        # 从下到上移动规范文本位置, 从上到下移动规范文本位置, 该算法可能有 直角三角形/梯形形式 聚集
        for range_ in [range(len(name_y_)), range(len(name_y_) - 1, -1, -1)]:
            for i in range_:
                y = name_y_[i]
                bottom = text_bottom + text_min_gap  # 不能比边界低
                if i > 0:
                    bottom = max(bottom, name_y_[i - 1][2] + text_min_gap)  # 不能比下一个点低
                top = text_top - text_min_gap  # 不能比边界高
                if i < len(name_y_) - 1:
                    top = min(top, name_y_[i + 1][2] - text_min_gap)  # 不能比上一个点高
                # 因为均匀分割, 故不可能发生. 受精度影响, 比如超15位后 1.9516675497853828<1.9516675497853830
                assert top / (abs(int(top)) + 1) >= bottom / (abs(int(top)) + 1) - 10 ** -14, \
                    'top < bottom ! %.19f<%.19f, %s' % (top, bottom, str(range_))
                yp = y[1]  # 目标点y轴坐标
                if bottom <= yp <= top:
                    y[2] = yp
                elif bottom > yp:
                    y[2] = bottom
                else:
                    y[2] = top
    # 计算标记宽度
    ml, mr = xl_text_margin, xr_text_margin  # 左右标记文本离y轴的距离, 单位基本与 length 相同
    text_len_max = -1  # 总体最大字体宽度
    if annotate_left:
        xl_text_len = max([get_text_w(i[0]) for i in name_y_left]) / custom_dpi  # 文本最大长度, 单位基本与 length 相同
        tl, al = xl_text_len, xl_arrow_len
        text_len_max = max(text_len_max, xl_text_len)
    else:  # 如果左边没有标记
        tl, al = 0, 0
    if annotate_right:
        xr_text_len = max([get_text_w(i[0]) for i in name_y_right]) / custom_dpi  # 文本最大长度, 单位基本与 length 相同
        tr, ar = xr_text_len, xr_arrow_len
        text_len_max = max(text_len_max, xr_text_len)
    else:  # 如果右边没有标记
        tr, ar = 0, 0
    if text_len_max > 0 and legend_right and ylabel_display:  # 空出图例
        label_len = 0.6  # 标签的长度, 单位基本与 length 相同
        mr += text_len_max + label_len
    # 计算x轴标记位置 和 x轴范围
    if annotate_left or annotate_right:
        ''' 根据 4元1次9参数方程组 解析解 计算
        from sympy import *
        leng, ll, lr, xl, xr, ml, tl, al, mr, tr, ar, yl, yr = symbols(
            'length xlim_left xlim_right x_min x_max ml tl al mr tr ar xtext_left xtext_right'
        )
        for i, j in solve([
            (ml + tl + al) * (lr - ll) / leng - (xl - ll),
            (mr + tr + ar) * (lr - ll) / leng - (lr - xr),
        ], [ll, lr]).items():
            print(i, '=', j)
        for i, j in solve([
            xl - (tl + al) * (lr - ll) / leng - yl,
            xr + ar * (lr - ll) / leng - yr,
        ], [yl, yr]).items():
            print(i, '=', j)
        '''
        xlim_left = (al * x_max + ar * x_min - length * x_min + ml * x_max + mr * x_min + tl * x_max + tr * x_min) / (
                al + ar - length + ml + mr + tl + tr)
        xlim_right = (al * x_max + ar * x_min - length * x_max + ml * x_max + mr * x_min + tl * x_max + tr * x_min) / (
                al + ar - length + ml + mr + tl + tr)
        xtext_right = (-ar * xlim_left + ar * xlim_right + length * x_max) / length
        xtext_left = (al * xlim_left - al * xlim_right + length * x_min + tl * xlim_left - tl * xlim_right) / length
        xlim = [xlim_left, xlim_right]  # x轴约束
    else:
        xtext_right = xtext_left = None  # 左侧标记 和 右侧标记 的x轴坐标
    # 绘制标记
    all = []
    if name_y_left:
        all.append([name_y_left, ax_annotate_left, x_min, xtext_left, y_left_ls])
    if name_y_right:
        all.append([name_y_right, ax_annotate_right, x_max, xtext_right, y_right_ls])
    for name_y_, ax_annotate_, x_m, xtext_, y__ls in all:
        for name, ya, yt, c in name_y_:
            if not annotate_color:
                c = 'black'
            ax_annotate_.annotate(s=name, c=c, xy=(x_m, ya), xytext=(xtext_, yt),
                                  arrowprops={'arrowstyle': '->', 'linestyle': y__ls})
    # 配置
    if xaxis:  # x轴名称
        ax_left.set_xlabel(xaxis)
    if xlim:  # x轴约束
        assert len(xlim) == 2, 'xlim 数量错误! %s' % str(xlim)
        if xlim[0] is None or xlim[0] > x_min:
            xlim[0] = x_min
        if xlim[1] is None or xlim[1] < x_max:
            xlim[1] = x_max
        assert xlim[0] < xlim[1], 'xlim 大小错误! %s' % str(xlim)
        ax_left.set_xlim(*xlim)
    if title:  # 标题
        plt.title(title)
    if all_polt and ylabel_display:  # 图例
        labs = [l.get_label() for l in all_polt]
        ax_left.legend(all_polt, labs, loc='best', labelspacing=0.)
    ax_left.set_xticks(x)  # 每个刻度都显示
    plt.setp(ax_left.get_xticklabels(), rotation=90, horizontalalignment='right')
    # 网格
    if grid_axis_yl:
        ax_left.grid(linestyle=y_left_ls, alpha=grid_alpha, axis='y')
    if grid_axis_yr and ax_right is not None:
        ax_right.grid(linestyle=y_right_ls, alpha=grid_alpha, axis='y')
    if grid_ls_x:
        ax_left.grid(linestyle=grid_ls_x, alpha=grid_alpha, axis='x')
    # 保存
    if save_path:
        fname, fextension = os.path.splitext(save_path)
        plt.tight_layout()  # 去除边缘空白
        plt.savefig(save_path, format=fextension[1:])
    plt.show()
    plt.close()  # 防止图片叠加变大, 无界面的Linux


def get_text_width(text, fig_obj=None, text_obj=None):
    '''
    获取一个字符串的绘制宽度
    :param text: str, 要计算宽度的字符
    :param fig_obj: matplotlib.figure.Figure, 没有可能导致宽度获取不准确
    :param text_obj: matplotlib.text.Text, 用于输入字体信息, 没有的话每次默认可能实例化效率低
    :return: float, 像素
    '''
    if text_obj is None:
        from matplotlib.text import Text
        text_obj = Text()
    clean_line, ismath = text_obj._preprocess_math(text)
    if fig_obj is None:
        from matplotlib.backend_bases import RendererBase
        renderer = RendererBase()
    else:
        renderer = fig_obj.canvas.get_renderer()
    w = renderer.get_text_width_height_descent(s=clean_line, prop=text_obj.get_fontproperties(), ismath=ismath)[0]
    # 还有一种获取长度的方法需要绘制出来并隐藏
    # lambda t: fig.text(0.15, 0.5, t, alpha=0).get_window_extent(fig.canvas.get_renderer()).bounds[2]
    return w


def format_metrics(metrics, d=4, names=None, seg=', ', lineMaxLen=None, frontText=None):
    '''
    简写格式化指标输出, 保留一定小数
    :param metrics: float or None or list; 指标. 值/一维列表/二维列表/三维列表
        值: 必须 names:None or names:str, 值是一个指标
        一维列表: 若 names:str 则表示一个指标, 若 names:list 则每个元素一个指标
        二维列表: 必须有 names:list, 每个元素一个指标
        三维列表: 必须有 names:list, 每个元素一个指标
    :param d: int; 保留几位小数
    :param names: None or str or [名字,..]; 每组指标的名称, 允许latex
    :param seg: str or [str,..]; 每组指标之间的分隔符, [', ','\n', 等等]
    :param lineMaxLen: None or float; 一行超过该长度则强制令seg='\n', 单位matplotlib像素. 字体宽度参考:
        {'A': 6.84375, 'a': 6.125, '1': 6.359375, '-': 3.609375, '#': 8.375, '.': 3.1875, '%': 9.5, '(': 3.90625, '$A$': 7.0}
        千像素绘图的标题一般在 550 以下
    :param frontText: None or str; 指标描述的前置文本, 允许$tex$
    :return:
    '''

    def 格式化(x):
        if isinstance(x, list) and len(x) > 1:
            ret = str([round(i, d) for i in x]).replace(' ', '')
        else:
            if x is not None:
                ret = str(round(x[0] if isinstance(x, list) else x, d))
            else:
                ret = ''
        return ret

    if metrics is None:
        return ''
    if frontText is None:
        frontText = ''
    out_s = frontText
    if isinstance(metrics, float) or isinstance(metrics, int):
        if names and not isinstance(names, list):
            out_s += names + ':' + 格式化(metrics)
        else:
            out_s += 格式化(metrics)
    elif isinstance(metrics, list):
        if not isinstance(names, list):
            names = [''] * len(metrics)
        assert len(metrics) == len(names), 'len(metrics) != len(names) 错误 %d!=%d' % (len(metrics), len(names))
        # seg 扩展
        if isinstance(seg, list) and len(seg) < len(metrics) - 1:
            seg_L = sum([seg for i in range(int(len(metrics) / len(seg)))], [])
        else:
            seg_L = [seg] * len(metrics)
        # 开始计算
        for i, (m, name) in enumerate(zip(metrics, names)):
            s = name + ':' + 格式化(m)
            if i > 0:  # 需要 seg 分割
                if lineMaxLen and get_text_width(out_s.split('\n')[-1] + seg_L[i - 1] + s) > lineMaxLen:
                    out_s += '\n'
                else:
                    out_s += seg_L[i - 1]
            else:  # 防止存在较长 frontText
                if lineMaxLen and get_text_width(out_s.split('\n')[-1] + s) > lineMaxLen:
                    out_s += '\n'
            out_s += s
    return out_s


class Test:
    @staticmethod
    def draw_line_chart():
        print('-' * 10 + sys._getframe().f_code.co_name)
        for i in range(1):
            x = [i * 20 for i in range(11)][::-1]
            y = lambda n, m: [[random.uniform(4000, m) for i in x] for j in range(n)]
            y_left = y(6, m=2)
            y_right = y(1, m=40)
            draw_line_chart(
                x=x,
                xaxis='x',
                xlim=None,
                title='test',
                y_left=y_left,
                yaxis_left='$y_1$',
                ylim_left=[100, None],
                ylabel_left=['$yl_{%d}$' % i for i in range(len(y_left))],
                y_right=y_right,
                yaxis_right='$y_2$',
                ylim_right=[100, None],
                ylabel_right=['$yr_{%d}$' % i for i in range(len(y_right))],
                length=12,
                width=6,
                save_path='ae_test.eps',
                lw=1, ytext_min_gap=0.25,
                ylabel_display=True,
                y_left_ls='-', y_right_ls='--',
                grid_ls_x=':', grid_axis_yl=True, grid_axis_yr=True, grid_alpha=0.2,
                annotate=True, annotate_in_left=True, annotate_color=True, legend_right=True,
                xl_text_margin=0.1, xl_arrow_len=0.3, xr_text_margin=0.05, xr_arrow_len=0.3, custom_dpi=90,
            )
            print(i)

    @staticmethod
    def format_metrics():
        print('-' * 10 + sys._getframe().f_code.co_name)
        metrics = [[0.9999877989950267],
                   [0.6320939334637965, 0.6379647749510763, 0.538160469667319, 0.6360078277886497],
                   [0.7651663405088063, 0.726027397260274, 0.7045009784735812, 0.4461839530332681],
                   [0.6457925636007827, 0.6457925636007827, 0.6262230919765166, 0.6457925636007827],
                   [0.7930299193365005, 0.277323965495023, 0.5074972574331496, 0.8412893868079733],
                   [0.23890332539083917, 0.3353278520965868, 0.2861508363754182, 0.5947635999654112,
                    0.32326189020916407]]
        names = ['$\\mathcal{M}_{1}$', '$\\mathcal{M}_{2}$', '$\\mathcal{M}_{3}$', '$\\mathcal{M}_{4}$',
                 '$\\mathcal{M}_{5}$', '$\\mathcal{M}_{6}$']
        s = format_metrics(metrics, d=4, names=names, lineMaxLen=550, frontText='m:0, epoch:0, acc:0.7396, ')
        print(s)
        for i in s.split('\n'):
            print(get_text_width(i))

    @staticmethod
    def get_text_width():
        print('-' * 10 + sys._getframe().f_code.co_name)
        print('计算字体宽度...')
        t = ['A', 'a', '1', '-', '#', '.', '%', '(', '$A$']
        w = []
        for i in t:
            w.append(get_text_width(i))
        print({i: j for i, j in zip(t, w)})

    @staticmethod
    def create_gif():
        print('-' * 10 + sys._getframe().f_code.co_name)
        image_list = [
            # '/Users/tanshicheng/code/python/paper/hyperbolic properties/an_test.pdf',
            '/Users/tanshicheng/code/python/paper/hyperbolic properties/ad_graph/ad_m0_0-trees.eps',
            '/Users/tanshicheng/code/python/paper/hyperbolic properties/ad_graph/ad_m0_0.eps',
        ]
        gif_name = 'ae_test.gif'
        create_gif(image_list, gif_name, duration=0.5)

    @staticmethod
    def multi_source_shortest_path():
        print('-' * 10 + sys._getframe().f_code.co_name)
        print('-' * 5, '值测试')
        inf = float("inf")
        mat = [
            [0., 12., inf, inf, inf, 16, 14],
            [12., 0, 10, inf, inf, 7, inf],
            [inf, 10, 0, 3, 5, 6, inf],
            [inf, inf, 3, 0, 4, inf, inf],
            [inf, inf, 5, 4, 0, 2, 8],
            [16, 7, 6, inf, 2, 0, 9],
            [14, inf, inf, inf, 8, 9, 0],
        ]
        edges = []
        for i in range(len(mat)):
            for j in range(len(mat)):
                edges.append((i, j, mat[i][j]))
        nodes = list(range(len(mat)))
        print('networkx 版本弗洛伊德算法:')
        print(multi_source_shortest_path(nodes, edges, force_use=1))
        print('tensorflow 版本弗洛伊德算法:')
        print(multi_source_shortest_path(nodes, edges, force_use=2))

        print('-' * 5, '速度测试')
        nodes = list(range(1000))
        edges = [(i, i + 1) for i in range(int(len(nodes) * 9 / 10))]
        start = time.time()
        multi_source_shortest_path(nodes, edges, force_use=3)
        print('加速计算树多源最短路径:', time.time() - start)
        edges += [(1, 3)]  # 变非树
        start = time.time()
        multi_source_shortest_path(nodes, edges, force_use=1)
        print('networkx 版本弗洛伊德算法:', time.time() - start)
        start = time.time()
        multi_source_shortest_path(nodes, edges, force_use=2)
        print('tensorflow 版本弗洛伊德算法:', time.time() - start)


if __name__ == '__main__':
    Test.create_gif()
    # Test.draw_line_chart()
    # Test.multi_source_shortest_path()
