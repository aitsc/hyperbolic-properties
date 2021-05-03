import datetime
from _ac_data_helper import *


class Combinatorial:
    @staticmethod
    def comb(julia='/Applications/Julia-0.7.app/Contents/Resources/julia/bin/julia',
             dataset='data/edges/phylo_tree.edges',
             comb_path='/Users/tanshicheng/git/paper/hyperbolics-julia-0.7', save='phylo_tree.r10.emb', stats=True,
             eps=1.0, dtype=3200, dim=2, procs=8, scale=None, forest=False, use_codes=False,
             save_dist=None, **kwargs):
        """
        调用原始 comb.jl 文件
        :param julia: str; julia executable 路径
        :param dataset: str; 数据集路径, 文件夹可用于森林. 相对于comb_path的路径
            文件格式: 连续的点, 从0开始, 父节点在前子节点在后, 编号不能断
        :param comb_path: str; Combinatorial Constructions 算法文件夹路径
        :param save: str; 嵌入保存路径. 相对于comb_path的路径
        :param stats: bool; 是否计算MAP等统计信息
        :param eps: float; Epsilon distortion
        :param dtype: int; 精度位数
        :param dim: int; 庞加莱球模型输出维度
        :param procs: int; 进程数量
        :param scale: None or float; 为None时scale受eps和dtype自动控制, 为float时eps和dtype失效, 值大于0
            注意: 越接近0效果越差, 过大也会报错(比如默认数据集用6,使用dtype可以自动变大), 越大向量值越近庞加莱球边缘(接近1)
        :param forest: bool; dataset是否是森林数据集(文件夹,多棵树)
        :param use_codes: bool; 是否使用 coding-theoretic child placement, 否则使用 uniform sphere child placement
        :param save_dist: None or str; 不为空则保存距离矩阵, 为距离矩阵保存路径. 相对于comb_path的路径
            文件格式: 第一行和第一列为点编号, 文件被分为n个文件存储, 文件后缀增加'.i', i从0到n-1
                源代码默认一个文件最多chunk_sz(1000)个点和所有点距离, 第一行点编号增序排序
        :param kwargs: dict; 兼容变量
        :return:
        """
        if stats:
            stats = ' -s'
        else:
            stats = ''
        if scale:
            scale = f' -t {scale}'
        else:
            scale = ''
        if forest:
            forest = ' -f'
        else:
            forest = ''
        if use_codes:
            use_codes = ' -c'
        else:
            use_codes = ''
        if save_dist:
            save_dist = f' -y "{save_dist}"'
        else:
            save_dist = ''
        return os.system(f'cd {comb_path}\n'
                         f'{julia} combinatorial/comb.jl -d "{dataset}" -m "{save}" -e {eps} -p {dtype} -r {dim}'
                         f'{scale}{stats}{forest}{use_codes}{save_dist} -q {procs} -a')

    @staticmethod
    def comb_api(edges, save_tmp=False, get_dist=True, **kwargs):
        """
        用于直接获取comb嵌入向量和距离
        :param edges: [(点1,点2),..]; 边数据, 前是父节点, 后是子节点. 需要一颗树的边, 无重复, 无自环
        :param save_tmp: bool; 是否保存临时生成的数据集和输出文件
        :param get_dist: bool; 是否计算距离
        :param kwargs: dict; 参数见 comb 方法, 除了数据集/输出嵌入/输出距离文件路径不需要
        :return: {点:[嵌入],..}, [(点1,点2,距离),..], [(点,离原点的距离),..]
        """
        # variable
        p_no = {}  # {点:点编号,..}; 用于编号
        no_p = {}  # {点编号:点,..}; 用于还原点
        points_emb = {}  # {点:[嵌入],..}
        p2p_dist = []  # [(点1,点2,距离),..]; 上三角所有距离, 不含对角线
        p0_dist = []  # [(点,离原点的距离),..]
        t = str(datetime.datetime.now())
        kwargs['dataset'] = f'{os.getcwd()}/ak_tmp_dataset {t}.txt'
        kwargs['save'] = f'{os.getcwd()}/ak_tmp_emb {t}.txt'
        if get_dist:
            kwargs['save_dist'] = f'{os.getcwd()}/ak_tmp_dist {t}.txt'

        # save dataset and run the comb
        i = 0  # point no start from 0
        for x, y in edges:
            for p in [x, y]:
                if p not in p_no:
                    p_no[p] = i
                    no_p[i] = p
                    i += 1
        with open(kwargs['dataset'], 'w', encoding='utf-8') as w:
            for x, y in edges:
                w.write(f'{p_no[x]} {p_no[y]}\n')
        print('运行comb:', Combinatorial.comb(**kwargs))

        # extract embedding
        with open(kwargs['save'], 'r', encoding='utf-8') as r:
            r.readline()
            for line in r:
                line = line.strip().split(',')[:-1]
                if len(line) < 2:
                    continue
                points_emb[int(line[0])] = [float(i) for i in line[1:]]
        # extract distence
        if get_dist:
            i = 0
            distence = [[]] * len(p_no)
            while True:
                dist_file = f"{kwargs['save_dist']}.{i}"
                i += 1
                if not os.path.exists(dist_file):
                    break
                with open(dist_file, 'r', encoding='utf-8') as r:
                    r.readline()
                    for line in r:
                        line = line.strip().split(',')
                        if len(line) < 2:
                            continue
                        distence[int(line[0])] = [float(j) for j in line[1:]]
            for i in range(len(p_no) - 1):
                p0_dist.append((no_p[i], distence[i][0]))
                for j in range(i + 1, len(p_no)):
                    p2p_dist.append((no_p[i], no_p[j], distence[i][j]))
            p0_dist.append((no_p[len(p_no) - 1], distence[-1][0]))

        # delete tmp file
        if not save_tmp:
            os.remove(kwargs['dataset'])
            os.remove(kwargs['save'])
            i = 0
            while get_dist:
                try:
                    os.remove(f"{kwargs['save_dist']}.{i}")
                    i += 1
                except:
                    break
        return points_emb, p2p_dist, p0_dist

    @staticmethod
    def test():
        edges = []
        with open('/Users/tanshicheng/git/paper/hyperbolics-julia-0.7/data/edges/phylo_tree.edges', 'r') as r:
            for line in r:
                line = line.strip().split(' ')
                if len(line) > 1:
                    edges.append((line[0], line[1]))
        points_emb, p2p_dist, p0_dist = Combinatorial.comb_api(edges)
        print(points_emb)
        print(p2p_dist)
        print(p0_dist)


def 自动评估绘图(RG_obj: 随机图, dataHelper: DataHelper, saveName: str, print_metrics=False, dim=2, **kwargs):
    # 设置备选的 julia comb_path, 以兼容不同的电脑配置环境
    julia_L = [
        kwargs['julia'] if 'julia' in kwargs else '',
        '~/soft/julia/julia-0.7.0/bin/julia',
        '/Applications/Julia-0.7.app/Contents/Resources/julia/bin/julia',
        '',
        '',
    ]
    comb_path_L = [
        kwargs['comb_path'] if 'comb_path' in kwargs else '',
        '~/git/paper/hyperbolics-julia-0.7',
        '',
        '',
        '',
    ]
    for julia in julia_L:
        julia = os.path.expanduser(julia)
        if os.path.exists(julia):
            kwargs['julia'] = julia
            break
    for comb_path in comb_path_L:
        comb_path = os.path.expanduser(comb_path)
        if os.path.exists(comb_path):
            kwargs['comb_path'] = comb_path
            break

    points_emb, p2p_dist, p0_dist = Combinatorial.comb_api(RG_obj.树_获取有向边()[0], dim=dim, get_dist=True, **kwargs)
    metrics = dataHelper.评估(p2p_dist, p0_dist, 计算节点距离保持指标=True, 强制计算图失真=True)
    if print_metrics:
        pprint(metrics)
    if dim == 2:
        fname, fextension = os.path.splitext(saveName)
        if RG_obj.type == 随机图.类型.混合树图:
            saveName_ = fname + '.T' + fextension
            dataHelper.绘图(points_emb, 使用树结构颜色=True, title='Comb, $\\mathbb{D}$, ', saveName=saveName_,
                          useLegend=True, metrics=metrics)
        else:
            saveName_ = fname + '.h' + fextension
            dataHelper.绘图(points_emb, title='Comb, $\\mathbb{D}$, ', saveName=saveName_, useLegend=True, metrics=metrics)
        saveName_ = fname + '.c' + fextension
        dataHelper.绘图(points_emb, 使用分类颜色=True, title='Comb, $\\mathbb{D}$, ', saveName=saveName_,
                      useLegend=True, metrics=metrics)
    return metrics, points_emb, p2p_dist, p0_dist


if __name__ == '__main__':
    # Combinatorial.test()

    RG = 随机图('aj_wordnet.mammal;1;[1182];[10];[0.0558];[1.1623];1182;0.pkl')
    # RG = 随机图('ab_完全多叉树;1;[1023];[10];[0.0];[1.0];1023;0.pkl')
    自动评估绘图(RG, DataHelper(RG), 'ak.test.eps', dtype=32, dim=2, print_metrics=True)
