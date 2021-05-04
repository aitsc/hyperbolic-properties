from _ac_data_helper import *
from _af_train import metrics_to_results
from tanshicheng import TaskDB, Draw, where


class 数据生成任务(TaskDB):
    @property
    def result(self):
        result = super().result
        result.update({
            # 待执行任务参数模版
            'paras': {
                'func': '不平衡高低树',  # 生成随机图 的函数名 or 外部RG的描述, 会放在文件夹名中
                'in': {  # 传入 func 函数的参数, 会放在文件夹名中
                    'n': 2047, 'βμs': 2, 'βμe': 5, 'βtμ': 1., 'βσs': 0.4, 'βσe': 2, 'βtσ': 1.,
                    # 'RG_L': 'm1', 'μ': 0.1, 'σ': 0.1,  # 这里的RG_L属于mixed_tree
                },
                'saved_RG': '',  # 随机图的保存数据的路径, 有了这个以上in参数会被忽略
                'class': {  # 随机图.节点类别分配() 需要的参数
                    'n': 6, '分配方式': 'branch', '允许不同层次': True, '最大σμ比': 0.2
                },
                'mark': ['111'],  # 分类标记, 用于过滤查询, 会放在文件夹名中
                'mixed_tree': [],  # 属于某个多树混合图, list que_task匹配不考虑顺序
            },
            # 结果
            'time_start': 1617292720.1552498,  # 任务开始执行时间戳 time.time(), 会放在文件夹名中
            'data_path': {  # 数据路径, 全部相对路径, 都包含数据库主目录的唯一子目录xxx
                'RG': 'xxx/yyy.pkl',
                'dh': 'xxx/zzz.pkl',
            },
            'graph_info': {'describe': 'wordnet.mammal', '树数量': 1, '每棵树节点数量': [1182], '每棵树的层数': [10],
                           '每棵树不平衡程度': [0.055837372325049164], '每棵树层次度分布的偏移度': [1.1622529654407538],
                           '树总节点数量': 1182, 'nx图节点数量': 0, 'nx图边数量': 0},
            # 只有 run_task 或 结果过滤 依赖的结果, 数据生成任务 子类需继承上面变量
            'metrics': {  # 二维欧式展示方法的统计结果
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
            'mixed_tree_order': [],  # 如果是多树混合图, 这里用no编号记录子树的顺序
        })
        return result

    def run_task(self):
        tasks = self.get_uncomplete_tasks()
        tasks_mix = []
        完成任务 = 0
        while len(tasks) != 0 or len(tasks_mix) != 0:
            if len(tasks) != 0:
                task = tasks.pop(0)
            else:
                task = tasks_mix.pop(0)
            paras = copy.deepcopy(task['paras'])  # 后续可能需要修改
            # 混合树图有依赖最后执行
            mixed_tree_order = []
            if paras['func'] == 生成随机图.混合树图.__name__:
                if len(tasks) != 0:
                    tasks_mix.append(task)
                    continue
                else:  # 修改 RG_L 参数为实例随机图, 如果缺失所有图会报错, 部分图不会报错
                    RG_L = []
                    for i in self.que_task({'paras': {'mixed_tree': [paras['in']['RG_L']]}}):
                        RG_L.append(随机图(f"{self.db_dir}/{i['data_path']['RG']}"))
                        mixed_tree_order.append(i['no'])
                    paras['in']['RG_L'] = RG_L
            print('=' * 20, '本次任务参数:')
            pprint(task['paras'])
            # 结果过滤
            result = {}
            filter = task['filter']
            if not filter:  # 防止参数输入错误
                filter = {'test': None}
            while self.filtrate_result(result, **filter):
                # 生成随机图/dh
                if paras['saved_RG']:
                    RG = 随机图(paras['saved_RG'])
                else:
                    func = eval(f"生成随机图.{paras['func']}")
                    RG = func(**paras['in'])  # 防止paras中有非基本类型被转str, 不能全部eval
                RG.节点类别分配(**paras['class'])
                dataHelper = DataHelper(RG)
                # 文件夹名
                time_start = time.time()
                if not paras['saved_RG']:
                    paras_in = str(sorted(task['paras']['in'].items())).replace(' ', '').replace("'", '')
                else:
                    paras_in = ''
                folder_name_ = f"{'_'.join(paras['mark'])};{paras['func']};{paras_in};{time_start}"
                folder_name = f"{self.db_dir}/{folder_name_}"  # 有根目录
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                # 构建结果参数
                图统计信息 = RG.统计信息()
                节点数量 = 图统计信息['树总节点数量'] + 图统计信息['nx图节点数量']
                metrics = dataHelper.自动评估绘图(titleFrontText='$\\mathbb{R}$, ' + f'n:{节点数量}, ', 使用分类颜色=True,
                                            使用树结构颜色=True, 使用层次颜色=True,
                                            saveName=f'{folder_name}/_.pdf')[3]  # 获取信息和写入绘图
                result = {
                    'executed': True,
                    'main_path': folder_name_,
                    'machine': self.get_machine(),
                    'graph_info': 图统计信息,
                    'metrics': metrics_to_results(metrics),
                    'time_start': time_start,
                    'data_path': {
                        'RG': f'{folder_name_}/RG.pkl',
                        'dh': f'{folder_name_}/dh.pkl',
                    },
                    'mixed_tree_order': mixed_tree_order,
                }
            # 写入数据
            RG.保存到文件(f"{self.db_dir}/{result['data_path']['RG']}")
            dataHelper.保存数据(f"{self.db_dir}/{result['data_path']['dh']}")
            self.update_one(result, task['no'])
            完成任务 += 1
            print('=' * 20, '本次任务结果:')
            pprint(result)
            print('=' * 20, f'已完成第{完成任务}个任务, 剩余{len(tasks_mix) + len(tasks)}个.')
            print()

    def 统计结果(self):
        tasks = self._db_tasks.search(where('executed') == True)
        print('已完成任务数:', len(tasks), '; 未完成任务数:', len(self._db_tasks.all()) - len(tasks), '; 已完成任务:')
        pprint(tasks)
        # 绘制 不平衡程度 和 层次度分布的偏移度 的散点图
        tasks = self.que_task({'graph_info': {'nx图节点数量': 0}})
        describe_info = []  # [(描述,IB,ID,αr,αt,βμs,βμe),..]
        for task in tasks:
            if len(task['graph_info']['每棵树层次度分布的偏移度']) == len(task['graph_info']['每棵树不平衡程度']) == 1:
                ID = task['graph_info']['每棵树层次度分布的偏移度'][0]
                IB = task['graph_info']['每棵树不平衡程度'][0]
                if task['paras']['saved_RG']:
                    αr = αt = βμs = βμe = '?'
                else:
                    αr = task['paras']['in']['αr'] if 'αr' in task['paras']['in'] else 1
                    αt = task['paras']['in']['αt'] if 'αt' in task['paras']['in'] else 3
                    βμs = task['paras']['in']['βμs'] if 'βμs' in task['paras']['in'] else 2
                    βμe = task['paras']['in']['βμe'] if 'βμe' in task['paras']['in'] else 2
                if len(task['paras']['mixed_tree']) > 0:
                    des = '_'.join(task['paras']['mixed_tree'])
                else:
                    des = task['paras']['mark'][0]
                describe_info.append((des, IB, ID, αr, αt, βμs, βμe))
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.figure(1)
        labels = list(set([i[0] for i in describe_info]))
        label_color_D = {l: c for l, c in zip(labels, ncolors(len(labels)))}
        label_info_D = {}  # {描述:[颜色,[IB,..],[ID,..],[αr,..],[αt,..],[βμs,..],[βμe,..]],..}
        for label, IB, ID, αr, αt, βμs, βμe in describe_info:
            if label not in label_info_D:
                label_info_D[label] = [label_color_D[label], [IB], [ID], [αr], [αt], [βμs], [βμe]]
            else:
                label_info_D[label][1].append(IB)
                label_info_D[label][2].append(ID)
                label_info_D[label][3].append(αr)
                label_info_D[label][4].append(αt)
                label_info_D[label][5].append(βμs)
                label_info_D[label][6].append(βμe)
        markers = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|',
                   ''] * len(label_info_D)
        for i, (label, (c, IB_L, ID_L, αr_L, αt_L, βμs_L, βμe_L)) in enumerate(label_info_D.items()):
            plt.scatter(IB_L, ID_L, s=10, c=c, marker=markers[i], label=f'{label}-{len(IB_L)}')
            for IB, ID, αr, αt, βμs, βμe in zip(IB_L, ID_L, αr_L, αt_L, βμs_L, βμe_L):
                txt = f'({αr};{αt};{βμs};{βμe})'.replace('0.', '.')
                plt.annotate(txt, (IB, ID), fontsize=2)
        plt.title(r'IB_ID-($\alpha_r$;$\alpha_t$;$\beta_{\mu_s}$;$\beta_{\mu_e}$)')
        plt.xlabel('IB')
        plt.ylabel('ID')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{self.db_dir}/IB_ID_scatter.pdf', format='pdf')
        plt.show()
        plt.close()
        # 绘制 不平衡程度 和 层次度分布的偏移度 的三维透视图, 数量最多的点作为曲面, 长宽尽量接近正方形
        draw = Draw(length=16, width=12, r=2, c=2)
        xyz_L_L = [[], [], [], []]  # [xyz_L,xyz_L,xyz_L,xyz_L]; 4个指标对应的图
        surf_label_index = 0  # 哪个label点是曲面
        xyz_scatter_L = [[], [], [], []]
        scatter_labels = []  # 标签每个子图都一样
        for i, (label, (c, IB_L, ID_L, αr_L, αt_L, βμs_L, βμe_L)) in enumerate(label_info_D.items()):
            IB_ID_αr = []
            IB_ID_αt = []
            IB_ID_βμs = []
            IB_ID_βμe = []
            for IB, ID, αr, αt, βμs, βμe in zip(IB_L, ID_L, αr_L, αt_L, βμs_L, βμe_L):
                if αr != '?' and αt != '?' and βμs != '?' and βμe != '?':
                    IB_ID_αr.append((IB, ID, αr))
                    IB_ID_αt.append((IB, ID, αt))
                    IB_ID_βμs.append((IB, ID, βμs))
                    IB_ID_βμe.append((IB, ID, βμe))
            if IB_ID_αr:
                if len(IB_ID_αr) > len(xyz_L_L[0]):
                    surf_label_index = len(scatter_labels)
                    xyz_L_L[0] = IB_ID_αr
                    xyz_L_L[1] = IB_ID_αt
                    xyz_L_L[2] = IB_ID_βμs
                    xyz_L_L[3] = IB_ID_βμe
                xyz_scatter_L[0].append(IB_ID_αr)
                xyz_scatter_L[1].append(IB_ID_αt)
                xyz_scatter_L[2].append(IB_ID_βμs)
                xyz_scatter_L[3].append(IB_ID_βμe)
                scatter_labels.append(label)
        scatter_labels[surf_label_index] += '(surf)'  # 标注这是曲面点
        for n, (xyz_L, xyz_scatter, zlabel) in enumerate(
                zip(xyz_L_L, xyz_scatter_L, [r'$\alpha_r$', r'$\alpha_t$', r'$\beta_{\mu_s}$', r'$\beta_{\mu_e}$'])):
            draw.add_3d(xyz_L=xyz_L, xyz_scatter=xyz_scatter, scatter_labels=scatter_labels, interp_kind='linear',
                        xlabel='IB', ylabel='ID', zlabel=zlabel, n=n + 1)
        draw.draw(f'{self.db_dir}/IB_ID_3d.pdf')
        self.output_table()


def 构建数据任务(obj: 数据生成任务):
    paras_f = lambda: obj.result['paras']  # 获取参数模版的方法
    n = 3280  # 节点数
    paras_L_L = []  # [[{第1个参数},..],..]; 所有生成参数, 不同组mark不同

    # 0. 4个可变树
    def 四个可变树(paras_L, n=n, mark_pre='t', mixed_tree=None):
        if mixed_tree is None:
            mixed_tree = []
        paras = paras_f()
        paras['func'] = '完全多叉树'
        paras['mark'] = [f'{mark_pre}1']
        paras['mixed_tree'] = mixed_tree
        paras['in'] = {'n': n, 'm': 3}
        paras_L.append(paras)
        paras = paras_f()
        paras['func'] = '不平衡高低树'
        paras['mark'] = [f'{mark_pre}2']  # 不平衡多叉树
        paras['mixed_tree'] = mixed_tree
        paras['in'] = {'n': n, 'αr': 0.2, 'αt': 2, 'βμs': 5, 'βμe': 5}
        paras_L.append(paras)
        paras = paras_f()
        paras['func'] = '不平衡高低树'
        paras['mark'] = [f'{mark_pre}3']  # 低高多叉树
        paras['mixed_tree'] = mixed_tree
        paras['in'] = {'n': n, 'βμs': 2, 'βμe': 7, 'βtμ': 1., 'βσs': 0.4, 'βσe': 1.5, 'βtσ': 1.}
        paras_L.append(paras)
        paras = paras_f()
        paras['func'] = '不平衡高低树'
        paras['mark'] = [f'{mark_pre}4']  # 高低多叉树
        paras['mixed_tree'] = mixed_tree
        paras['in'] = {'n': n, 'βμs': 6, 'βμe': 1, 'βtμ': 0.3, 'βσs': 0.1, 'βσe': 1., 'βtσ': 1.}
        paras_L.append(paras)
        return paras_L

    paras_L_L.append(四个可变树([], n=n, mark_pre='t', mixed_tree=None))
    # 1. 4个可变图
    paras_L = []
    paras = paras_f()
    paras['func'] = '无标度图'
    paras['mark'] = ['g1']
    paras['in'] = {'n': n, 'm': 2, 'p': 0.2, 'q': 0.2}
    paras_L.append(paras)
    paras = paras_f()
    paras['func'] = '小世界网络'
    paras['mark'] = ['g2']
    paras['in'] = {'n': n, 'k': 3, 'p': 0.4}
    paras_L.append(paras)
    paras = paras_f()
    paras['func'] = 'ER随机图'
    paras['mark'] = ['g3']
    paras['in'] = {'n': n, 'p': 2 / n}
    paras_L.append(paras)
    paras = paras_f()
    paras['func'] = '规则图'
    paras['mark'] = ['g4']
    paras['in'] = {'n': n, 'd': 3}
    paras_L.append(paras)
    paras_L_L.append(paras_L)
    # 2. 3个公开数据集
    paras_L = []
    paras = paras_f()
    paras['func'] = 'disease_lp'
    paras['mark'] = ['o1']
    paras['saved_RG'] = 'ag_disease_lp;1;[2665];[5];[0.0];[1.0268];2665;0.pkl'
    paras_L.append(paras)
    paras = paras_f()
    paras['func'] = 'disease_nc'
    paras['mark'] = ['o2']
    paras['saved_RG'] = 'ag_disease_nc;1;[1044];[6];[0.0];[0.9725];1044;0.pkl'
    paras_L.append(paras)
    paras = paras_f()
    paras['func'] = 'animal'
    paras['mark'] = ['o3']
    paras['saved_RG'] = 'aj_wordnet.animal;1;[4017];[14];[0.0582];[1.8894];4017;0.pkl'
    paras_L.append(paras)
    paras_L_L.append(paras_L)
    # 3. 1个混合树图+8个子树
    paras_L = []
    paras = paras_f()
    paras['func'] = '混合树图'
    paras['mark'] = ['t5']
    paras['in'] = {'RG_L': 'm1', 'μ': 0.1, 'σ': 0.1}
    paras_L.append(paras)
    四个可变树(paras_L, n=int(n / 3), mark_pre='t5.1.', mixed_tree=['m1'])
    四个可变树(paras_L, n=int(n / 3), mark_pre='t5.2.', mixed_tree=['m1'])
    paras_L_L.append(paras_L)
    # 4. 多个可变树
    paras_L = []
    αr_αt_βμs_βμe_L = [
        (0.0, 4.4, 22, 1),  # IB: 0.9741 ; ID: 0.9814
        (0.14, 3, 14, 1),  # IB: 0.9872 ; ID: 0.8149
        (0.1, 3, 7, 3),  # IB: 0.9868 ; ID: 0.6485
        (0.03, 3, 4, 6),  # IB: 0.9948 ; ID: 0.4434
        (0.15, 3, 2, 11),  # IB: 0.9732 ; ID: 0.1962
        (0.0, 4.5, 2, 19),  # IB: 0.9666 ; ID: 0.0156
        (0.0, 3.0, 20, 1),  # IB: 0.833 ; ID: 0.9652
        (0.25, 3, 13.5, 1),  # IB: 0.8308 ; ID: 0.8056
        (0.45, 3, 9, 1),  # IB: 0.8249 ; ID: 0.6449
        (0.16, 3, 4, 6),  # IB: 0.8342 ; ID: 0.4445
        (0.22, 3, 2, 12),  # IB: 0.8478 ; ID: 0.1852
        (0.01, 3, 2, 20),  # IB: 0.8095 ; ID: 0.0326
        (0.17, 4, 20, 1),  # IB: 0.5699 ; ID: 0.9764
        (0.33, 3.5, 13.5, 1),  # IB: 0.6039 ; ID: 0.8277
        (0.47, 3, 6.5, 2),  # IB: 0.5953 ; ID: 0.6278
        (0.49, 3, 2, 7),  # IB: 0.6381 ; ID: 0.3714
        (0.32, 3, 2, 12),  # IB: 0.6262 ; ID: 0.1995
        (0.06, 3, 2, 22),  # IB: 0.6367 ; ID: 0.0246
        (0.19, 3.3, 18, 1),  # IB: 0.3967 ; ID: 0.9667
        (0.365, 3, 12, 1),  # IB: 0.4404 ; ID: 0.7908
        (0.26, 3, 6.3, 3.8),  # IB: 0.4112 ; ID: 0.5705
        (0.28, 3, 4, 6),  # IB: 0.431 ; ID: 0.4459
        (0.4, 3, 2, 12),  # IB: 0.4179 ; ID: 0.203
        (0.22, 3, 2, 20),  # IB: 0.4107 ; ID: 0.0415
        (0.18, 3, 18, 1),  # IB: 0.2095 ; ID: 0.9828
        (0.39, 3, 12, 1),  # IB: 0.2278 ; ID: 0.8373
        (0.35, 3, 6.3, 3.8),  # IB: 0.1823 ; ID: 0.5749
        (0.35, 3, 4, 6),  # IB: 0.2296 ; ID: 0.4458
        (0.5, 3, 2, 12),  # IB: 0.232 ; ID: 0.1998
        (0.35, 3, 2, 22),  # IB: 0.1835 ; ID: 0.0214
        (1, 1, 10, 1),  # IB: 0.0002 ; ID: 0.983
        (1, 1, 7.5, 2.5),  # IB: 0.0003 ; ID: 0.816
        (1, 1, 6, 3),  # IB: 0.0003 ; ID: 0.5955
        (1, 1, 3, 7),  # IB: 0.0004 ; ID: 0.3846
        (1, 1, 2, 8),  # IB: 0.0005 ; ID: 0.2227
        (1, 1, 2, 15),  # IB: 0.0005 ; ID: 0.0302
    ]
    for i, (αr, αt, βμs, βμe) in enumerate(αr_αt_βμs_βμe_L):
        paras = paras_f()
        paras['func'] = '不平衡高低树'
        paras['mark'] = ['t6', f'c{i}']  # 高低多叉树
        paras['in'] = {'n': n, 'βμs': βμs, 'βμe': βμe, 'αr': αr, 'αt': αt}
        paras_L.append(paras)
    # αr_range = [.0, .1, .14, .18, .21, .25, .29, .38, .9]
    # βμs_range = [10, 9, 7.5, 7, 5, 3, 2, 2, 2]
    # βμe_range = [1, 2, 3, 4, 5, 7, 10, 13, 15]
    # assert len(βμs_range) == len(βμe_range), 'len(βμs_range) != len(βμe_range)'
    # print('可变树数量:', len(αr_range) * len(βμs_range))
    # for i, αr in enumerate(αr_range):
    #     for j, (βμs, βμe) in enumerate(zip(βμs_range, βμe_range)):
    #         paras = paras_f()
    #         paras['func'] = '不平衡高低树'
    #         paras['mark'] = ['t6', f'c{i}.{j}']  # 高低多叉树
    #         paras['in'] = {'n': n, 'βμs': βμs, 'βμe': βμe, 'αr': αr, 'αt': 3}
    #         paras_L.append(paras)
    paras_L_L.append(paras_L)
    return paras_L_L


if __name__ == '__main__':
    # 数据生成任务.test()

    构建新任务 = False
    重构可变树 = False  # 构建新任务 = False 时
    路径 = 'al_all_data'

    if not os.path.exists(路径):
        构建新任务 = True
    start = time.time()
    if 构建新任务:
        print('构建新任务:')
        obj = 数据生成任务(路径, new=True)
        info_L = []
        for paras_L in 构建数据任务(obj):
            for paras in paras_L:
                if 'RG_L' in paras['in'] and paras['in']['RG_L']:
                    priority = 0
                else:
                    priority = 1
                info_L.append({'paras': paras, 'priority': priority})
        print('一共增加任务数:', len(obj.add_tasks(info_L)))
    else:
        obj = 数据生成任务(路径)
        if 重构可变树:
            print('重构可变树:')
            obj.del_task({'paras': {'mark': ['t6']}})
            info_L = []
            for paras in 构建数据任务(obj)[4]:
                if 'RG_L' in paras['in'] and paras['in']['RG_L']:
                    priority = 0
                else:
                    priority = 1
                info_L.append({'paras': paras, 'priority': priority})
            print('一共重构任务数:', len(obj.add_tasks(info_L)))
    obj.clean()
    obj.run_task()
    print('=' * 10, '统计结果:')
    obj.统计结果()
    obj.close()
    print('总耗时:', (time.time() - start) / 3600, '小时')
