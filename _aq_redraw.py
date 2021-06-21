from _am_create_all_train import *

obj_data = 数据生成任务('al_all_data')
for data in ['o2', 'o3', 't1', 't2', 't3', 't4']:
    task = obj_data.que_tasks({'paras': {'mark': [data]}})[0]
    print('提取任务:', task['paras']['mark'])
    load_file = f"{obj_data.db_dir}/{task['data_path']['dh']}"
    dataHelper = DataHelper(load_file=load_file)
    图统计信息 = dataHelper.data['图统计信息']
    节点数量 = 图统计信息['树总节点数量'] + 图统计信息['nx图节点数量']
    metrics = dataHelper.自动评估绘图(titleFrontText='$\\mathbb{R}$, ' + f'n:{节点数量}, ', 使用分类颜色=True,
                                使用树结构颜色=True, 使用层次颜色=True,
                                saveName=f'{obj_data.db_dir}/{task["main_path"]}/_.pdf')[3]  # 获取信息和写入绘图
    if data == 't5':  # 多树混合图更新图失真指标, 每次会随机构建所以指标会不同
        print('多树混合图更新图失真指标...')
        obj_data.update_tasks([{'metrics': metrics_to_results(metrics)}], [task['no']])
        dataHelper.保存数据(load_file)
    print()
obj_data.output_table()
