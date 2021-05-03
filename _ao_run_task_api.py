from tanshicheng import TaskDBapi, get_logger, TaskDB
from _af_train import 随机图, DataHelper, metrics_to_results, main_api, tf, datetime
from _ak_sala2018comb import 自动评估绘图
import os
from pprint import pprint, pformat
import time
import shutil

logger = get_logger(f'log/{os.path.split(__file__)[1]}.log')


def run_task_train(dataset_db_path, db_dir, db_api, url):
    """
    通过api运行任务
    :param dataset_db_path: 数据生成任务_obj 的位置, 用于寻找数据. 任务执行依赖其他 TaskDB
    :param db_dir: 生成数据的主目录, 用于保存数据, 最后其子目录需要剪切到api数据库的位置
        没有会自动生成, 如果新生成的子文件夹也存在会被删除
    :param db_api: 访问 api 的哪个数据库
    :param url: str; 获取数据的api接口
    :return:
    """
    all_time = time.time()
    完成任务 = 1
    api写入任务 = 0
    response = TaskDBapi.request_api(request_data={'type': 'request', 'db': db_api}, url=url, db_dir=db_dir)
    while 'task' in response and response['task']:
        paras = response['task']['paras']
        # 增加数据集db位置
        if 'comb' in paras and paras['comb']['RG'] and paras['comb']['dh']:
            paras['comb']['RG'] = f"{dataset_db_path}/" + paras['comb']['RG']
            paras['comb']['dh'] = f"{dataset_db_path}/" + paras['comb']['dh']
        if 'trainParas' in paras and paras['trainParas']['dh_path']:
            paras['trainParas']['dh_path'] = f"{dataset_db_path}/" + paras['trainParas']['dh_path']
        # 提示
        print('=' * 20, f'开始任务({完成任务})参数({datetime.datetime.now()}):')
        pprint(paras)
        logger.critical(f'开始任务({完成任务})参数:\n' + pformat(paras))
        # 文件夹名
        time_start = time.time()
        folder_name_ = f"{'_'.join(paras['mark'])};{time_start}/"
        folder_name = f"{db_dir}/{folder_name_}/"
        if os.path.exists(folder_name):  # 存在则删除
            shutil.rmtree(folder_name)
        if not os.path.exists(folder_name):  # 不存在新建
            os.makedirs(folder_name)
        # 构建结果参数
        is_gpu = None
        if 'comb' in paras and os.path.exists(paras['comb']['RG']) and os.path.exists(paras['comb']['dh']):
            RG = 随机图(paras['comb']['RG'])
            dataHelper = DataHelper(load_file=paras['comb']['dh'])
            metrics = 自动评估绘图(RG, dataHelper, f'{folder_name}/_.eps', **paras['comb'])[0]
            result = {
                'epoch': {'0': {'to_m': {'2': metrics_to_results(metrics)}}},
                'dh_graph_info': dataHelper.data['图统计信息'],
                'best_result': None,
            }
            is_gpu = False
        else:
            paras['trainParas']['ap'] = folder_name  # 这个导致返回参数会用到 db_dir
            try:
                result = main_api(**paras)
            except:
                logger.error('尝试使用cpu运行 main_api', exc_info=True)
                with tf.device('/cpu:0'):
                    result = main_api(**paras)
                logger.critical('使用cpu运行 main_api 成功!')
                is_gpu = False
        result = {
            'executed': True,
            'main_path': folder_name_,
            'machine': TaskDB.get_machine(is_gpu=is_gpu),
            'graph_info': result['dh_graph_info'],
            'time_start': time_start,
            'result_all': result,
        }
        logger.critical(f'完成任务({完成任务})结果:\ngraph_info:\n' +
                        pformat(result['graph_info']) + '\nbest_result:\n' +
                        pformat(result['result_all']['best_result']))
        # 更新任务
        response = TaskDBapi.request_api(request_data={'type': 'complete', 'db': db_api, 'no': response['task']['no'],
                                                       'result': str(result)}, url=url, db_dir=db_dir)
        if 'status' in response and response['status'] >= 1:  # 表示完成
            api写入任务 += 1
        else:
            print('服务器未写入结果!')
            logger.critical('服务器未写入结果!')
        print('=' * 20, '本次任务结果:')
        pprint(result)
        out = f'已完成{完成任务}个任务, api已写入{api写入任务}个任务, 本次耗时{(time.time() - time_start) / 60}分钟.'
        print('=' * 20, out)
        logger.critical(out)
        完成任务 += 1
        print()
        response = TaskDBapi.request_api(request_data={'type': 'request', 'db': db_api}, url=url, db_dir=db_dir)
    print(f'总耗时: {(time.time() - all_time) / 3600}h; 最后一次 response:')
    pprint(response)


if __name__ == '__main__':
    url = 'http://10.10.1.101:38000/api'
    if TaskDBapi.request_api({}, url=url, try_times=3):
        run_task_train(
            dataset_db_path='al_all_data',
            db_dir='ao_result',
            db_api='am_all_train',
            url=url,
        )
    else:
        print('\n启动服务端:')
        TaskDBapi.app_run(
            db_dirs=['am_all_train'],
            port=19999,
            log_path=f'log/TaskDBapi.app_run.log',
        )
