
## 安装
pip install -r requirements.txt
pip install -r requirements2.txt  # 安装这个需要依赖其他, 不安装这个也能运行, 可能缺少一些功能, 例如 pygraphviz


## 说明
_am_create_all_train.py: 基本可以直接执行: 不会修改已执行的任务和代码,未执行的任务会被删除,可能补充一些新的未执行任务

_an_all_result_draw.py: 早期绘图;

_ap_all_result_draw2.py: 修改增加维度平均绘图,支持两个公开数据集;

_aq_redraw.py: data 重新绘图可视化, 因为指标展示的问题;