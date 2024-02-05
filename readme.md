## Install and run
- conda create -n 3_hp python=3.8 && conda activate 3_hp
- pip install -r requirements.txt
- pip install -r requirements2.txt  # Installing this requires dependencies, but it can still run without installing it. However, some features may be missing, such as pygraphviz.
- get ready: al_all_data, connect.txt, cuda
- ta -t -a _ao_run_task_api.py
- ta -t -o python -u _ao_run_task_api.py -q "\"[{'\$match': {'paras.mark': {'\$nin': ['mt0']}}}]\""  # Way of adding constraints.

## Explanation
_am_create_all_train.py: Can be executed directly: It will not modify the tasks and code that have been executed. Unexecuted tasks will be deleted, and some new unexecuted tasks may be added.

_an_all_result_draw.py: Early drawing.;

_ap_all_result_draw2.py: Modify and add dimension average plotting, supporting two public datasets.;

_aq_redraw.py: data Redraw visualization due to issues with displaying indicators;

## Download
Data, visualization charts, and complete experimental details download: https://pan.baidu.com/s/16olIfyZrwnBCdViYOk-mVQ?pwd=3djk

## Citation
```bibtex
@article{tan2024why,
  title = {Why are hyperbolic neural networks effective? A study on hierarchical representation capability},
  author = {Tan, Shicheng and Zhao, Huanjing and Zhao, Shu and Zhang, Yanping},
  journal = {arXiv},
  year = {2024},
  type = {Journal Article}
}
```
