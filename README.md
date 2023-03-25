# Dual-Cross
This is the code related to "Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation" (ACMMM 2022).
![](https://github.com/limiaoyu/Dual-Cross/blob/main/Dual-Cross.jpg)
## Paper
[Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation](https://dl.acm.org/doi/10.1145/3503161.3547990)

**MM '22: Proceedings of the 30th ACM International Conference on Multimedia**

If you find it helpful to your research, please cite as follows:
```
@article{Li2022CrossDomainAC,
  title={Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation},
  author={Miaoyu Li and Yachao Zhang and Yuan Xie and Zuodong Gao and Cuihua Li and Zhizhong Zhang and Yanyun Qu},
  journal={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```
## Preparation (Refer to [xMUDA](https://github.com/valeoai/xmuda))
### Prerequisites
Tested with
* PyTorch 1.4
* CUDA 10.0
* Python 3.8
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

### Installation
As 3D network we use SparseConvNet. It requires to use CUDA 10.0 (it did not work with 10.1 when we tried). We advise to create a new conda environment for installation. PyTorch and CUDA can be installed, and SparseConvNet installed/compiled as follows:
```
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
$ pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
```

```
$ cd xmuda
$ pip install -ve .
```
The `-e` option means that you can edit the code on the fly.
## Datasets
### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org) and extract it.

You need to perform preprocessing to generate the data for Dual-Cross.

Please edit the script `xmuda/data/nuscenes/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the NuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files

## Usage
### Training
You can run the training with:
```
$ python xmuda/train_dual.py --cfg=configs/nuscenes/day_night/xmuda.yaml 
```
You can change the path OUTPUT_DIR in the config file `xmuda.yaml`.
### Testing
You can provide which checkpoints you want to use for testing. We used the ones that performed best on the validation set during training (the best valiteration for 2D and 3D is shown at the end of each training). Note that @ will be replaced by the output directory for that config file. For example:
```
$ python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda.yaml  @/model_2d_065000.pth @/model_3d_095000.pth
```
You can also provide an absolute path without `@`. 

## Acknowledgements
Note that this code borrows from the [xMUDA](https://github.com/valeoai/xmuda) repo.
