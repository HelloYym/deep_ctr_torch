# DeepCTR-Torch

PyTorch version of [DeepCTR](https://github.com/shenweichen/DeepCTR).

框架具体细节参考文档 [**Get Started!**](https://deepctr-torch.readthedocs.io/en/latest/Quick-Start.html)

##### DeepCTR-Torch 框架改进：

原本支持 SparseFeat 和 DenseFeat 两种类型的 embedding 方法，增加了 VectorFeat 类型将原始特征向量转换为定长 embedding；

使用 DataParallel 支持多 GPU 训练；

针对 fideepfm 模型更改了 fit 方法；

删除一些模型的正则项；

##### 特征处理：

连续值特征离散化，转换为类别特征；

合并低频类别特征为相同的 ID，降低类别数目；

labelencoder；



 

