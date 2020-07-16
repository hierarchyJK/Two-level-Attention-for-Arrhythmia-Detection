# ECG_Seq2Seq
#### 该实验引入注意力机制对病人间基于心拍分类的心律失常进行检测


> __Requirement__
* Python 3.6.8
* Tensorflow 1.14.0

> __Train__

    (CPU) python Train.py --use_Embedding
    (GPU) CUDA_VISIBLE_DEVICES=0 python Train.py --use_Embedding
**其他说明**：
- --data_dir：为数据路径，需要改为自己的路径
- --checkpoint_dir：保存模型的目录，改为自己的目录
- --result_dir：保存每一次实验所有的epoch结果的目录，改为自己的目录
- --other:其他的参数不需要改
> **实验结果**

实验结果在保持一些参数不改变的情况下，训练十次取的平均结果，如下表

| Type\indice | SEN | SPEC | PPV | ACC |
| :----: | :----: | :----: | :----: | :----:|
| N |0.9987|0.9856|0.9984|0.9973|
| S |0.9569|0.9989|0.9706|0.9973|
| V |0.9998|0.9997|0.9989|0.9997|
> ** 消融实验 **

    1. 不带Attention的CNN + Seq2Seq
     (GPU) CUDA_VISIBLE_DEVICES=0 python Ablation1.py --use_Embedding

    2. 直接Seq2Seq(需要对输入进行标准化)，即不需要Embedding层
     (GPU) CUDA_VISIBLE_DEVICES=0 python Ablation1.py
**其他说明**
1. 对于以上消融实验，跑10次，每次结果通过result_dir保存