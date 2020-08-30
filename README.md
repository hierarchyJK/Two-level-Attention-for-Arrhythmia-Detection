# A Two-level Attention-based Sequence to Sequence Model for Accurate Inter-patient Arrhythmia Detection
#### 基于双级注意力机制和seq2seq的高准确性患者间心律失常检测模型


> __Requirement__
* Python 3.6.8
* Tensorflow 1.14.0

> __Train__

    (CPU) python Train.py --use_SE True
    (GPU) CUDA_VISIBLE_DEVICES=0 python Train.py --use_SE True

**Notes 说明**
- --data_dir: Data directory, change this to your own path（数据路径，需要改为自己的路径）
- --checkpoint_dir: Directory where the model is saved, change this to your own path （保存模型的目录，需要改为自己的目录）
- --result_dir: Directory that saves the raw results, change this to your own path （保存每一次实验所有的epoch结果的目录，需要改为自己的目录）
- --other: Do NOT change the other perameters unless you know what you are doing. （其他的参数不需要改）

> **Inter-patient Performance 患者间心律性能**

All experimental results are averaged over ten runs. 取10次实验的平均结果

| Class\Metric | SEN | SPEC | PPV | ACC |
| :----: | :----: | :----: | :----: | :----:|
| N |0.9987|0.9856|0.9984|0.9973|
| S |0.9569|0.9989|0.9706|0.9973|
| V |0.9998|0.9997|0.9989|0.9997|

> **Ablation Study 消融实验**

    1. w/o local & w/o contextual
     (GPU) CUDA_VISIBLE_DEVICES=0 python Ablation1.py --use_Embedding True
    
    2. w/ local & w/o contextual
    （GPU）CUDA_VISIBLE_DEVICES=0 python Ablation1.py --use_Embedding True --use_SE True
    
    3. w/o local & w/ contextual
    （GPU）CUDA_VISIBLE_DEVICES=0 python Train.py 
    
    4. w/ local & w/ cotextual (origianl model)
    （GPU）CUDA_VISIBLE_DEVICES=0 python Train.py --use_SE True
**Notes 说明**
1. By executing the aforementioned code, you obtain the raw results from 10 runs. 直接运行以上几个命令，每个消融实验会自动跑10次。
2. You only need to change the parameters --checkpoint_dir (where you save the model) and --result_dir (where you save the raw results) for the ablation experiments. 每个消融实验，只需修改--checkpoint_dir和--result_dir，它们分别是保存模型的目录，以及保存每个Epoch中测试结果的目录

> Ablation Study Results 消融实验结果

1. w/o local & w/o contextual

| Class\Metric | SEN | SPEC | PPV | ACC |
| :----: | :----: | :----: | :----: | :----:|
| N |0.9967|0.9786|0.9975|0.9948|
| S |0.9395|0.9969|0.9232|0.9948|
| V |0.9991|0.9997|0.9964|0.9997|

2. w/ local & w/o contextual

| Class\Metric | SEN | SPEC | PPV | ACC |
| :----: | :----: | :----: | :----: | :----:|
| N |0.9978|0.9733|0.9970|0.9953|
| S |0.9235|0.9986|0.9614|0.9957|
| V |0.9988|0.9992|0.9889|0.9992|

3. w/o local & w/ contextual

| Class\Metric | SEN | SPEC | PPV | ACC |
| :----: | :----: | :----: | :----: | :----:|
| N |0.9976|0.9847|0.9983|0.9962|
| S |0.9556|0.9978|0.9448|0.9963|
| V |0.9989|0.9997|0.9959|0.9997|

4. w/ local & w/ cotextual: same as the results for the original model 即原始模型的实验结果

> **Intra-patient Performance 患者内性能**

| Class\Metric | SEN | SPEC | PPV | ACC |
| :----: | :----: | :----: | :----: | :----:|
| N |1.00|0.9970|0.9997|0.9997|
| S |0.9765|1.00|1.00|0.9994|
| V |1.00|0.9997|1.00|0.9997|
| F |1.00|1.00|1.00|1.00|

> **License 使用许可**
This code is for academic and non-commercial usage only. Also, our code is a modified version of the code of Dr. Sajad Mousavi and Dr. Fatemeh Afghah's excellent paper "Inter- and Intra-Patient ECG Heartbeat Classification For Arrhythmia Detection: A Sequence to Sequence Deep Learning Approach" (https://github.com/MousaviSajad/ECG-Heartbeat-Classification-seq2seq-model). We sincerely thank Dr. Mousavi for granting us permission to publish our modified code. If you wish to republish our code (including any modified versions of it), please first gain permission from Dr. Mousavi and Dr. Afghah. We are more than happy for you to republish our code as long as you have their permission to do so.



