[TOC]

# 赛题背景

[CCF2020训练赛：通用音频分类](https://www.datafountain.cn/competitions/486)

- **赛题名**：通用音频分类

- **赛道**：训练赛道

- **背景**：随着移动终端的广泛应用以及数据量的不断积累，海量多媒体信息的处理需求日益凸显。作为多媒体信息的重要载体，音频信息处理应用广泛且多样，如自动语音识别、音乐风格识别等。有些声音是独特的，可以立即识别，例如婴儿的笑声或吉他的弹拨声。有些音频背景噪声复杂，很难区分。如果闭上眼睛，您能说出电锯和搅拌机是下面哪种声音？音频分类是音频信息处理领域的一个基本问题，从本质上说，音频分类的性能依赖于音频中的特征提取。传统特征提取算法使用音频特征的统计信息作为分类的依据,使用到的音频特征包括线性预测编码、短时平均能量等。近年来，基于深度学习的音频分类取得了较大进展。基于端到端的特征提取方式，深度学习可以避免繁琐的人工特征设计。音频的多样化给“机器听觉”带来了巨大挑战。如何对音频信息进行有效的分类,从繁芜丛杂的数据集中将具有某种特定形态的音频归属到同一个集合，对于学术研究及工业应用具有重要意义。

- **任务**：基于上述实际需求以及深度学习的进展，本次训练赛旨在构建通用的基于深度学习的自动音频分类系统。通过本赛题建立准确的音频分类模型，希望大家探索更为鲁棒的音频表述方法，以及转移学习、自监督学习等方法在音频分类中的应用。
- 训练集大约6万条音频数据，测试集大约6千条。一共30类，采样率为16000，每条数据大约1秒。打榜指标为accuracy。
- [代码地址](https://github.com/3017218062/Universal-Audio-Classification)

# 文件概括

- \_\_init\_\_\.py：导入所需的库。
- arg\.py：命令行参数。
- callback\.py：进度条、日志等辅助工具。
- dataset\.py：数据集文件。
- model\.py：定义模型和训练逻辑。
- preprocess\.py：预处理和数据划分。
- transform\.py：数据增强文件。
- util\.py：指标和损失函数。
- train\.py：训练文件。
- inference\.py：推理文件。

# 环境要求

- 硬件：2080Ti*5
- 框架：Pytorch1.6，Pytorch Lightning
- 库：见requirements.txt
- 数据：修改train\.py和inference中的input_path为训练集路径

# 文件运行

- 训练：
  - python train.py -t 224 -m "dla60_res2next" -f 0 -g 0
  - python train.py -t 224 -m "dla60_res2next" -f 1 -g 1
  - python train.py -t 224 -m "dla60_res2next" -f 2 -g 2
  - python train.py -t 224 -m "dla60_res2next" -f 3 -g 3
  - python train.py -t 224 -m "dla60_res2next" -f 4 -g 4
- 推理：
  - python inference.py -t 224 -m "dla60_res2next" -f 5 -a "y"

# 总体思路

- 将数据进行五折划分，使用第一折进行试验。
- 使用librosa.feature.melspectrogram提取频谱图，从小分辨率开始实验（高32维持不变），注意归一化。
- 数据增强主要是高斯噪声、音频偏移和音量调节。
- 从resnet18开始，依次替换为更大更复杂的模型。
- 找到最终模型后进行五折集成。
- 进行不同种类模型的集成。
- 进行测试时增强集成。

# 实验过程

- 0.95259692758
  - 模型：resnet50
  - n_mels：64
- 0.95610826628
  - 模型：resnet50d
  - n_mels：64
- 0.95918068764
  - 模型：res2next50
  - n_mels：64
- 0.96576444770
  - 模型：res2next50
  - n_mels：64
  - width：64
- 0.96971470373
  - 模型：res2next50
  - n_mels：128
  - width：128
  - more augment
- 0.96898317484
  - 模型：resnest50d
  - n_mels：128
  - width：128
  - more augment
- 0.97307973665
  - 模型：res2next50
  - n_mels：224
  - width：224
  - more augment
- 0.97527432334
  - 模型：res2next50
  - n_mels：224
  - width：224
  - more augment
  - 5-fold hard ensemble
- 0.97542062911
  - 模型：res2next50
  - n_mels：224
  - width：224
  - more augment
  - 5-fold soft ensemble
- 0.97585954645
  - 模型：res2next50
  - n_mels：224
  - width：224
  - more augment
  - 5-fold soft ensemble
  - 4TTA
- 0.97527432334
  - 模型：res2next50
  - n_mels：224
  - width：224
  - more augment
  - 5-fold soft ensemble
  - 4TTA
  - smooth0.1
  - ohem0.9

# 反思总结

- 更大的分辨率可以达到更好的效果，但对机器要求也会随之提高。
- efficientnet系列训练快，效果好，但容易过拟合。
- 五折和TTA永远的神。
- 数据增强时不要使用音调调整，太慢了。
- 标签平滑为什么没用呢，俺也没有明白。
- OHEM可以更好地分类tree/three这种难例，但对整体的精度有所损失，可能需要训练更多epoch。