# 脑肿瘤 MRI 四分类：ResNet50 迁移学习

Brain tumor MRI 4-class classification with ResNet50 transfer learning (PyTorch).

以 ImageNet 预训练的 ResNet50 为基础，冻结浅层、只微调 layer4 与自定义分类头，并与 VGG16、EfficientNet-B0 做对比，同时用消融实验量化微调与 Dropout 各自的贡献。

## 数据集

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)（Kaggle，作者 Masoud Nickparvar），由 figshare、SARTAJ、Br35H 三个公开数据集整合而成，共 7,200 张人脑 MRI 图像，四个类别完全均衡：

| 类别 | 训练集 | 测试集 |
| --- | ---: | ---: |
| Glioma（胶质瘤） | 1,400 | 400 |
| Meningioma（脑膜瘤） | 1,400 | 400 |
| Pituitary（垂体瘤） | 1,400 | 400 |
| No tumor（无肿瘤） | 1,400 | 400 |
| **合计** | **5,600** | **1,600** |

沿用数据集官方划分：`Training/` 作为训练集，`Testing/` 作为独立测试集。训练集再按 70/30 随机划分为训练子集（3,920 张）与验证子集（1,680 张）。

预处理：所有图像统一缩放至 224×224；训练集使用随机水平翻转（p=0.5），验证集与测试集不做随机增强；全部按 ImageNet 的均值与标准差归一化。

## 方法

- 主干网络：`resnet50(weights=IMAGENET1K_V1)`，冻结 layer1–layer3，仅微调 layer4
- 分类头：`Dropout(0.6) → Linear(2048, 128) → ReLU → Dropout(0.5) → Linear(128, 4)`
- 可训练参数 15,227,524 / 23,770,820（64.1%）
- 损失函数：CrossEntropyLoss
- 优化器：Adam，`lr=1e-4`，`weight_decay=0.05`
- 学习率调度：`ReduceLROnPlateau(mode='max', factor=0.5, patience=3, min_lr=1e-7)`
- 早停：patience=5，共训练 10 个 epoch
- 运行环境：Kaggle Notebook，Tesla T4 ×2，16GB 内存

## 结果

ResNet50 在 1,600 张独立测试集上的混淆矩阵：

| 真实类别 \ 预测类别 | pituitary | notumor | meningioma | glioma |
| --- | ---: | ---: | ---: | ---: |
| pituitary | **394** | 0 | 4 | 2 |
| notumor | 0 | **400** | 0 | 0 |
| meningioma | 4 | 3 | **388** | 5 |
| glioma | 2 | 29 | 54 | **315** |

**测试集准确率 93.56%**（1,497 / 1,600），训练过程中最高验证准确率为 97.86%。

由混淆矩阵计算的各类指标：

| 类别 | 精确率 | 召回率 | F1 |
| --- | ---: | ---: | ---: |
| pituitary | 98.50% | 98.50% | 98.50% |
| notumor | 92.59% | 100.00% | 96.15% |
| meningioma | 87.00% | 97.00% | 91.72% |
| glioma | 97.83% | 78.75% | 87.26% |
| macro avg | 93.98% | 93.56% | 93.41% |

三个模型在同一训练配置下的测试集对比：

| 模型 | 微调策略 | 测试准确率 |
| --- | --- | ---: |
| ResNet50 | 微调 layer4 + 自定义分类头 | 93.56% |
| VGG16 | 仅训练分类头 | 78.81% |
| EfficientNet-B0 | 仅训练分类头 | 80.75% |

## 消融实验

以完整模型（ResNet50 + 微调 layer4 + Dropout 0.6/0.5）为基线，在相同的验证集上评估，指标为训练过程中的最高验证准确率：

| 变体 | 最高验证准确率 | 相对变化 |
| --- | ---: | ---: |
| 完整模型 | 97.56% | — |
| 无 Dropout | 96.37% | −1.19 |
| 无微调（完全冻结卷积层，仅训练分类头） | 87.80% | −9.76 |

结论：让预训练模型适配目标域的数据分布（微调 layer4）是主要增益来源，贡献 +9.76 个百分点；Dropout 起进一步的稳定与正则作用，贡献 +1.19 个百分点。

## 误差分析

模型对"有无肿瘤"的判断非常可靠，notumor 类别 400 张全部正确。误差主要来自 glioma：召回率 78.75%，其中 54 张被误判为 meningioma、29 张误判为 notumor。胶质瘤与脑膜瘤在 MRI 上的纹理与边界特征相近，是最主要的混淆来源。

## 局限与改进方向

- 两个基线（VGG16、EfficientNet-B0）只训练了各自新接的分类头，主干全部冻结，而 ResNet50 微调了 layer4，可训练参数量相差一个量级。因此上表的对比反映的是"在当前这套微调配置下"的差异，不能直接当作网络结构优劣的结论。更严格的对比需要让三个模型的可训练参数量接近。
- 仅使用单一随机种子（42），未做多次实验取均值。
- 各模型统一训练 10 个 epoch，未针对每个模型单独调参。

## 文件

- `main.ipynb`：数据处理、模型构建、训练、对比实验与消融实验的完整代码，含运行输出

## 依赖

```
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
pillow
opencv-python
tqdm
```

## 运行

代码按 Kaggle 数据集路径（`/kaggle/input/...`）编写，在 Kaggle Notebook 上可直接运行。本地运行需要修改 `base_directory`，指向本地数据集目录。
