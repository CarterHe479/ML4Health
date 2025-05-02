## 🥗 项目简介

这个项目基于 **Nutrition5K Dataset**，旨在通过深度学习模型预测食品的营养成分（脂肪、碳水、蛋白质）以及卡路里含量。

🚀 **新增功能：**

* **跨模态 Cross-Attention 模型：**

  * 输入为 **RGB + side 图像拼接** 与 **Depth 图像**。
  * 使用 **ResNet18** 作为 backbone 提取特征。
  * 引入 **Cross-Attention 层** 以融合 RGB+side 与 Depth 的信息。
  * 最终通过 **MLP** 输出 4 个回归值（脂肪g、碳水g、蛋白质g、卡路里 kcal）。

---

## 🗂️ 目录结构

```
ML4Health/
├── cross_attention_model.py        # 新增 Cross-Attention 模型
├── data_loader.py                  # 修改: 返回 rgb+side 拼接图像 & depth & label
├── run_cross_attention_model.py    # 新增: 训练新模型的入口脚本
├── train.py                        # 训练框架 (Trainer 类)
├── loss.py                         # MSELoss, NutritionLoss 等
├── utils.py                        # 工具函数
├── requirements.txt
└── ... 其他文件
```

---

## 🖼️ 输入数据说明

* **RGB 图像**: 从 overhead 相机拍摄。
* **Side 图像**: 从 4 个侧面相机中选取（默认 `camera_A`）。
* **Depth 图像**: 使用 Depth-Anything-V2 生成的深度图。

拼接方式：

* `RGB + side`: 通道维度拼接，形成 `[6, H, W]` 输入。
* `Depth`: 单通道输入 `[1, H, W]`。

---

## 🏗️ 模型架构

1️⃣ **RGB+Side 分支:**

* ResNet18
* 修改 `conv1` 为 6 通道输入

2️⃣ **Depth 分支:**

* ResNet18
* 修改 `conv1` 为 1 通道输入

3️⃣ **Cross-Attention 层:**

* `query`: RGB+Side 提取的特征
* `key/value`: Depth 提取的特征
* `nn.MultiheadAttention` 实现

4️⃣ **MLP:**

* 输入: Attention 融合后的特征
* 输出: `[脂肪g, 碳水g, 蛋白质g, 卡路里 kcal]`

---

## 🛠️ 使用方法

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2️⃣ 数据准备

数据集结构示例：

```
nutrition5k_dataset/
├── imagery/
│   └── realsense_overhead/
│       └── {dish_id}/
│           ├── rgb.png
│           ├── depth_color.png
│           ├── camera_A.h264
│           └── camera_B.h264 ...
├── metadata/
│   ├── dish_metadata_cafe1.csv
│   └── dish_metadata_cafe2.csv
└── dataset_split.json
```

* **Depth 图像**：已通过 Depth-Anything-V2 预生成为 `depth_color.png`。
* **Side 图像**：代码从 `camera_A` 提取第一帧作为 side 图。

---

### 3️⃣ 训练 Cross-Attention 模型

运行：

```bash
python run_cross_attention_model.py
```

**配置参数在 `run_cross_attention_model.py` 中设置，包括：**

* 数据路径 (`root_dir`)
* 批量大小 (`batch_size`)
* 训练轮数 (`num_epochs`)
* 学习率等超参数

---

## ⚙️ 主要修改点

* `data_loader.py`:

  * 加载 `RGB`、`Depth`、4 个 `Side Cameras`。
  * 提取 `camera_A` 帧，与 `RGB` 拼接 → `[6, H, W]`。
  * 同时读取标签 `[脂肪g, 碳水g, 蛋白质g, 卡路里]`。

* `cross_attention_model.py`:

  * 实现 `CrossAttentionNutritionModel`。
  * 双 ResNet 分支 + Cross-Attention + MLP。

* `run_cross_attention_model.py`:

  * 新增的训练脚本，集成了数据加载 + 模型 + 训练流程。

---

## 🧪 评估

损失函数默认是 `MSELoss`，对 4 个指标计算均方误差，输出 `[B, 4]`。

---

## 💡 TODO & 可选扩展

* ✅ 增加测试/验证集推理脚本。
* ✅ 可尝试 `NutritionLoss`（支持可训练的 scale 权重）。
* ✅ 考虑使用全部 4 个 side 图像（而非只用 `camera_A`）。

---

## 🙌 致谢

数据集来源: [Nutrition5K Dataset](https://nutrition5k.stanford.edu/)

---

**作者:** Carter He 🚀

