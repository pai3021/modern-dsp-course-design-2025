# 异常心拍位置智能识别 - 课程设计项目

## 📋 项目简介

本项目是2025年现代信号处理理论及应用课程设计作业，主要任务是基于ECG信号进行异常心拍的智能识别与定位。

**主要功能：**
- 自动识别ECG信号中的异常心拍（室性早搏等）
- 精确定位异常心拍的时间位置
- 基于小波变换、RR间期、熵等多维特征的综合分析
- F1-Score优化的分类算法

## 🎯 核心技术

### 信号处理
- **带通滤波**：Butterworth滤波器 (0.5-40Hz) 去除基线漂移和高频噪声
- **小波变换**：Daubechies-4小波进行4层分解，提取多尺度频域特征
- **R波检测**：基于QRS波形特征的R波定位算法

### 特征提取
1. **RR间期特征**：Pre-RR、Post-RR、局部平均RR及其比率
2. **小波能量特征**：低频、中频、高频能量分布
3. **形态特征**：峰峰值、偏度、峭度
4. **信息熵**：Shannon熵评估波形复杂度

### 分类算法
- **F1优化分类器**：针对不平衡样本优化的规则分类器
- **自适应阈值**：基于训练集统计特性自动确定分类阈值
- **记录级归一化**：抗跨记录幅值漂移

## 📁 项目结构

```
modern-dsp-course-design-2025/
│
├── config.py                    # 全局配置参数
├── data_loader.py               # 数据加载与预处理
├── features.py                  # 特征提取模块
├── algorithm_f1_optimized.py    # F1优化分类算法
├── train.py                     # 模型训练与交叉验证
├── predict.py                   # 测试集预测
├── preprocess_features.py       # 特征预处理与缓存
│
├── dataset/                     # 原始ECG数据 (.mat & .csv)
├── feature_cache/               # 特征缓存文件
├── models/                      # 训练好的模型
├── output/                      # 预测结果输出 (.mat)
│
└── src/                         # 辅助脚本与分析工具
    ├── check_data.py            # 数据可视化检查
    ├── check_features.py        # 特征分布分析
    ├── check_all_alignment.py   # 数据对齐检查
    ├── check_test_boundary_drops.py  # 边界心拍检查
    ├── compare_algorithms.py    # 算法性能对比
    └── ReadData.py              # 数据读取工具
```

## 🚀 快速开始（完整复现步骤）

### 第一步：环境配置

**1. 创建Conda环境**
```bash
conda create -n hw python=3.10
conda activate hw
```

**2. 安装依赖包**
```bash
cd E:\MS_code\modern-dsp-course-design-2025
pip install -r requirement.txt
```

**验证安装：**
```bash
python -c "import numpy, scipy, pywt; print('环境配置成功!')"
```

### 第二步：数据准备

**1. 数据文件结构**

确保 `dataset/` 目录包含以下文件：

```
dataset/
├── H2503272XE92U.mat & H2503272XE92U.csv  (训练集)
├── H2511143S7N68.mat & H2511143S7N68.csv  (训练集)
├── H2511272H6BIX.mat & H2511272H6BIX.csv  (训练集)
├── H2512032UWO3B.mat & H2512032UWO3B.csv  (训练集)
├── H2512043EY485.mat & H2512043EY485.csv  (训练集)
├── H2512050H8V3I.mat & H2512050H8V3I.csv  (训练集)
├── H25113017U411.mat & H25113017U411.csv  (测试集)
├── H251204210MS0.mat & H251204210MS0.csv  (测试集)
├── H2512053F662E.mat & H2512053F662E.csv  (测试集)
└── H2512082SFFUB.mat & H2512082SFFUB.csv  (测试集)
```

**2. 修改配置文件**

编辑 [config.py](config.py#L5)，修改数据路径为你的实际路径：

```python
# 修改前
DATA_PATH = r"I:/Pai/课程作业/课程作业"

# 修改后（示例）
DATA_PATH = r"E:\MS_code\modern-dsp-course-design-2025\dataset"
```

**3. 验证数据加载**
```bash
python -c "import config; import data_loader; signal = data_loader.load_ecg_signal('H2503272XE92U'); print(f'数据加载成功! 信号长度: {len(signal)}')"
```

### 第三步：特征预处理

**提取并缓存所有训练文件的特征，加速后续训练：**

```bash
python preprocess_features.py
```

预期输出：
```
==================================================
特征预处理 - 提取并缓存所有训练文件的特征
==================================================
[1/6] 处理: H2503272XE92U
  ✅ 完成: 5234 个心拍, 用时 15.3s
[2/6] 处理: H2511143S7N68
  ✅ 完成: 6891 个心拍, 用时 18.7s
...
✅ 所有特征已缓存，总用时: 2.5 分钟
```


### 第四步：模型训练

**运行训练脚本：**

```bash
python train.py
```

**训练过程：**
- 采用Leave-One-Out交叉验证（6折）
- 每轮使用5个文件训练，1个文件验证
- 自动保存最佳模型到 `models/model_latest.pkl`
- 同时保存带时间戳的版本（如 `model_20250216_143052.pkl`）

**预期输出示例：**
```
=== 第 1/6 轮验证 ===
验证集: H2503272XE92U
训练集: H2511143S7N68, H2511272H6BIX, ...
  [结果] Acc: 0.9532 | Precision: 0.8912 | Recall: 0.8234 | F1: 0.8560
  [详情] TP:145  TN:4867  FP:34  FN:22

=== 第 2/6 轮验证 ===
...

✅ 交叉验证完成
平均 F1-Score: 0.8523
最佳模型已保存: models/model_latest.pkl
```


### 第五步：测试集预测

**运行预测脚本：**

```bash
python predict.py
```

**预测流程：**
1. 自动加载 `models/model_latest.pkl`
2. 对4个测试文件逐一预测
3. 结果保存为 `.mat` 格式到 `output/` 目录

**预期输出：**
```
可用模型 (3 个):
→ 1. model_latest.pkl (125.3 KB, 2025-02-16 14:30)
  2. model_20250216_143052.pkl (125.3 KB, 2025-02-16 14:30)
  ...

[1/4] 正在预测: H25113017U411
  ✅ 预测完成: 5123 个心拍, 异常: 245 个 (4.8%)
  保存至: output/H25113017U411.mat

[2/4] 正在预测: H251204210MS0
  ...

✅ 所有测试集预测完成!
```

### 第六步：查看结果

**输出文件位置：**
```
output/
├── H25113017U411.mat
├── H251204210MS0.mat
├── H2512053F662E.mat
└── H2512082SFFUB.mat
```
---
## 🔍 数据说明

### 训练集（带标签）
- H2503272XE92U
- H2511143S7N68
- H2511272H6BIX
- H2512032UWO3B
- H2512043EY485
- H2512050H8V3I

### 测试集（无标签）
- H25113017U411
- H251204210MS0
- H2512053F662E
- H2512082SFFUB

### 标签说明
- **N (Normal)**：正常心拍
- **X (Abnormal)**：异常心拍（室性早搏等）

## 💡 关键参数

在 `config.py` 中可调整以下参数：

```python
FS = 200              # 采样率 (Hz)
WINDOW_LEFT = 50      # R波左侧窗口 (0.25秒)
WINDOW_RIGHT = 90     # R波右侧窗口 (0.45秒)
```

## 🛠️ 辅助工具

### 数据可视化
```bash
python src/check_data.py          # 查看ECG波形与标注
python src/check_features.py      # 分析特征分布
```

### 算法对比
```bash
python src/compare_algorithms.py  # 对比不同算法版本性能
```

### 边界检查
```bash
python src/check_test_boundary_drops.py  # 检查测试集边界心拍处理
```


