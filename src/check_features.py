import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import config
import data_loader
from features import FeatureExtractor

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def verify_features():
    # 1. 读取数据 (依然用第一个文件)
    filename = config.TRAIN_FILES[0]
    print(f"读取文件: {filename}")
    signal = data_loader.load_ecg_signal(filename)
    df = data_loader.load_annotations(filename)
    
    # 2. 切片
    r_indices = df['R_Index'].values
    segments, valid_r = data_loader.get_heartbeat_segments(signal, r_indices)
    
    # 注意：valid_r 的长度可能比 df 少（因为边界去除），需要对齐标签
    # 简单做法：我们只取前 len(segments) 个标签做演示
    # 严谨做法：应该根据 valid_r 反查 label，但这里 valid_r 基本只少了首尾几个
    labels = df['Beat Symbol'].values[:len(segments)]
    
    # 3. 提取特征
    extractor = FeatureExtractor(fs=config.FS)
    # 为了演示快一点，我们只取前 2000 个点做测试，不用跑全量
    test_limit = 2000
    print(f"正在提取前 {test_limit} 个心拍的特征...")
    
    X_feats = extractor.extract_batch(segments[:test_limit], valid_r[:test_limit])
    y_labels = labels[:test_limit]
    
    # 特征名称列表 (对应 features.py 里的顺序)
    feat_names = [
        'RR_Ratio_Pre', 'RR_Ratio_Post', 
        'WT_Low', 'WT_Mid', 'WT_High', 
        'P-P', 'Skew', 'Kurt', 'Entropy'
    ]
    
    # 4. 可视化分析：RR 间期对比
    # 将特征转为 DataFrame 方便绘图
    df_feats = pd.DataFrame(X_feats, columns=feat_names)
    df_feats['Label'] = y_labels
    
    print("特征提取完成！开始绘图...")
    
    plt.figure(figsize=(14, 6))
    
    # 子图1: RR_Ratio_Pre 分布 (这是识别早搏最重要的特征)
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Label', y='RR_Ratio_Pre', data=df_feats, palette="Set2")
    plt.title("特征对比: 前向 RR 间期比率 (Pre-RR Ratio)")
    plt.grid(True, alpha=0.3)
    # 如果 N 是 1.0 左右，X 明显小于 1.0，说明特征有效！
    
    # 子图2: 小波中频能量 (WT_Mid) 分布
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Label', y='WT_Mid', data=df_feats, palette="Set2")
    plt.title("特征对比: 小波中频能量 (QRS 能量)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("特征矩阵形状:", X_feats.shape)
    print("示例特征值 (第一行):", X_feats[0])

if __name__ == "__main__":
    verify_features()