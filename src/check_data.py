import matplotlib.pyplot as plt
import config
import data_loader
import numpy as np

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def check_first_file():
    # 取第一个训练文件
    filename = config.TRAIN_FILES[0]
    print(f"正在读取: {filename}...")
    
    # 1. 加载信号
    signal = data_loader.load_ecg_signal(filename)
    print(f"信号长度: {len(signal)} 点")
    
    # 2. 加载标签
    df = data_loader.load_annotations(filename)
    if df is not None:
        print(f"标注数量: {len(df)}")
        print("前5个标注:\n", df.head())
        
        # 3. 切片测试
        r_indices = df['R_Index'].values
        segments, valid_r = data_loader.get_heartbeat_segments(signal, r_indices)
        print(f"切片形状: {segments.shape}") # 应该是 (数量, 130)
        
        # 4. 可视化检查：画出前 3 个正常(N) 和 3 个异常(X) 的心拍
        plt.figure(figsize=(12, 6))
        
        # 找 N 和 X 的索引
        n_indices = df[df['Beat Symbol'] == 'N'].index[:3]
        x_indices = df[df['Beat Symbol'] == 'X'].index[:3]
        
        # 画 N (蓝色)
        for i, idx in enumerate(n_indices):
            plt.subplot(2, 3, i+1)
            # 注意：segments 的索引可能因为边界检查而少于 df，这里简化假设没越界
            plt.plot(segments[idx])
            plt.title(f"正常心拍 (N) - {idx}")
            plt.axvline(x=config.WINDOW_LEFT, color='r', linestyle='--', alpha=0.5) # 标记 R 波中心
            
        # 画 X (红色)
        for i, idx in enumerate(x_indices):
            plt.subplot(2, 3, i+4)
            plt.plot(segments[idx], color='orange')
            plt.title(f"异常心拍 (X) - {idx}")
            plt.axvline(x=config.WINDOW_LEFT, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    check_first_file()