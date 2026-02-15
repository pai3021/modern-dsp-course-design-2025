import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt
import config

def butter_bandpass_filter(signal, lowcut=0.5, highcut=40, fs=200, order=4):
    """
    巴特沃斯带通滤波器
    :param signal: 输入信号
    :param lowcut: 低频截止频率 (Hz)
    :param highcut: 高频截止频率 (Hz)
    :param fs: 采样率
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)  # 零相位滤波
    return filtered_signal

def load_ecg_signal(filename, apply_filter=True):
    """
    读取 .mat 文件中的 ECG 信号，并应用滤波
    返回: 单导联信号 (1D array)
    """
    mat_path = os.path.join(config.DATA_PATH, f"{filename}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"找不到文件: {mat_path}")
    
    data = sio.loadmat(mat_path)
    # ECG_Data 形状通常是 (12, N) 或 (N, 12)，根据 Source 14 是 12xN
    # 我们暂时取第2个通道 (索引1)，通常是 II 导联，信号较清晰
    ecg_signal = data['ECG_Data'][1, :]
    
    # 应用带通滤波去除基线漂移和高频噪声
    if apply_filter:
        ecg_signal = butter_bandpass_filter(ecg_signal, lowcut=0.5, highcut=40, fs=config.FS)
    
    return ecg_signal

def load_annotations(filename):
    """
    读取 .csv 标注文件
    返回: DataFrame (包含 Time Offset 和 Beat Symbol)
    """
    csv_path = os.path.join(config.DATA_PATH, f"{filename}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"警告: {filename} 没有找到标注文件")
        return None

    # 清洗数据：只要 Time Offset 和 Beat Symbol
    # 确保列名没有空格干扰
    df.columns = [c.strip() for c in df.columns]
    
    # Time Offset 是 ms，我们需要转为 采样点索引
    # Index = (Time_ms / 1000) * Fs
    df['R_Index'] = (df['Time Offset(ms)'] / 1000 * config.FS).astype(int)
    
    # 处理标签：如果是测试集，没有 Beat Symbol 列
    if 'Beat Symbol' not in df.columns:
        df['Beat Symbol'] = '?' # 占位符
        
    return df[['R_Index', 'Beat Symbol']]

def get_heartbeat_segments(ecg_signal, r_indices, use_padding=False):
    """
    心拍切片：根据 R 波位置切出一段波形
    
    参数:
        ecg_signal: ECG信号数组
        r_indices: R波位置索引
        use_padding: 是否使用padding保证全覆盖（测试集必须True）
    
    返回: 
        segments (N_beats, Window_Size) - 切片数组
        valid_beat_indices (N_beats,) - 有效心拍在原始列表中的索引位置（用于对齐标签）
        is_boundary (N_beats,) - 是否为边界心拍（仅use_padding=True时返回）
    """
    segments = []
    valid_beat_indices = []  # 记录有效心拍在原始r_indices中的索引位置
    is_boundary = []  # 标记边界心拍
    
    for i, r in enumerate(r_indices):
        start = r - config.WINDOW_LEFT
        end = r + config.WINDOW_RIGHT
        
        # 边界检查
        if start >= 0 and end < len(ecg_signal):
            # 正常情况：完整窗口
            seg = ecg_signal[start:end]
            segments.append(seg)
            valid_beat_indices.append(i)
            is_boundary.append(False)
        elif use_padding:
            # 测试集模式：使用padding保证全覆盖
            seg = np.zeros(config.WINDOW_LEFT + config.WINDOW_RIGHT)
            
            # 计算有效范围
            valid_start = max(0, start)
            valid_end = min(len(ecg_signal), end)
            
            # 计算在seg中的偏移
            offset_in_seg = valid_start - start
            length = valid_end - valid_start
            
            # 填充有效数据
            seg[offset_in_seg:offset_in_seg+length] = ecg_signal[valid_start:valid_end]
            
            segments.append(seg)
            valid_beat_indices.append(i)
            is_boundary.append(True)  # 标记为边界
    
    if use_padding:
        return np.array(segments), np.array(valid_beat_indices), np.array(is_boundary)
    else:
        return np.array(segments), np.array(valid_beat_indices)

