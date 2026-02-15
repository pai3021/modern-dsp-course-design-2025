import numpy as np
import pywt
import scipy.stats
from scipy.stats import entropy

class FeatureExtractor:
    def __init__(self, fs=200):
        self.fs = fs

    def compute_rr_features(self, r_indices):
        """
        计算 RR 间期特征 (全局计算，避免逐个切片计算丢失上下文)
        :param r_indices: 整条数据的 R 波索引列表
        :return: (pre_rr, post_rr, local_avg_rr) 的数组
        """
        # 1. 计算所有相邻 R 波的距离 (单位: 采样点)
        rr_intervals = np.diff(r_indices)
        
        # 2. 为了对齐心拍，我们需要补齐长度
        # 第一个心拍没有 Pre-RR，最后一个没有 Post-RR
        # 策略：用相邻的值填充
        pre_rr = np.concatenate(([rr_intervals[0]], rr_intervals))
        post_rr = np.concatenate((rr_intervals, [rr_intervals[-1]]))
        
        # 3. 计算局部平均 RR (Local Average RR)
        # 这是一个滑动窗口平均，用来适应心率变异。如果当前 RR 远小于局部平均，就是早搏。
        # 使用卷积计算滑动平均，窗口大小取 10 个心拍
        window_size = 10
        local_avg = np.convolve(pre_rr, np.ones(window_size)/window_size, mode='same')
        
        return pre_rr, post_rr, local_avg

    def compute_wavelet_features(self, segment):
        """
        利用小波变换提取频域/时频特征
        要求：满足作业中的 '小波变换' 要求
        """
        # 使用 Daubechies 4 小波进行 4 层分解
        # cA4: 低频近似 (P波, T波轮廓)
        # cD4, cD3: 中频细节 (QRS 波主要能量)
        # cD2, cD1: 高频噪声
        coeffs = pywt.wavedec(segment, 'db4', level=4)
        cA4, cD4, cD3, cD2, cD1 = coeffs
        
        # 提取能量特征 (数值的平方和)
        # 某些异常心拍(如室早)低频能量会显著变大
        feat_energy_low = np.sum(cA4 ** 2)
        feat_energy_mid = np.sum(cD4 ** 2) + np.sum(cD3 ** 2)
        feat_energy_high = np.sum(cD2 ** 2) + np.sum(cD1 ** 2)
        
        return [feat_energy_low, feat_energy_mid, feat_energy_high]

    def compute_morph_features(self, segment):
        """
        提取统计与形态特征 (含熵)
        要求：满足作业中的 '熵' 要求
        """
        # 1. 峰峰值 (信号振幅范围)
        p_p = np.max(segment) - np.min(segment)
        
        # 2. 偏度 (Skewness) - 衡量波形的不对称性
        skew = scipy.stats.skew(segment)
        
        # 3. 峭度 (Kurtosis) - 衡量波形的尖锐程度 (QRS越尖，峭度越高)
        kurt = scipy.stats.kurtosis(segment)
        
        # 4. 香农熵 (Shannon Entropy) - 衡量分布的复杂度
        # 先将信号归一化为概率分布
        hist_counts, _ = np.histogram(segment, bins=10, density=True)
        # 加上微小值避免 log(0)
        hist_counts += 1e-10 
        ent = entropy(hist_counts)
        
        return [p_p, skew, kurt, ent]

    def extract_batch(self, segments, r_indices):
        """
        批量提取所有特征，返回特征矩阵
        """
        num_beats = len(segments)
        features = []
        
        # 1. 获取 RR 特征
        pre_rr, post_rr, local_avg = self.compute_rr_features(r_indices)
        
        print(f"开始提取特征，共 {num_beats} 个心拍...")
        
        for i in range(num_beats):
            seg = segments[i]
            
            # --- RR 比率 (无量纲化) ---
            # 如果 ratio < 0.8 甚至 0.6，极大概率是早搏
            rr_ratio_pre = pre_rr[i] / (local_avg[i] + 1e-5)
            rr_ratio_post = post_rr[i] / (local_avg[i] + 1e-5)
            
            # --- 小波特征 ---
            wt_feats = self.compute_wavelet_features(seg)
            
            # --- 形态/熵特征 ---
            morph_feats = self.compute_morph_features(seg)
            
            # 合并所有特征向量
            # 特征向量结构: [RR_Ratio_Pre, RR_Ratio_Post, Low_Energy, Mid_Energy, High_Energy, P-P, Skew, Kurt, Entropy]
            row = [rr_ratio_pre, rr_ratio_post] + wt_feats + morph_feats
            features.append(row)
            
            if i % 10000 == 0 and i > 0:
                print(f"已处理 {i}/{num_beats}...")
                
        return np.array(features)


# ============================================================================
# 特征缓存功能 - 避免LOOCV重复提取特征
# ============================================================================

import os
import pickle
import hashlib
import config
import data_loader

def get_cache_path(filename, cache_dir='feature_cache'):
    """生成缓存文件路径"""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{filename}_features.pkl")

def compute_feature_hash():
    """计算特征提取代码的哈希值，用于检测代码变更"""
    # 简化版：使用窗口大小和采样率作为版本标识
    version_str = f"v1_window_{config.WINDOW_LEFT}_{config.WINDOW_RIGHT}_fs_{config.FS}"
    return hashlib.md5(version_str.encode()).hexdigest()[:8]

def save_features_to_cache(filename, X, y, r_indices, valid_beat_indices):
    """
    保存提取的特征到缓存
    
    参数:
        filename: 文件名
        X: 特征矩阵 (N, 9)
        y: 标签数组 (N,)
        r_indices: 原始R波索引
        valid_beat_indices: 有效心拍索引
    """
    cache_path = get_cache_path(filename)
    cache_data = {
        'X': X,
        'y': y,
        'r_indices': r_indices,
        'valid_beat_indices': valid_beat_indices,
        'version': compute_feature_hash()
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"[缓存] 特征已保存: {cache_path}")

def load_features_from_cache(filename):
    """
    从缓存加载特征
    
    返回:
        (X, y, r_indices, valid_beat_indices) 或 None (缓存不存在/过期)
    """
    cache_path = get_cache_path(filename)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 检查版本
        if cache_data.get('version') != compute_feature_hash():
            print(f"[缓存] 版本不匹配，需要重新提取: {filename}")
            return None
        
        print(f"[缓存] 特征已加载: {cache_path}")
        return (
            cache_data['X'],
            cache_data['y'],
            cache_data['r_indices'],
            cache_data['valid_beat_indices']
        )
    except Exception as e:
        print(f"[缓存] 加载失败: {e}")
        return None

def extract_features_with_cache(filename, use_cache=True):
    """
    提取特征（支持缓存）
    
    参数:
        filename: 文件名
        use_cache: 是否使用缓存
    
    返回:
        (X, y, r_indices, valid_beat_indices)
    """
    # 尝试加载缓存
    if use_cache:
        cached = load_features_from_cache(filename)
        if cached is not None:
            return cached
    
    # 缓存未命中，重新提取
    print(f"[提取] 正在提取特征: {filename}")
    
    # 加载数据
    signal = data_loader.load_ecg_signal(filename)
    df = data_loader.load_annotations(filename)
    r_indices = df['R_Index'].values
    
    # 切片（训练集不使用padding）
    segments, valid_beat_indices = data_loader.get_heartbeat_segments(
        signal, r_indices, use_padding=False
    )
    
    # 提取特征
    extractor = FeatureExtractor(fs=config.FS)
    actual_r_positions = r_indices[valid_beat_indices]
    X = extractor.extract_batch(segments, actual_r_positions)
    
    # 获取标签
    y = df['Beat Symbol'].values[valid_beat_indices]
    
    # 保存到缓存
    if use_cache:
        save_features_to_cache(filename, X, y, r_indices, valid_beat_indices)
    
    return X, y, r_indices, valid_beat_indices