"""
极端优化版算法 - 专门针对持续性心律失常（如H2511143S7N68）
"""
import numpy as np

class AggressiveClassifier:
    """
    激进分类器 - 极高敏感度
    适用于异常率极高的持续性心律失常病例
    """
    def __init__(self):
        self.thresholds = {}
        self.stat_params = {}
        
    def fit(self, X_train, y_train):
        """训练：统计正常心拍分布"""
        n_indices = [i for i, label in enumerate(y_train) if label == 'N']
        X_normal = X_train[n_indices]
        
        x_indices = [i for i, label in enumerate(y_train) if label == 'X']
        X_abnormal = X_train[x_indices] if len(x_indices) > 0 else None
        
        print(f"[训练] 正常:{len(X_normal)}, 异常:{len(x_indices)}")
        
        # === 关键：极端放宽RR阈值 ===
        # 从训练数据看，异常心拍的RR_Pre均值是0.7552
        # 但H2511143S7N68的RR可能在0.3-0.9之间波动
        # 我们需要用更低的阈值
        
        # 方案1: 基于异常样本的统计（如果有的话）
        if X_abnormal is not None and len(X_abnormal) > 10:
            abnormal_rr_mean = np.mean(X_abnormal[:, 0])
            abnormal_rr_std = np.std(X_abnormal[:, 0])
            
            # 设置阈值：异常均值 - 3*标准差（覆盖99.7%的异常样本）
            self.thresholds['rr_pre_min'] = max(0.3, abnormal_rr_mean - 3 * abnormal_rr_std)
            print(f"[激进模式] 基于异常样本，RR下限={self.thresholds['rr_pre_min']:.3f}")
        else:
            # 方案2: 使用固定的极低阈值
            self.thresholds['rr_pre_min'] = 0.5  # 非常激进
        
        # 上限保持宽松
        self.thresholds['rr_pre_max'] = np.percentile(X_normal[:, 0], 99)
        self.thresholds['rr_post_min'] = 0.5
        self.thresholds['rr_post_max'] = np.percentile(X_normal[:, 1], 99)
        
        print(f"[阈值] RR_Pre: [{self.thresholds['rr_pre_min']:.3f}, {self.thresholds['rr_pre_max']:.3f}]")
        
        # === 记录所有特征的统计量 ===
        for idx in range(X_train.shape[1]):
            mean_n = np.mean(X_normal[:, idx])
            std_n = np.std(X_normal[:, idx])
            self.stat_params[f'feat_{idx}'] = (mean_n, std_n, idx)
    
    def predict(self, X_test):
        """激进预测：只要有任何异常迹象就标记为X"""
        predictions = []
        
        for row in X_test:
            score = 0
            
            # === 规则1: RR间期（极宽松）===
            if row[0] < self.thresholds['rr_pre_min']:
                score += 5  # 超强信号
            elif row[0] < 0.95:  # 即使不到阈值，只要偏低就加分
                score += 2
            
            if row[0] > self.thresholds['rr_pre_max']:
                score += 3
            
            # Post-RR
            if row[1] < self.thresholds['rr_post_min'] or row[1] > self.thresholds['rr_post_max']:
                score += 2
            
            # === 规则2: 小波能量（放宽到2倍标准差）===
            for feat_name in ['feat_2', 'feat_3', 'feat_4']:  # wt_low, wt_mid, wt_high
                mean, std, idx = self.stat_params[feat_name]
                z = abs(row[idx] - mean) / (std + 1e-6)
                if z > 2:  # 原来是4-6，现在降到2
                    score += 1
            
            # === 规则3: 形态统计量 ===
            for feat_name in ['feat_5', 'feat_6', 'feat_7', 'feat_8']:
                mean, std, idx = self.stat_params[feat_name]
                z = abs(row[idx] - mean) / (std + 1e-6)
                if z > 2:
                    score += 1
            
            # === 判决：score >= 1 就算异常（极其宽松）===
            predictions.append('X' if score >= 1 else 'N')
        
        return np.array(predictions)


class InvertedLogicClassifier:
    """
    反转逻辑分类器 - 专门针对"异常为主"的文件
    思路：不是判断"什么是异常"，而是判断"什么是正常"
    只有明确是正常的才标N，其余全是X
    """
    def __init__(self):
        self.normal_profile = {}
        
    def fit(self, X_train, y_train):
        """学习正常心拍的严格特征"""
        n_indices = [i for i, label in enumerate(y_train) if label == 'N']
        X_normal = X_train[n_indices]
        
        print(f"[反转逻辑] 学习{len(X_normal)}个正常样本的严格特征")
        
        # 计算正常心拍的紧密边界（使用10%和90%分位点，非常严格）
        self.normal_profile = {}
        for idx in range(X_train.shape[1]):
            self.normal_profile[f'min_{idx}'] = np.percentile(X_normal[:, idx], 10)
            self.normal_profile[f'max_{idx}'] = np.percentile(X_normal[:, idx], 90)
            print(f"  特征{idx}: [{self.normal_profile[f'min_{idx}']:.3f}, "
                  f"{self.normal_profile[f'max_{idx}']:.3f}]")
    
    def predict(self, X_test):
        """只有完全符合正常特征的才标N，其余都是X"""
        predictions = []
        
        for row in X_test:
            is_normal = True
            
            # 检查关键特征（RR间期、小波能量、熵）
            critical_features = [0, 1, 2, 3, 8]  # RR_Pre, RR_Post, WT_Low, WT_Mid, Entropy
            
            for idx in critical_features:
                if row[idx] < self.normal_profile[f'min_{idx}'] or \
                   row[idx] > self.normal_profile[f'max_{idx}']:
                    is_normal = False
                    break
            
            predictions.append('N' if is_normal else 'X')
        
        return np.array(predictions)


class HybridClassifier:
    """
    混合分类器 - 根据文件特征自动选择策略
    """
    def __init__(self):
        self.aggressive = AggressiveClassifier()
        self.inverted = InvertedLogicClassifier()
        self.mode = 'aggressive'
        
    def fit(self, X_train, y_train):
        """训练两个分类器"""
        self.aggressive.fit(X_train, y_train)
        self.inverted.fit(X_train, y_train)
    
    def predict(self, X_test, estimated_abnormal_rate=None):
        """
        根据预估的异常率选择策略
        :param estimated_abnormal_rate: 预估异常率（0-1）
        """
        # 如果不知道异常率，先用激进模式试探
        if estimated_abnormal_rate is None:
            pred_aggressive = self.aggressive.predict(X_test)
            estimated_abnormal_rate = sum(pred_aggressive == 'X') / len(pred_aggressive)
            print(f"[混合模式] 预估异常率: {estimated_abnormal_rate:.2%}")
        
        # 如果异常率超过50%，使用反转逻辑
        if estimated_abnormal_rate > 0.5:
            print(f"[混合模式] 异常率>{50}%，启用反转逻辑")
            return self.inverted.predict(X_test)
        else:
            print(f"[混合模式] 异常率<{50}%，使用激进模式")
            return self.aggressive.predict(X_test)
