"""
F1优化版规则分类器 - 专门针对F1分数优化
核心思路：平衡Precision和Recall，减少误报同时保持高召回率
"""
import numpy as np

class F1OptimizedClassifier:
    """
    F1优化分类器
    - 使用更严格的多重条件判断
    - 引入置信度评分机制
    - 针对训练数据的异常特征分布调优
    """
    def __init__(self):
        self.thresholds = {}
        self.stat_params = {}
        self.abnormal_patterns = {}
        
    def fit(self, X_train, y_train):
        """训练：学习正常和异常的特征分布"""
        # 分离正常和异常样本
        n_indices = [i for i, label in enumerate(y_train) if label == 'N']
        x_indices = [i for i, label in enumerate(y_train) if label == 'X']
        
        X_normal = X_train[n_indices]
        X_abnormal = X_train[x_indices] if len(x_indices) > 0 else None
        
        print(f"[F1优化训练] 正常:{len(X_normal)}, 异常:{len(x_indices)}")
        
        # === 1. RR间期阈值（核心特征）===
        # 根据训练数据的异常样本统计，异常RR均值≈0.75
        # 正常RR均值≈1.0，标准差≈0.06
        
        # 策略：使用更精确的阈值，避免误判
        # 下界：设在正常均值-2.5倍标准差
        normal_rr_mean = np.mean(X_normal[:, 0])
        normal_rr_std = np.std(X_normal[:, 0])
        
        self.thresholds['rr_pre_min'] = normal_rr_mean - 2.5 * normal_rr_std
        self.thresholds['rr_pre_max'] = normal_rr_mean + 2.5 * normal_rr_std
        
        # 如果有异常样本，学习其分布
        if X_abnormal is not None and len(X_abnormal) > 10:
            abnormal_rr_mean = np.mean(X_abnormal[:, 0])
            abnormal_rr_std = np.std(X_abnormal[:, 0])
            
            # 异常的RR通常在0.5-0.9之间
            # 设置一个更保守的下界：正常-异常之间的中点
            midpoint = (normal_rr_mean + abnormal_rr_mean) / 2
            self.thresholds['rr_pre_min'] = min(self.thresholds['rr_pre_min'], midpoint)
            
            self.abnormal_patterns['rr_mean'] = abnormal_rr_mean
            self.abnormal_patterns['rr_std'] = abnormal_rr_std
            
            print(f"[异常模式] RR均值={abnormal_rr_mean:.3f}±{abnormal_rr_std:.3f}")
        
        print(f"[F1阈值] RR下界={self.thresholds['rr_pre_min']:.3f}, 上界={self.thresholds['rr_pre_max']:.3f}")
        
        # === 2. 记录所有特征的统计量 ===
        feature_names = ['rr_pre', 'rr_post', 'wt_low', 'wt_mid', 'wt_high', 
                        'peak_peak', 'skew', 'kurt', 'entropy']
        
        for idx, name in enumerate(feature_names):
            mean_n = np.mean(X_normal[:, idx])
            std_n = np.std(X_normal[:, idx])
            
            self.stat_params[f'{name}_mean'] = mean_n
            self.stat_params[f'{name}_std'] = std_n
            
            # 记录异常样本的统计量
            if X_abnormal is not None and len(X_abnormal) > 0:
                mean_x = np.mean(X_abnormal[:, idx])
                std_x = np.std(X_abnormal[:, idx])
                self.abnormal_patterns[f'{name}_mean'] = mean_x
                self.abnormal_patterns[f'{name}_std'] = std_x
    
    def predict(self, X_test):
        """
        F1优化的判决策略：
        - 使用多重证据组合
        - 提高判决阈值，减少误报
        - 保证真正的异常能被识别
        """
        predictions = []
        
        for row in X_test:
            # 初始化证据得分
            evidence_score = 0
            confidence = 0.0
            
            # === 证据1: RR间期异常（权重最高）===
            rr_pre = row[0]
            rr_post = row[1]
            
            # 强异常：RR显著低于阈值
            if rr_pre < self.thresholds['rr_pre_min']:
                deviation = (self.thresholds['rr_pre_min'] - rr_pre) / self.stat_params['rr_pre_std']
                if deviation > 2:  # 偏离超过2个标准差
                    evidence_score += 4  # 强证据
                    confidence += 0.8
                else:
                    evidence_score += 2  # 中等证据
                    confidence += 0.5
            
            # 异常：RR显著高于阈值（代偿间期）
            elif rr_pre > self.thresholds['rr_pre_max']:
                evidence_score += 2
                confidence += 0.4
            
            # 后续RR异常
            if rr_post < self.thresholds['rr_pre_min'] * 0.95:
                evidence_score += 1
                confidence += 0.3
            
            # === 证据2: 小波能量异常（中等权重）===
            # 异常心拍通常低频能量显著增加
            wt_low = row[2]
            wt_mid = row[3]
            
            z_low = abs(wt_low - self.stat_params['wt_low_mean']) / (self.stat_params['wt_low_std'] + 1e-6)
            z_mid = abs(wt_mid - self.stat_params['wt_mid_mean']) / (self.stat_params['wt_mid_std'] + 1e-6)
            
            if z_low > 3:  # 低频能量异常
                evidence_score += 2
                confidence += 0.4
            elif z_low > 2:
                evidence_score += 1
                confidence += 0.2
            
            if z_mid > 3:  # 中频能量异常
                evidence_score += 1
                confidence += 0.3
            
            # === 证据3: 熵异常（信号复杂度）===
            entropy = row[8]
            z_ent = (entropy - self.stat_params['entropy_mean']) / (self.stat_params['entropy_std'] + 1e-6)
            
            if z_ent > 2:  # 熵显著增加
                evidence_score += 1
                confidence += 0.3
            
            # === 证据4: 峭度异常（波形形状）===
            kurt = row[7]
            z_kurt = abs(kurt - self.stat_params['kurt_mean']) / (self.stat_params['kurt_std'] + 1e-6)
            
            if z_kurt > 3:
                evidence_score += 1
                confidence += 0.2
            
            # === 综合判决（F1优化策略）===
            # 策略1: 强证据直接判定（Recall保证）
            if evidence_score >= 4:
                predictions.append('X')
            
            # 策略2: 中等证据+高置信度（平衡）
            elif evidence_score >= 3 and confidence >= 0.8:
                predictions.append('X')
            
            # 策略3: 弱证据但RR严重异常（关键特征）
            elif evidence_score >= 2 and rr_pre < self.thresholds['rr_pre_min'] * 0.9:
                predictions.append('X')
            
            # 否则判为正常
            else:
                predictions.append('N')
        
        return np.array(predictions)


class BalancedClassifier:
    """
    平衡分类器 - 针对不同文件自适应调整
    结合全局模型和局部统计
    """
    def __init__(self):
        self.global_model = F1OptimizedClassifier()
        
    def fit(self, X_train, y_train):
        """训练全局模型"""
        self.global_model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """预测，可以根据数据特征微调"""
        # 使用F1优化模型
        predictions = self.global_model.predict(X_test)
        
        # 可选：后处理优化
        # 例如：平滑化，去除孤立异常等
        predictions = self._smooth_predictions(predictions, X_test)
        
        return predictions
    
    def _smooth_predictions(self, predictions, X_test, window=5):
        """
        平滑预测结果：
        - 如果一个X周围都是N，可能是误报
        - 如果一个N周围都是X，可能是漏检
        """
        smoothed = predictions.copy()
        n = len(predictions)
        
        for i in range(n):
            # 获取窗口
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            window_preds = predictions[start:end]
            
            # 统计窗口内的X比例
            x_ratio = sum(window_preds == 'X') / len(window_preds)
            
            # 如果是孤立的X，可能是误报（提高Precision）
            if predictions[i] == 'X' and x_ratio < 0.2:
                # 检查RR特征，如果不是强异常就改为N
                if X_test[i, 0] > 0.85:  # RR不是很低
                    smoothed[i] = 'N'
            
            # 如果是孤立的N在X群中，可能是漏检（提高Recall）
            elif predictions[i] == 'N' and x_ratio > 0.8:
                smoothed[i] = 'X'
        
        return smoothed
