"""
改进的规则分类器 - 增加多层判决逻辑和自适应阈值
"""
import numpy as np

class RuleBasedClassifier:
    def __init__(self, sensitivity='medium'):
        """
        :param sensitivity: 敏感度设置
            - 'high': 高敏感度（宁可误报，不漏检） - 适合医疗场景
            - 'medium': 中等敏感度（平衡精确率和召回率）
            - 'low': 低敏感度（减少误报）
        """
        self.thresholds = {}
        self.stat_params = {}
        self.sensitivity = sensitivity
        
        # 根据敏感度调整参数
        if sensitivity == 'high':
            self.rr_threshold = 0.88  # 更宽松（检测更多早搏）
            self.z_threshold = 4.0
        elif sensitivity == 'low':
            self.rr_threshold = 0.80  # 更严格
            self.z_threshold = 6.0
        else:  # medium
            self.rr_threshold = 0.85
            self.z_threshold = 5.0
        
    def fit(self, X_train, y_train):
        """
        '训练'过程：统计正常心拍的分布规律
        """
        # 筛选正常心拍
        n_indices = [i for i, label in enumerate(y_train) if label == 'N']
        X_normal = X_train[n_indices]
        
        # 同时筛选异常心拍，用于参考
        x_indices = [i for i, label in enumerate(y_train) if label == 'X']
        X_abnormal = X_train[x_indices] if len(x_indices) > 0 else None
        
        print(f"[训练] 正常样本: {len(X_normal)}, 异常样本: {len(x_indices)}")
        
        # === 特征1: RR间期 ===
        self.thresholds['rr_pre_min'] = max(
            np.percentile(X_normal[:, 0], 2),  # 2%分位点
            self.rr_threshold  # 保底阈值
        )
        self.thresholds['rr_pre_max'] = np.percentile(X_normal[:, 0], 98)
        
        self.thresholds['rr_post_min'] = max(
            np.percentile(X_normal[:, 1], 2),
            self.rr_threshold
        )
        self.thresholds['rr_post_max'] = np.percentile(X_normal[:, 1], 98)
        
        print(f"[阈值] RR_Pre: [{self.thresholds['rr_pre_min']:.3f}, {self.thresholds['rr_pre_max']:.3f}]")
        print(f"[阈值] RR_Post: [{self.thresholds['rr_post_min']:.3f}, {self.thresholds['rr_post_max']:.3f}]")
        
        # === 特征2-9: 形态特征 ===
        # 记录所有特征的均值和标准差
        feature_names = ['rr_pre', 'rr_post', 'wt_low', 'wt_mid', 'wt_high', 
                        'peak_peak', 'skew', 'kurt', 'entropy']
        
        for idx, name in enumerate(feature_names):
            mean_n = np.mean(X_normal[:, idx])
            std_n = np.std(X_normal[:, idx])
            
            # 如果有异常样本，也计算其统计量作为参考
            if X_abnormal is not None and len(X_abnormal) > 0:
                mean_x = np.mean(X_abnormal[:, idx])
                print(f"[特征{idx}] {name}: 正常={mean_n:.4f}±{std_n:.4f}, 异常={mean_x:.4f}")
            else:
                print(f"[特征{idx}] {name}: 正常={mean_n:.4f}±{std_n:.4f}")
            
            self.stat_params[name] = (mean_n, std_n, idx)
    
    def predict(self, X_test):
        """
        多层判决预测
        """
        predictions = []
        
        for row in X_test:
            score = 0  # 异常评分（越高越可能是异常）
            
            # === 第1层：RR间期检查（权重最高）===
            rr_pre = row[0]
            rr_post = row[1]
            
            # 早搏：RR间期显著缩短
            if rr_pre < self.thresholds['rr_pre_min']:
                score += 3  # 强异常信号
            
            # 代偿间期：RR间期显著延长
            if rr_pre > self.thresholds['rr_pre_max']:
                score += 2
            
            # 后续RR也异常
            if rr_post < self.thresholds['rr_post_min'] or rr_post > self.thresholds['rr_post_max']:
                score += 1
            
            # === 第2层：形态特征检查 ===
            # 只有当RR间期不是非常明确的异常时，才看形态
            if score < 3:
                # 小波能量异常
                mean_low, std_low, idx_low = self.stat_params['wt_low']
                mean_mid, std_mid, idx_mid = self.stat_params['wt_mid']
                
                z_low = abs(row[idx_low] - mean_low) / (std_low + 1e-6)
                z_mid = abs(row[idx_mid] - mean_mid) / (std_mid + 1e-6)
                
                if z_low > self.z_threshold:
                    score += 1
                if z_mid > self.z_threshold:
                    score += 1
                
                # 熵异常（信号过于混乱）
                mean_ent, std_ent, idx_ent = self.stat_params['entropy']
                z_ent = (row[idx_ent] - mean_ent) / (std_ent + 1e-6)
                
                if z_ent > 3:  # 熵过高
                    score += 1
                
                # 峭度异常（波形过于尖锐或平坦）
                mean_kurt, std_kurt, idx_kurt = self.stat_params['kurt']
                z_kurt = abs(row[idx_kurt] - mean_kurt) / (std_kurt + 1e-6)
                
                if z_kurt > self.z_threshold:
                    score += 1
            
            # === 判决 ===
            # score >= 2: 判为异常
            if score >= 2:
                predictions.append('X')
            else:
                predictions.append('N')
        
        return np.array(predictions)


class AdaptiveClassifier:
    """
    自适应分类器 - 根据每个病人的个体特征动态调整
    适用于个体差异大的场景
    """
    def __init__(self):
        self.global_params = {}
        
    def fit(self, X_train, y_train):
        """训练全局参数"""
        n_indices = [i for i, label in enumerate(y_train) if label == 'N']
        X_normal = X_train[n_indices]
        
        # 记录全局统计量
        self.global_params['rr_mean'] = np.mean(X_normal[:, 0])
        self.global_params['rr_std'] = np.std(X_normal[:, 0])
        
        print(f"[全局参数] RR均值={self.global_params['rr_mean']:.3f}, "
              f"标准差={self.global_params['rr_std']:.3f}")
    
    def predict_adaptive(self, X_test, window=100):
        """
        自适应预测：使用滑动窗口估计局部基线
        :param window: 滑动窗口大小
        """
        predictions = []
        n = len(X_test)
        
        for i in range(n):
            # 获取局部窗口
            start = max(0, i - window // 2)
            end = min(n, i + window // 2)
            
            local_rr = X_test[start:end, 0]
            local_median = np.median(local_rr)  # 使用中位数更鲁棒
            
            current_rr = X_test[i, 0]
            
            # 判断：当前RR与局部基线的偏差
            if current_rr < 0.85 * local_median:
                predictions.append('X')
            elif current_rr > 1.2 * local_median:
                predictions.append('X')
            else:
                predictions.append('N')
        
        return np.array(predictions)
