import numpy as np

class RuleBasedClassifier:
    def __init__(self):
        # 存储正常心拍的统计界限
        self.thresholds = {}
        
    def fit(self, X_train, y_train):
        """
        '训练'过程：实际上是统计正常心拍(N)的分布规律
        """
        # 1. 筛选出所有的正常心拍样本
        # 假设 y_train 中 'N' 是正常，其他是异常
        n_indices = [i for i, label in enumerate(y_train) if label == 'N']
        X_normal = X_train[n_indices]
        
        print(f"模型初始化：基于 {len(X_normal)} 个正常样本建立基准...")
        
        # 2. 针对每个特征，计算正常范围 (Mean ± K * Std)
        # 特征索引对应: 
        # 0: RR_Pre, 1: RR_Post, 
        # 2: WT_Low, 3: WT_Mid, 4: WT_High
        # 5: P-P, 6: Skew, 7: Kurt, 8: Entropy
        
        # --- 规则 A: 节律 (RR 间期) ---
        # RR 间期分布非常偏态，用分位数比用均值更准
        # 我们允许正常心拍在 0.85 到 1.15 倍均值之间波动
        self.thresholds['rr_pre_min'] = np.percentile(X_normal[:, 0], 1)  # 1% 分位点 (防极端离群值)
        self.thresholds['rr_pre_max'] = np.percentile(X_normal[:, 0], 99) # 99% 分位点
        
        # 手动修正一下，根据刚才的图，0.85 是个很安全的边界
        # 如果统计出来的下界比 0.85 还小，就强制设为 0.85，防止漏掉早搏
        self.thresholds['rr_pre_min'] = max(self.thresholds['rr_pre_min'], 0.85)
        
        # --- 规则 B: 形态 (小波能量 & 统计量) ---
        # 对于形态特征，我们使用 Z-score 思想：偏离均值太远就算异常
        # 记录每个形态特征的均值和标准差
        self.stat_params = {}
        # 我们重点关注几个核心特征：WT_Mid (能量), Kurt (尖锐度), Entropy (混乱度)
        target_features = {
            2: 'wt_low', 
            3: 'wt_mid', 
            7: 'kurt', 
            8: 'entropy'
        }
        
        for idx, name in target_features.items():
            mean = np.mean(X_normal[:, idx])
            std = np.std(X_normal[:, idx])
            self.stat_params[name] = (mean, std, idx)

    def predict(self, X_test):
        """
        预测过程：基于硬规则判决
        """
        predictions = []
        
        # 获取阈值
        rr_min = self.thresholds['rr_pre_min']
        
        for row in X_test:
            label = 'N' # 默认为正常
            reasons = [] # 调试用：记录为什么被判为异常
            
            # --- 判决 1: 节律检查 (Rhythm Check) ---
            # 这是最强的特征，权重最高
            rr_pre = row[0]
            
            # 如果 RR 间期显著小于正常值 (早搏)
            if rr_pre < 0.85: 
                label = 'X'
                reasons.append('Early_Beat')
            
            # 如果 RR 间期显著大于正常值 (漏搏/停搏)
            elif rr_pre > 1.25:
                label = 'X'
                reasons.append('Late_Beat')
                
            # --- 判决 2: 形态检查 (Morphology Check) ---
            # 只有当节律看似正常时，我们才去细扣形态
            # (因为如果节律都不对了，它肯定是 X，没必要再看形状)
            if label == 'N':
                # 检查小波能量是否异常 (过大或过小)
                # 例如：室性早搏(PVC)通常低频能量极大，波形宽大
                
                # 取出参数: 均值, 标准差, 索引
                m_low, s_low, i_low = self.stat_params['wt_low']
                m_mid, s_mid, i_mid = self.stat_params['wt_mid']
                
                val_low = row[i_low]
                val_mid = row[i_mid]
                
                # 计算 Z-score (偏离了几个标准差)
                z_low = abs(val_low - m_low) / (s_low + 1e-6)
                z_mid = abs(val_mid - m_mid) / (s_mid + 1e-6)
                
                # 阈值设定：如果偏离超过 5 倍标准差 (非常严格，防止误判)
                if z_low > 6.0: 
                    label = 'X'
                    reasons.append('Abnormal_Low_Energy')
                
                # 检查熵 (信号是否过于混乱)
                m_ent, s_ent, i_ent = self.stat_params['entropy']
                val_ent = row[i_ent]
                if val_ent > m_ent + 4 * s_ent:
                    label = 'X'
                    reasons.append('High_Entropy')

            predictions.append(label)
            
        return np.array(predictions)