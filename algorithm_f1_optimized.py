"""
F1优化 + 抗跨记录幅值漂移 + 记录级自适应（不改features，只改算法）

特征顺序（与 features.py 一致）：
[0] rr_pre, [1] rr_post, [2] wt_low, [3] wt_mid, [4] wt_high,
[5] peak_peak, [6] skew, [7] kurt, [8] entropy
"""

import numpy as np


class F1OptimizedClassifier:
    def __init__(self):
        self._mean = None
        self._std = None
        self._q_low = None   # 正常 5% 分位
        self._q_high = None  # 正常 95% 分位
        self.thresholds = {}

        # 尺度不敏感派生能量特征的正常分布
        self.derived_mean = {}
        self.derived_std = {}
        self.derived_q05 = {}
        self.derived_q95 = {}

    @staticmethod
    def _safe_std(x: np.ndarray) -> float:
        s = float(np.std(x))
        return s if s > 1e-6 else 1e-6

    @staticmethod
    def _derive_energy_feats(X: np.ndarray):
        """
        从 wt_low, wt_mid, wt_high 计算尺度不敏感派生特征
        返回 shape=(n, 4): [low_frac, mid_frac, high_frac, mid_over_low]
        """
        e_low = X[:, 2]
        e_mid = X[:, 3]
        e_high = X[:, 4]
        eps = 1e-8
        total = e_low + e_mid + e_high + eps

        low_frac = e_low / total
        mid_frac = e_mid / total
        high_frac = e_high / total
        mid_over_low = e_mid / (e_low + eps)
        return np.vstack([low_frac, mid_frac, high_frac, mid_over_low]).T

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        n_idx = np.where(y_train == 'N')[0]
        x_idx = np.where(y_train == 'X')[0]
        if len(n_idx) == 0:
            raise ValueError("训练集中没有正常样本(N)，无法建立正常分布。")

        XN = X_train[n_idx]
        XA = X_train[x_idx] if len(x_idx) > 0 else None

        print(f"[F1优化训练] 正常:{len(XN)}, 异常:{len(x_idx)}")

        # 正常统计（用于 skew/kurt/entropy 等相对稳定的统计量）
        self._mean = np.mean(XN, axis=0)
        self._std = np.std(XN, axis=0)
        self._std = np.where(self._std < 1e-6, 1e-6, self._std)

        # 正常 5%~95% 分位（用于“正常判断/兜底”）
        self._q_low = np.quantile(XN, 0.05, axis=0)
        self._q_high = np.quantile(XN, 0.95, axis=0)

        # RR阈值（保留你之前的思路）
        rr_mean = float(self._mean[0])
        rr_std = float(self._std[0])
        rr_min = rr_mean - 2.5 * rr_std
        rr_max = rr_mean + 2.5 * rr_std

        # 用正常分位夹紧
        rr_min = min(rr_min, float(self._q_low[0]))
        rr_max = max(rr_max, float(self._q_high[0]))

        if XA is not None and len(XA) > 10:
            abnormal_rr_mean = float(np.mean(XA[:, 0]))
            abnormal_rr_std = self._safe_std(XA[:, 0])
            midpoint = (rr_mean + abnormal_rr_mean) / 2.0
            rr_min = min(rr_min, midpoint)
            print(f"[异常模式] RR均值={abnormal_rr_mean:.3f}±{abnormal_rr_std:.3f}")

        self.thresholds['rr_pre_min'] = float(rr_min)
        self.thresholds['rr_pre_max'] = float(rr_max)
        print(f"[F1阈值] RR下界={rr_min:.3f}, 上界={rr_max:.3f}")

        # 派生能量特征的正常分布（解决跨记录幅值漂移）
        D = self._derive_energy_feats(XN)
        d_mean = np.mean(D, axis=0)
        d_std = np.std(D, axis=0)
        d_std = np.where(d_std < 1e-6, 1e-6, d_std)
        d_q05 = np.quantile(D, 0.05, axis=0)
        d_q95 = np.quantile(D, 0.95, axis=0)

        names = ['low_frac', 'mid_frac', 'high_frac', 'mid_over_low']
        for k in range(4):
            self.derived_mean[names[k]] = float(d_mean[k])
            self.derived_std[names[k]] = float(d_std[k])
            self.derived_q05[names[k]] = float(d_q05[k])
            self.derived_q95[names[k]] = float(d_q95[k])

    def _outside_count(self, rr_pre, rr_post, low_frac, mid_frac, entropy, kurt):
        """
        只用相对稳定/尺度不敏感特征做 outside 统计：
        RR、能量占比、熵、峭度（不使用绝对能量/peak_peak）
        """
        cnt = 0
        rr_min = self.thresholds['rr_pre_min']
        rr_max = self.thresholds['rr_pre_max']

        # RR
        if rr_pre < rr_min or rr_pre > rr_max:
            cnt += 1
        if rr_post < rr_min * 0.95 or rr_post > rr_max * 1.05:
            cnt += 1

        # 能量占比
        if low_frac < self.derived_q05['low_frac'] or low_frac > self.derived_q95['low_frac']:
            cnt += 1
        if mid_frac < self.derived_q05['mid_frac'] or mid_frac > self.derived_q95['mid_frac']:
            cnt += 1

        # 熵、峭度（正常分位）
        if entropy < float(self._q_low[8]) or entropy > float(self._q_high[8]):
            cnt += 1
        if kurt < float(self._q_low[7]) or kurt > float(self._q_high[7]):
            cnt += 1

        return cnt

    def _predict_raw(self, X_test: np.ndarray):
        n = len(X_test)
        preds = np.empty(n, dtype='<U1')
        scores = np.zeros(n, dtype=np.int32)
        outside = np.zeros(n, dtype=np.int32)

        rr_min = self.thresholds['rr_pre_min']
        rr_max = self.thresholds['rr_pre_max']

        D = self._derive_energy_feats(X_test)
        low_frac = D[:, 0]
        mid_frac = D[:, 1]
        high_frac = D[:, 2]
        mid_over_low = D[:, 3]

        def dz(val, name):
            return (val - self.derived_mean[name]) / (self.derived_std[name] + 1e-6)

        rr_pre_arr = X_test[:, 0]
        rr_post_arr = X_test[:, 1]

        for i in range(n):
            row = X_test[i]
            rr_pre = float(row[0])
            rr_post = float(row[1])

            z_skew = (float(row[6]) - float(self._mean[6])) / (float(self._std[6]) + 1e-6)
            z_kurt = (float(row[7]) - float(self._mean[7])) / (float(self._std[7]) + 1e-6)
            z_ent = (float(row[8]) - float(self._mean[8])) / (float(self._std[8]) + 1e-6)

            z_lf = dz(float(low_frac[i]), 'low_frac')
            z_mf = dz(float(mid_frac[i]), 'mid_frac')
            z_hf = dz(float(high_frac[i]), 'high_frac')
            z_mol = dz(float(mid_over_low[i]), 'mid_over_low')

            outside[i] = self._outside_count(
                rr_pre, rr_post,
                float(low_frac[i]), float(mid_frac[i]),
                float(row[8]), float(row[7])
            )

            score = 0
            rr_strong = False

            # RR（强证据）
            if rr_pre < rr_min:
                dev = (rr_min - rr_pre) / (float(self._std[0]) + 1e-6)
                score += 4 if dev > 2.0 else 2
                rr_strong = True
            elif rr_pre > rr_max:
                score += 2
                rr_strong = True

            if rr_post < rr_min * 0.95 or rr_post > rr_max * 1.05:
                score += 1

            # 尺度不敏感能量形态
            if abs(z_lf) > 3.0: score += 1
            if abs(z_mf) > 3.0: score += 1
            if abs(z_mol) > 3.0: score += 1
            if abs(z_hf) > 3.5: score += 1

            # 形态统计（相对稳定）
            if abs(z_kurt) > 3.2: score += 1
            if abs(z_skew) > 3.8: score += 1
            if abs(z_ent) > 2.8: score += 1

            # peak_peak 降权：只补分
            if rr_strong or score >= 4:
                z_pp = (float(row[5]) - float(self._mean[5])) / (float(self._std[5]) + 1e-6)
                if abs(z_pp) > 3.8:
                    score += 1

            # outside 兜底
            if outside[i] >= 4:
                score += 2
            elif outside[i] >= 3:
                score += 1

            scores[i] = score

            # 判决：RR“看起来正常”时稍严格，但不要太死（先用 6）
            rr_normalish = (rr_min * 0.97 <= rr_pre <= rr_max * 1.03) and (rr_min * 0.97 <= rr_post <= rr_max * 1.03)
            if rr_normalish:
                preds[i] = 'X' if score >= 6 else 'N'
            else:
                preds[i] = 'X' if score >= 4 else 'N'

        return preds, scores, outside, rr_pre_arr, rr_post_arr, low_frac, mid_frac

    def predict(self, X_test: np.ndarray):
        preds, _, _, _, _, _, _ = self._predict_raw(X_test)
        return preds


class BalancedClassifier:
    def __init__(self):
        self.global_model = F1OptimizedClassifier()

    def fit(self, X_train, y_train):
        self.global_model.fit(X_train, y_train)

    def predict(self, X_test):
        preds, scores, outside, rr_pre, rr_post, low_frac, mid_frac = self.global_model._predict_raw(X_test)

        rr_min = self.global_model.thresholds['rr_pre_min']
        rr_max = self.global_model.thresholds['rr_pre_max']

        pred_abn_rate = float(np.mean(preds == 'X'))
        rr_outlier_rate = float(np.mean((rr_pre < rr_min) | (rr_pre > rr_max)))

        # ==========================================================
        # 模式 1：高异常模式（反向逻辑，专治 H2511143S7N68 这种 91% 异常）
        # 先估计“非常正常”的拍占比，如果很低 => 直接采用“非正常即异常”
        # ==========================================================
        ent = X_test[:, 8]
        ku = X_test[:, 7]

        is_normal = (
            (rr_min * 0.97 <= rr_pre) & (rr_pre <= rr_max * 1.03) &
            (rr_min * 0.97 <= rr_post) & (rr_post <= rr_max * 1.03) &
            (self.global_model.derived_q05['low_frac'] <= low_frac) & (low_frac <= self.global_model.derived_q95['low_frac']) &
            (self.global_model.derived_q05['mid_frac'] <= mid_frac) & (mid_frac <= self.global_model.derived_q95['mid_frac']) &
            (self.global_model._q_low[8] <= ent) & (ent <= self.global_model._q_high[8]) &
            (self.global_model._q_low[7] <= ku) & (ku <= self.global_model._q_high[7])
        )
        normal_rate = float(np.mean(is_normal))

        if normal_rate < 0.35:
            print(f"[高异常模式] normal_rate={normal_rate:.2%}，启用“非正常即异常”")
            preds = np.where(is_normal, 'N', 'X')
            pred_abn_rate = float(np.mean(preds == 'X'))
            # 高异常模式下不再做“收紧”，只做轻度平滑即可

        # ==========================================================
        # 模式 2：误报收紧（防止 Fold4 类的整段漂移误报）
        # 如果预测异常比例很高但 RR 基本正常 => 收紧
        # ==========================================================
        if pred_abn_rate > 0.35 and rr_outlier_rate < 0.08:
            print(f"[自适应-收紧] pred_X={pred_abn_rate:.2%}, rr_outlier={rr_outlier_rate:.2%}，疑似误报偏多，收紧")
            keep = (scores >= 7) | (rr_pre < rr_min) | (rr_pre > rr_max)
            preds = np.where(keep, 'X', 'N')
            pred_abn_rate = float(np.mean(preds == 'X'))

        # ==========================================================
        # 模式 3：低异常放宽（专治 1%~6% 异常文件的漏检）
        # ==========================================================
        if pred_abn_rate < 0.03:
            print(f"[低异常放宽] pred_X={pred_abn_rate:.2%}，放宽：score>=5 判X")
            preds = np.where((scores >= 5) | (rr_pre < rr_min) | (rr_pre > rr_max), 'X', preds)

        # 平滑：去掉孤立低分X，补充孤立N
        preds = self._smooth_predictions(preds, scores, rr_pre, rr_post, rr_min, rr_max, window=5)
        return preds

    def _smooth_predictions(self, preds, scores, rr_pre, rr_post, rr_min, rr_max, window=5):
        sm = preds.copy()
        n = len(preds)

        for i in range(n):
            s = max(0, i - window // 2)
            e = min(n, i + window // 2 + 1)
            x_ratio = float(np.mean(preds[s:e] == 'X'))

            rr_normalish = (rr_min * 0.97 <= rr_pre[i] <= rr_max * 1.03) and (rr_min * 0.97 <= rr_post[i] <= rr_max * 1.03)

            # 孤立X：窗口内X很少 + RR正常 + 分数不高 => 回填N（压FP）
            if preds[i] == 'X' and x_ratio < 0.25 and rr_normalish and scores[i] <= 6:
                sm[i] = 'N'

            # 孤立N：窗口内几乎全X => 回填X（压FN）
            if preds[i] == 'N' and x_ratio > 0.85:
                sm[i] = 'X'

        return sm
