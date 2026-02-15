"""
快速对比不同算法的性能 - 包含极端优化版本
"""
import numpy as np
import config
import data_loader
from features import FeatureExtractor
from algorithm import RuleBasedClassifier as OriginalClassifier
from algorithm_v2 import RuleBasedClassifier as ImprovedClassifier, AdaptiveClassifier
from algorithm_v3 import AggressiveClassifier, InvertedLogicClassifier, HybridClassifier

def quick_test_on_file(filename, model, model_name):
    """在单个文件上快速测试"""
    # 加载数据
    signal = data_loader.load_ecg_signal(filename)
    df = data_loader.load_annotations(filename)
    r_indices = df['R_Index'].values
    segments, valid_beat_indices = data_loader.get_heartbeat_segments(signal, r_indices)
    
    # 限制测试量
    n_test = min(5000, len(segments))
    segments = segments[:n_test]
    valid_beat_indices = valid_beat_indices[:n_test]
    
    # 提取特征
    extractor = FeatureExtractor(fs=config.FS)
    actual_r_positions = r_indices[valid_beat_indices]
    X = extractor.extract_batch(segments, actual_r_positions)
    y_true = df['Beat Symbol'].values[valid_beat_indices]
    
    # 预测
    if isinstance(model, AdaptiveClassifier):
        y_pred = model.predict_adaptive(X)
    elif isinstance(model, HybridClassifier):
        y_pred = model.predict(X)  # 自动判断模式
    else:
        y_pred = model.predict(X)
    
    # 评估
    TP = sum((y_true == 'X') & (y_pred == 'X'))
    TN = sum((y_true == 'N') & (y_pred == 'N'))
    FP = sum((y_true == 'N') & (y_pred == 'X'))
    FN = sum((y_true == 'X') & (y_pred == 'N'))
    
    acc = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"{model_name:<25} | Acc:{acc:.4f} | Recall:{recall:.4f} | F1:{f1:.4f} | TP:{TP} FN:{FN} FP:{FP}")
    
    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

def compare_algorithms():
    """对比所有算法"""
    print("\n" + "="*90)
    print("算法性能对比 - 针对问题文件 H2511143S7N68")
    print("="*90)
    
    # 准备训练数据（其他5个文件）
    train_files = [f for f in config.TRAIN_FILES if f != 'H2511143S7N68']
    
    print("\n准备训练数据...")
    all_X = []
    all_y = []
    extractor = FeatureExtractor(fs=config.FS)
    
    for fname in train_files:
        signal = data_loader.load_ecg_signal(fname)
        df = data_loader.load_annotations(fname)
        r_indices = df['R_Index'].values
        segments, valid_beat_indices = data_loader.get_heartbeat_segments(signal, r_indices)
        
        # 限制训练量
        n = min(5000, len(segments))
        actual_r_positions = r_indices[valid_beat_indices[:n]]
        X = extractor.extract_batch(segments[:n], actual_r_positions)
        y = df['Beat Symbol'].values[valid_beat_indices[:n]]
        
        all_X.append(X)
        all_y.append(y)
    
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    
    print(f"训练集: {X_train.shape}, 异常率: {sum(y_train=='X')/len(y_train)*100:.2f}%")
    
    # 初始化所有模型
    print("\n" + "="*90)
    print("训练模型...")
    print("="*90)
    
    model1 = OriginalClassifier()
    model1.fit(X_train, y_train)
    
    model2 = ImprovedClassifier(sensitivity='medium')
    model2.fit(X_train, y_train)
    
    model3 = ImprovedClassifier(sensitivity='high')
    model3.fit(X_train, y_train)
    
    model4 = AdaptiveClassifier()
    model4.fit(X_train, y_train)
    
    print("\n训练极端优化模型...")
    model5 = AggressiveClassifier()
    model5.fit(X_train, y_train)
    
    model6 = InvertedLogicClassifier()
    model6.fit(X_train, y_train)
    
    model7 = HybridClassifier()
    model7.fit(X_train, y_train)
    
    # 在H2511143S7N68上测试
    print("\n" + "="*90)
    print("测试结果 (H2511143S7N68):")
    print("="*90)
    
    test_file = 'H2511143S7N68'
    
    print("\n【基础算法】")
    r1 = quick_test_on_file(test_file, model1, "原始算法")
    r2 = quick_test_on_file(test_file, model2, "改进算法(中敏感度)")
    r3 = quick_test_on_file(test_file, model3, "改进算法(高敏感度)")
    r4 = quick_test_on_file(test_file, model4, "自适应算法")
    
    print("\n【极端优化算法】")
    r5 = quick_test_on_file(test_file, model5, "激进分类器")
    r6 = quick_test_on_file(test_file, model6, "反转逻辑分类器")
    r7 = quick_test_on_file(test_file, model7, "混合分类器")
    
    # 结果分析
    print("\n" + "="*90)
    print("结论:")
    print("="*90)
    
    results = {
        '原始': r1,
        '改进(中)': r2,
        '改进(高)': r3,
        '自适应': r4,
        '激进': r5,
        '反转逻辑': r6,
        '混合': r7
    }
    
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    
    print(f"\n✅ 召回率最高: {best_recall[0]} (Recall={best_recall[1]['recall']:.4f}, F1={best_recall[1]['f1']:.4f})")
    print(f"✅ F1最高: {best_f1[0]} (F1={best_f1[1]['f1']:.4f}, Recall={best_f1[1]['recall']:.4f})")
    
    if best_recall[1]['recall'] > 0.90:
        print(f"\n🎉 推荐使用: {best_recall[0]} （已达到优秀水平！）")
    elif best_recall[1]['recall'] > 0.80:
        print(f"\n✅ 推荐使用: {best_recall[0]} （表现良好）")
    elif best_recall[1]['recall'] > 0.70:
        print(f"\n⚠️  推荐使用: {best_recall[0]} （勉强可用，Recall={best_recall[1]['recall']:.2%}）")
    else:
        print(f"\n❌ 最佳Recall仅{best_recall[1]['recall']:.2%}，该文件可能需要特殊处理:")
        print("   1. 该文件为持续性心律失常（如房颤），异常率接近100%")
        print("   2. 建议单独标记或使用无监督方法")
        print("   3. 或考虑调整评估标准（该病例本身就是特例）")
    
    return results

if __name__ == "__main__":
    results = compare_algorithms()
