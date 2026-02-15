import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import config
import data_loader
from features import FeatureExtractor
from algorithm import RuleBasedClassifier

def evaluate_metrics(y_true, y_pred):
    """计算准确率、F1等指标"""
    # 简单的混淆矩阵逻辑
    TP = sum((y_true == 'X') & (y_pred == 'X'))
    TN = sum((y_true == 'N') & (y_pred == 'N'))
    FP = sum((y_true == 'N') & (y_pred == 'X'))
    FN = sum((y_true == 'X') & (y_pred == 'N'))
    
    acc = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6) # 查全率 (最重要的指标：找出所有病变)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"  [结果] Acc: {acc:.4f} | Recall(查全): {recall:.4f} | F1: {f1:.4f}")
    print(f"  [详情] TP(抓对):{TP}  FN(漏抓):{FN}  FP(误抓):{FP}")
    return acc, f1

def process_files(file_list, is_training=True):
    """读取文件列表并提取特征"""
    all_features = []
    all_labels = [] # 测试集时这里会是空的
    
    extractor = FeatureExtractor(fs=config.FS)
    
    print(f"正在处理 {'训练' if is_training else '测试'} 数据集...")
    for fname in file_list:
        print(f"-> 读取: {fname}")
        
        # 1. 读信号
        try:
            signal = data_loader.load_ecg_signal(fname)
            df = data_loader.load_annotations(fname)
        except Exception as e:
            print(f"   出错跳过: {e}")
            continue
            
        # 2. 切片
        r_indices = df['R_Index'].values
        segments, valid_r = data_loader.get_heartbeat_segments(signal, r_indices)
        
        # 3. 提取特征
        USE_DEBUG_MODE = True  # 设为False使用全部数据
        
        if USE_DEBUG_MODE:
            segments_to_use = segments[:5000]
            valid_r_to_use = valid_r[:5000]
        else:
            segments_to_use = segments
            valid_r_to_use = valid_r
            
        feats = extractor.extract_batch(segments_to_use, valid_r_to_use)
        all_features.append(feats)
        
        if is_training:
            # 对齐标签 - 必须与特征数量一致！
            labels = df['Beat Symbol'].values[:len(segments_to_use)]
            all_labels.append(labels)
            
    # 拼接所有数据
    X = np.vstack(all_features)
    if is_training:
        y = np.concatenate(all_labels)
        return X, y
    else:
        return X, df # 测试集需要返回 df 以便知道有多少行

def run_loocv():
    """
    留一交叉验证 (Leave-One-Out Cross Validation)
    """
    print("\n=== 留一交叉验证 ===")
    all_files = config.TRAIN_FILES
    
    all_accuracies = []
    all_recalls = []
    all_f1s = []
    
    for i, test_file in enumerate(all_files):
        print(f"\n--- 第 {i+1}/{len(all_files)} 折：测试 {test_file} ---")
        
        # 留一法：当前文件作为测试集，其余作为训练集
        train_files = [f for f in all_files if f != test_file]
        
        # 训练
        print("训练集：", train_files)
        X_train, y_train = process_files(train_files, is_training=True)
        print(f"训练集形状: {X_train.shape}")
        
        model = RuleBasedClassifier()
        model.fit(X_train, y_train)
        
        # 测试
        print(f"测试集：[{test_file}]")
        X_test, y_test = process_files([test_file], is_training=True)
        print(f"测试集形状: {X_test.shape}")
        
        y_pred = model.predict(X_test)
        acc, f1 = evaluate_metrics(y_test, y_pred)
        
        all_accuracies.append(acc)
        all_f1s.append(f1)
        
    # 汇总结果
    print("\n=== 留一交叉验证总结 ===")
    print(f"平均准确率: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"平均F1分数: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
    
    return np.mean(all_accuracies), np.mean(all_f1s)

def run_pipeline():
    # --- 1. 留一交叉验证阶段 ---
    print("\n=== 阶段一：留一交叉验证 ===")
    run_loocv()
    
    # --- 2. 使用全部训练数据训练最终模型 ---
    print("\n=== 阶段二：训练最终模型 (使用全部训练数据) ===")
    X_train, y_train = process_files(config.TRAIN_FILES, is_training=True)
    print(f"训练集形状: {X_train.shape}")
    
    model = RuleBasedClassifier()
    model.fit(X_train, y_train)
    
    # --- 3. 测试/提交阶段 ---
    print("\n=== 阶段三：生成提交结果 (测试) ===")
    # 遍历每个测试文件，单独预测并保存
    for test_file in config.TEST_FILES:
        print(f"正在处理测试文件: {test_file}")
        
        # 1. 读+提特征
        # 注意：这里我们单独处理每个文件，因为要单独保存
        signal = data_loader.load_ecg_signal(test_file)
        df = data_loader.load_annotations(test_file)
        r_indices = df['R_Index'].values
        segments, valid_r = data_loader.get_heartbeat_segments(signal, r_indices)
        
        extractor = FeatureExtractor(fs=config.FS)
        X_test = extractor.extract_batch(segments, valid_r)
        
        # 2. 预测
        y_pred = model.predict(X_test)
        
        # 3. 格式化保存
        # 保存到output目录，避免覆盖原始数据
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"{test_file}_prediction.mat")
        
        # 转换格式：['N', 'X'] -> np.array
        # 使用Beat_Types作为变量名
        sio.savemat(save_path, {'Beat_Types': y_pred})
        print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    run_pipeline()