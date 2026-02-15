"""
训练与验证模块 - 仅用于模型开发和交叉验证
"""
import os
import numpy as np
import pickle
from datetime import datetime
from algorithm_f1_optimized import BalancedClassifier as RuleBasedClassifier
import config
from features import extract_features_with_cache

def evaluate_metrics(y_true, y_pred):
    """计算准确率、F1等指标"""
    TP = sum((y_true == 'X') & (y_pred == 'X'))
    TN = sum((y_true == 'N') & (y_pred == 'N'))
    FP = sum((y_true == 'N') & (y_pred == 'X'))
    FN = sum((y_true == 'X') & (y_pred == 'N'))
    
    acc = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"  [结果] Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"  [详情] TP:{TP}  TN:{TN}  FP:{FP}  FN:{FN}")
    print(f"  [分析] 异常总数:{TP+FN}  误报率:{FP/(TN+FP+1e-6):.4f}")
    
    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

def process_files(file_list, is_training=True, use_full_data=False, use_cache=True):
    """读取文件列表并提取特征（支持缓存加速）"""
    all_features = []
    all_labels = []
    all_file_info = []  # 记录每个心拍来自哪个文件
    
    cache_status = "缓存:开启 🚀" if use_cache else "缓存:关闭"
    print(f"\n正在处理 {'训练' if is_training else '测试'} 数据集... ({cache_status})")
    
    for fname in file_list:
        print(f"-> {fname}", end=" ")
        
        try:
            # 使用缓存加载特征
            X_file, y_file, r_indices, valid_beat_indices = extract_features_with_cache(
                fname, use_cache=use_cache
            )
            
            # 调试模式限制数据量
            if not use_full_data and len(X_file) > 5000:
                print(f"[调试:前5000/{len(X_file)}]", end=" ")
                X_file = X_file[:5000]
                y_file = y_file[:5000]
            
            all_features.append(X_file)
            
            if is_training:
                all_labels.append(y_file)
                all_file_info.extend([fname] * len(X_file))
                
                # 统计
                n_abnormal = sum(y_file == 'X')
                abnormal_rate = n_abnormal / len(y_file) * 100
                print(f"✓ {len(X_file)}心拍, 异常:{n_abnormal}({abnormal_rate:.1f}%)")
            else:
                print(f"✓ {len(X_file)}心拍")
                
        except Exception as e:
            print(f"✗ 错误: {e}")
            continue
    
    X = np.vstack(all_features)
    
    if is_training:
        y = np.concatenate(all_labels)
        total_abnormal = sum(y == 'X')
        print(f"\n[汇总] 特征:{X.shape}, 异常:{total_abnormal}/{len(y)}({total_abnormal/len(y)*100:.2f}%)")
        return X, y, all_file_info
    else:
        return X

def analyze_failed_cases(X, y, y_pred, file_info, fold_name):
    """分析预测失败的案例"""
    print(f"\n[深度分析] {fold_name}")
    
    # 找出假阴性（漏检）
    fn_indices = np.where((y == 'X') & (y_pred == 'N'))[0]
    if len(fn_indices) > 0:
        print(f"  假阴性（漏检）样本数: {len(fn_indices)}")
        # 分析这些样本的特征分布
        fn_features = X[fn_indices]
        print(f"  漏检样本RR比率均值: {np.mean(fn_features[:, 0]):.3f}")
        print(f"  漏检样本RR比率最小: {np.min(fn_features[:, 0]):.3f}")
        
    # 找出假阳性（误报）
    fp_indices = np.where((y == 'N') & (y_pred == 'X'))[0]
    if len(fp_indices) > 0:
        print(f"  假阳性（误报）样本数: {len(fp_indices)}")
        fp_features = X[fp_indices]
        print(f"  误报样本RR比率均值: {np.mean(fp_features[:, 0]):.3f}")

def run_loocv(use_full_data=False, use_cache=True):
    """留一交叉验证（支持缓存加速）"""
    print("\n" + "="*70)
    print("=== 留一交叉验证（LOOCV）===")
    print("="*70)
    
    all_files = config.TRAIN_FILES
    results = []
    
    import time
    total_start = time.time()
    
    for i, test_file in enumerate(all_files):
        print(f"\n{'='*70}")
        print(f"Fold {i+1}/{len(all_files)}: 测试 {test_file}")
        print(f"{'='*70}")
        
        fold_start = time.time()
        
        # 留一法划分
        train_files = [f for f in all_files if f != test_file]
        
        # 训练
        print("\n[训练阶段]")
        X_train, y_train, _ = process_files(train_files, is_training=True, 
                                           use_full_data=use_full_data, use_cache=use_cache)
        
        model = RuleBasedClassifier()
        model.fit(X_train, y_train)
        
        # 测试
        print("\n[测试阶段]")
        X_test, y_test, file_info = process_files([test_file], is_training=True, 
                                                  use_full_data=use_full_data, use_cache=use_cache)
        
        y_pred = model.predict(X_test)
        metrics = evaluate_metrics(y_test, y_pred)
        
        # 深度分析
        analyze_failed_cases(X_test, y_test, y_pred, file_info, test_file)
        
        # 保存结果
        metrics['file'] = test_file
        results.append(metrics)
        
        fold_time = time.time() - fold_start
        print(f"\n⏱️  本折用时: {fold_time:.1f}秒")
    
    # 汇总报告
    print("\n" + "="*60)
    print("=== 交叉验证汇总 ===")
    print("="*60)
    
    print(f"\n{'文件名':<20} {'Acc':<8} {'Recall':<8} {'F1':<8} {'异常数':<8}")
    print("-"*60)
    for r in results:
        print(f"{r['file']:<20} {r['acc']:<8.4f} {r['recall']:<8.4f} {r['f1']:<8.4f} {r['TP']+r['FN']:<8}")
    
    print("\n" + "-"*60)
    accs = [r['acc'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    print(f"平均 Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"平均 Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"平均 F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    total_time = time.time() - total_start
    print(f"\n⏱️  LOOCV总用时: {total_time/60:.1f}分钟")
    
    return results

def train_final_model(use_full_data=False, use_cache=True, save_model=True):
    """使用全部训练数据训练最终模型"""
    print("\n" + "="*70)
    print("=== 训练最终模型（使用全部训练数据）===")
    print("="*70)
    
    X_train, y_train, _ = process_files(config.TRAIN_FILES, is_training=True, 
                                       use_full_data=use_full_data, use_cache=use_cache)
    
    model = RuleBasedClassifier()
    model.fit(X_train, y_train)
    
    # 保存模型
    if save_model:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"model_{timestamp}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n[模型保存] {model_path}")
        
        # 同时保存一个"latest"版本方便预测时加载
        latest_path = os.path.join(model_dir, "model_latest.pkl")
        with open(latest_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[模型保存] {latest_path}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练与验证心拍分类模型')
    parser.add_argument('--full', action='store_true', help='使用全部数据（默认只用前5000条调试）')
    parser.add_argument('--skip-cv', action='store_true', help='跳过交叉验证，直接训练最终模型')
    parser.add_argument('--no-cache', action='store_true', help='不使用特征缓存（调试用）')
    
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    
    if args.full:
        print("\n⚠️  使用全部数据模式，训练时间会较长...")
    else:
        print("\n💡 调试模式：每个文件只用前5000个心拍")
        print("   正式训练请加 --full 参数")
    
    if use_cache:
        print("🚀 特征缓存已启用（首次需运行 python preprocess_features.py）")
    else:
        print("⚠️  特征缓存已禁用，将重新提取特征")
    print()
    
    # 留一交叉验证
    if not args.skip_cv:
        cv_results = run_loocv(use_full_data=args.full, use_cache=use_cache)
    
    # 训练最终模型
    final_model = train_final_model(use_full_data=args.full, use_cache=use_cache, save_model=True)
    
    print("\n✅ 训练完成！现在可以运行 predict.py 进行测试集预测")
