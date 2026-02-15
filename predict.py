"""
预测模块 - 仅用于对测试集进行推理
"""
import os
import numpy as np
import scipy.io as sio
import pickle
from datetime import datetime

import config
import data_loader
from features import FeatureExtractor
from algorithm import RuleBasedClassifier

def load_model(model_path=None):
    """加载训练好的模型"""
    if model_path is None:
        # 默认加载最新模型
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_latest.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            "请先运行 train.py 训练模型！"
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"[模型加载] {model_path}")
    return model

def predict_single_file(model, filename, save_output=True):
    """预测单个测试文件"""
    print(f"\n{'='*60}")
    print(f"预测文件: {filename}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    try:
        signal = data_loader.load_ecg_signal(filename)
        df = data_loader.load_annotations(filename)
    except Exception as e:
        print(f"[错误] 无法加载文件: {e}")
        return None
    
    # 2. 切片（测试集必须使用padding保证全覆盖）
    r_indices = df['R_Index'].values
    segments, valid_beat_indices, is_boundary = data_loader.get_heartbeat_segments(
        signal, r_indices, use_padding=True
    )
    
    print(f"[信息] 共检测到 {len(segments)} 个心拍（CSV总数:{len(r_indices)}，边界:{sum(is_boundary)}）")
    
    # 确认长度完全匹配
    if len(segments) != len(r_indices):
        print(f"⚠️ [警告] 长度不匹配！预测={len(segments)}, CSV={len(r_indices)}"))
    
    # 3. 提取特征
    extractor = FeatureExtractor(fs=config.FS)
    actual_r_positions = r_indices[valid_beat_indices]
    X_test = extractor.extract_batch(segments, actual_r_positions)
    
    # 4. 预测
    print("[预测中...]")
    y_pred = model.predict(X_test)
    
    # 5. 统计结果
    n_abnormal = sum(y_pred == 'X')
    print(f"[结果] 检测到 {n_abnormal} 个异常心拍 ({n_abnormal/len(y_pred)*100:.2f}%)")
    print(f"        其中边界心拍: {sum(is_boundary)} 个")
    
    # 6. 保存结果
    if save_output:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"{filename}_prediction.mat")
        sio.savemat(save_path, {'Beat_Types': y_pred})
        
        print(f"[保存] {save_path}")
    
    return y_pred

def predict_all_test_files(model, save_output=True):
    """预测所有测试文件"""
    print("\n" + "="*60)
    print("=== 批量预测测试集 ===")
    print("="*60)
    
    results = {}
    
    for test_file in config.TEST_FILES:
        y_pred = predict_single_file(model, test_file, save_output=save_output)
        if y_pred is not None:
            results[test_file] = y_pred
    
    print("\n" + "="*60)
    print("=== 预测完成 ===")
    print("="*60)
    print(f"\n成功预测 {len(results)}/{len(config.TEST_FILES)} 个文件")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='对测试集进行预测')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径（默认使用最新模型）')
    parser.add_argument('--file', type=str, default=None, help='预测单个文件（默认预测所有测试文件）')
    
    args = parser.parse_args()
    
    # 加载模型
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        exit(1)
    
    # 预测
    if args.file:
        # 单文件预测
        predict_single_file(model, args.file, save_output=True)
    else:
        # 批量预测
        predict_all_test_files(model, save_output=True)
    
    print("\n✅ 预测完成！结果保存在 output/ 目录下")
