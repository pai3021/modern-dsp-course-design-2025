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
from features import FeatureExtractor, extract_features_with_cache
from algorithm_f1_optimized import BalancedClassifier

def list_available_models():
    """列出所有可用模型"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        print("⚠️ models/ 目录不存在")
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    model_files.sort(reverse=True)  # 最新的在前
    
    if not model_files:
        print("⚠️ 没有找到任何模型文件")
        return []
    
    print(f"\n可用模型 ({len(model_files)} 个):")
    print("="*60)
    for i, model_file in enumerate(model_files[:10], 1):  # 只显示前10个
        model_path = os.path.join(models_dir, model_file)
        size = os.path.getsize(model_path) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        
        marker = "→" if model_file == "model_latest.pkl" else " "
        print(f"{marker} {i}. {model_file} ({size:.1f} KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    if len(model_files) > 10:
        print(f"   ... 及其他 {len(model_files) - 10} 个模型")
    
    return model_files

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
    
    print(f"\n[\u6a21型加载] {os.path.basename(model_path)}")
    print(f"[\u8def\u5f84] {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 显示模型类型
    model_type = type(model).__name__
    print(f"[\u7c7b\u578b] {model_type}")
    
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
    result = data_loader.get_heartbeat_segments(
        signal, r_indices, use_padding=True
    )
    
    # 处理返回值（use_padding=True 时返回3个值）
    if len(result) == 3:
        segments, valid_beat_indices, is_boundary = result
        print(f"[信息] 共检测到 {len(segments)} 个心拍（CSV总数:{len(r_indices)}，边界:{sum(is_boundary)}）")
    else:
        segments, valid_beat_indices = result
        is_boundary = [False] * len(segments)
        print(f"[信息] 共检测到 {len(segments)} 个心拍（CSV总数:{len(r_indices)}）")
    
    # 确认长度完全匹配
    if len(segments) != len(r_indices):
        print(f"⚠️ [警告] 长度不匹配！预测={len(segments)}, CSV={len(r_indices)}")
    
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
    if sum(is_boundary) > 0:
        print(f"        其中边界心拍: {sum(is_boundary)} 个")
    
    # 6. 保存结果
    if save_output:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"{filename}.mat")
        # 转换为 object 类型数组，确保在 MATLAB 中是 Cell Array 格式 (N x 1)
        y_pred_obj = y_pred.astype(object)
        sio.savemat(save_path, {'Beat_Types': y_pred_obj})
        
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
    parser.add_argument('--list-models', action='store_true', help='列出所有可用模型')
    
    args = parser.parse_args()
    
    # 如果是列出模型，则只显示模型列表
    if args.list_models:
        list_available_models()
        exit(0)
    
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
