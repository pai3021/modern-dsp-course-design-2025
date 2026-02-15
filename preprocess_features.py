"""
预处理：为所有训练文件提取并缓存特征
运行一次即可，后续训练直接使用缓存
"""
import time
import os
import config
from features import extract_features_with_cache, get_cache_path

print("="*70)
print("特征预处理 - 提取并缓存所有训练文件的特征")
print("="*70)
print(f"训练文件数量: {len(config.TRAIN_FILES)}")
print(f"缓存目录: feature_cache/")
print("="*70)

total_start = time.time()

for i, filename in enumerate(config.TRAIN_FILES, 1):
    print(f"\n[{i}/{len(config.TRAIN_FILES)}] 处理: {filename}")
    
    # 删除旧缓存（如果存在）
    cache_path = get_cache_path(filename)
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  [清理] 删除旧缓存: {cache_path}")
    
    start = time.time()
    
    # 提取并缓存特征（use_cache=True 会自动保存）
    X, y, r_indices, valid_beat_indices = extract_features_with_cache(
        filename, use_cache=True  # 缓存不存在时会重新提取并保存
    )
    
    elapsed = time.time() - start
    print(f"  ✅ 完成: {len(X)} 个心拍, 用时 {elapsed:.1f}s")
    print(f"      特征: {X.shape}, 异常率: {sum(y=='X')/len(y)*100:.2f}%")

total_elapsed = time.time() - total_start
print("\n" + "="*70)
print(f"✅ 所有特征已缓存，总用时: {total_elapsed/60:.1f} 分钟")
