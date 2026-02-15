"""
完整测试所有训练文件的心拍丢弃情况
"""
import data_loader
import config

print("="*70)
print("检查所有训练文件的心拍丢弃情况")
print("="*70)

total_original = 0
total_valid = 0
total_dropped = 0

for filename in config.TRAIN_FILES:
    try:
        signal = data_loader.load_ecg_signal(filename)
        df = data_loader.load_annotations(filename)
        
        r_indices = df['R_Index'].values
        segments, valid_beat_indices = data_loader.get_heartbeat_segments(signal, r_indices)
        
        n_original = len(r_indices)
        n_valid = len(segments)
        n_dropped = n_original - n_valid
        
        total_original += n_original
        total_valid += n_valid
        total_dropped += n_dropped
        
        status = "✅" if n_dropped == 0 else "⚠️"
        print(f"\n{status} {filename}")
        print(f"   原始: {n_original}  有效: {n_valid}  丢弃: {n_dropped}")
        
        if n_dropped > 0:
            drop_rate = n_dropped / n_original * 100
            print(f"   丢弃率: {drop_rate:.2f}%")
            print(f"   首个丢弃索引: {[i for i in range(n_original) if i not in valid_beat_indices][:5]}")
        
    except Exception as e:
        print(f"\n❌ {filename}: {e}")

print("\n" + "="*70)
print("汇总统计")
print("="*70)
print(f"总原始心拍: {total_original}")
print(f"总有效心拍: {total_valid}")
print(f"总丢弃心拍: {total_dropped}")
print(f"总丢弃率: {total_dropped/total_original*100:.4f}%")

if total_dropped == 0:
    print("\n✅ 所有训练文件都没有心拍被丢弃！")
    print("   这意味着：")
    print("   1. 训练数据质量很好，没有边界问题")
    print("   2. 之前的bug虽然存在，但在训练集上没有实际影响")
    print("   3. 但修复仍然是必要的，因为测试集可能有丢弃")
else:
    print(f"\n⚠️  有 {total_dropped} 个心拍被丢弃")
    print("   标签对齐修复非常重要！")

# 测试一下测试集
print("\n" + "="*70)
print("检查测试文件（如果有标注）")
print("="*70)

for filename in config.TEST_FILES:
    try:
        signal = data_loader.load_ecg_signal(filename)
        df = data_loader.load_annotations(filename)
        
        if df is not None:
            r_indices = df['R_Index'].values
            segments, valid_beat_indices = data_loader.get_heartbeat_segments(signal, r_indices)
            
            n_original = len(r_indices)
            n_valid = len(segments)
            n_dropped = n_original - n_valid
            
            status = "✅" if n_dropped == 0 else "⚠️"
            print(f"\n{status} {filename}")
            print(f"   原始: {n_original}  有效: {n_valid}  丢弃: {n_dropped}")
            
            if n_dropped > 0:
                drop_rate = n_dropped / n_original * 100
                print(f"   丢弃率: {drop_rate:.2f}%")
        else:
            print(f"\n- {filename}: 无标注文件（正常）")
            
    except Exception as e:
        print(f"\n- {filename}: {e}")
