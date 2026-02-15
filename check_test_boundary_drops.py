"""
检查测试集会丢失多少边界心拍
"""
import config
import data_loader

print("="*70)
print("测试集边界心拍丢失检查")
print("="*70)

test_files = ['H25113017U411', 'H251204210MS0', 'H2512053F662E', 'H2512082SFFUB']

total_drops = 0
for fname in test_files:
    signal = data_loader.load_ecg_signal(fname)
    df = data_loader.load_annotations(fname)
    r_indices = df['R_Index'].values
    
    # 使用padding模式
    segments, valid_beat_indices, is_boundary = data_loader.get_heartbeat_segments(
        signal, r_indices, use_padding=True
    )
    
    dropped = len(r_indices) - len(valid_beat_indices)
    drop_rate = dropped / len(r_indices) * 100
    
    print(f"\n{fname}")
    print(f"  CSV心拍数: {len(r_indices)}")
    print(f"  有效心拍数: {len(valid_beat_indices)}")
    print(f"  边界心拍数: {sum(is_boundary)}")
    print(f"  丢失心拍数: {dropped} ({drop_rate:.3f}%)")
    
    if dropped > 0:
        print(f"  ❌ 长度不匹配！评分脚本会报错！")
    elif len(valid_beat_indices) == len(r_indices):
        print(f"  ✅ 长度完美匹配！")
        
    total_drops += dropped

print("\n" + "="*70)
if total_drops > 0:
    print(f"⚠️ 总计丢失 {total_drops} 个心拍，预测文件长度会对不上CSV！")
    print("💡 建议：使用padding保证每个心拍都能输出预测结果")
else:
    print("✅ 所有心拍都能切出有效窗口，长度对齐OK")
print("="*70)
