"""
检查指定目录下所有 .mat 文件的内部结构
显示每个文件包含的变量名、形状、数据类型等信息
"""
import os
import glob
from scipy.io import loadmat
import numpy as np


def inspect_mat_structure(mat_path):
    """
    检查单个 .mat 文件的结构
    
    参数:
        mat_path: .mat 文件的完整路径
    """
    print(f"\n{'='*80}")
    print(f"文件: {os.path.basename(mat_path)}")
    print(f"完整路径: {mat_path}")
    print(f"{'='*80}")
    
    try:
        # 加载 .mat 文件
        mat_data = loadmat(mat_path)
        
        # 过滤掉 MATLAB 内部元数据字段（以双下划线开头和结尾）
        data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        
        if not data_keys:
            print("  ⚠ 文件中没有找到数据变量（仅包含元数据）")
            return
        
        print(f"\n找到 {len(data_keys)} 个变量:\n")
        
        for key in data_keys:
            value = mat_data[key]
            
            # 基本信息
            print(f"  变量名: {key}")
            print(f"    类型: {type(value).__name__}")
            
            # 如果是 numpy 数组，显示详细信息
            if isinstance(value, np.ndarray):
                print(f"    数据类型: {value.dtype}")
                print(f"    形状: {value.shape}")
                print(f"    维度: {value.ndim}D")
                print(f"    元素总数: {value.size}")
                
                # 如果是标量或小数组，显示实际值
                if value.size == 1:
                    print(f"    值: {value.item()}")
                elif value.size <= 10 and value.ndim == 1:
                    print(f"    值: {value.flatten()}")
                elif value.size <= 20:
                    print(f"    前几个值: {value.flatten()[:10]}")
                    if value.size > 10:
                        print(f"    ... (共 {value.size} 个元素)")
                else:
                    # 显示统计信息
                    if np.issubdtype(value.dtype, np.number):
                        print(f"    最小值: {np.min(value)}")
                        print(f"    最大值: {np.max(value)}")
                        print(f"    均值: {np.mean(value):.6f}")
                        print(f"    标准差: {np.std(value):.6f}")
            else:
                # 非数组类型，尝试显示内容
                print(f"    内容: {value}")
            
            print()  # 空行分隔
            
    except Exception as e:
        print(f"  ❌ 读取文件时出错: {type(e).__name__}: {e}")


def main():
    """主函数：扫描目录并检查所有 .mat 文件"""
    # 目标路径
    target_dir = r"I:\Pai\课程作业\课程作业"
    
    print(f"正在扫描目录: {target_dir}")
    print(f"查找所有 .mat 文件...\n")
    
    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"❌ 错误: 目录不存在: {target_dir}")
        print("请确认路径是否正确，或者修改脚本中的 target_dir 变量")
        return
    
    # 查找所有 .mat 文件
    mat_files = glob.glob(os.path.join(target_dir, "*.mat"))
    
    if not mat_files:
        print(f"⚠ 在目录 {target_dir} 中未找到任何 .mat 文件")
        return
    
    print(f"找到 {len(mat_files)} 个 .mat 文件\n")
    
    # 逐个检查文件结构
    for mat_file in sorted(mat_files):
        inspect_mat_structure(mat_file)
    
    print(f"\n{'='*80}")
    print(f"检查完成！共处理 {len(mat_files)} 个文件")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
