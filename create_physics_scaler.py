# -*- coding: utf-8 -*-
"""
create_physics_scaler.py (修正版)
创建并保存物理特征scaler，特征工程与训练代码完全一致
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    """
    特征工程（与训练代码 main.py 中完全一致）
    输入：DataFrame，必须包含 'year', 'doy', 'hour', 'BX', 'BY', 'BZ', 'SW_speed', 'Dst', 'ap', 'f107', 'AE'
    返回：添加所有特征后的DataFrame
    """
    # 1. 基础时间特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    
    # 2. 周期特征（太阳活动周期约11年）
    df['year_sin'] = np.sin(2 * np.pi * (df['year'] - 2019) / 11)
    df['year_cos'] = np.cos(2 * np.pi * (df['year'] - 2019) / 11)
    
    # 3. 组合特征
    df['B_total'] = np.sqrt(df['BX']**2 + df['BY']**2 + df['BZ']**2)
    df['vBs'] = df['SW_speed'] * df['BZ'].clip(upper=0)
    
    # 4. 滞后特征（shift后会有NaN，后续统一填充）
    for col in ['BX', 'BY', 'BZ', 'Dst', 'f107', 'AE']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_lag3'] = df[col].shift(3)
    
    # 5. 滚动统计
    for col in ['Dst', 'ap', 'AE', 'f107']:
        df[f'{col}_mean3'] = df[col].rolling(window=3, min_periods=1).mean()
        df[f'{col}_mean6'] = df[col].rolling(window=6, min_periods=1).mean()
        df[f'{col}_std3'] = df[col].rolling(window=3, min_periods=1).std()
    
    # 6. 差分特征
    for col in ['Dst', 'ap', 'AE']:
        df[f'{col}_diff1'] = df[col].diff(1)
        df[f'{col}_diff3'] = df[col].diff(3)
    
    # 统一填充NaN（与训练代码一致：先前向填充，再后向填充，最后用0填充）
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

def create_physics_scaler():
    """创建物理特征scaler"""
    print("="*80)
    print("创建物理特征scaler")
    print("="*80)
    
    # 物理文件路径（与训练代码完全相同）
    physics_files = {
        2019: r"F:\wenjian\2019_231-2019_365.xlsx",
        2020: r"F:\wenjian\2020_001-2020_366.xlsx",
        2021: r"F:\wenjian\2021_001-2021_365.xlsx",
        2022: r"F:\wenjian\2022_001-2022_365.xlsx",
        2023: r"F:\wenjian\2023_001-2023_365.xlsx",
        2024: r"F:\wenjian\2024_001-2024_366.xlsx",
        2025: r"F:\wenjian\2025_001-2025_131.xlsx"   # 注意：与训练代码一致，只有131天
    }
    
    all_data = []
    
    for year, file_path in physics_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            continue
        
        print(f"加载 {year} 年数据...")
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
        
        # 生成时间索引（从该年1月1日0时开始，逐小时）
        start_date = pd.Timestamp(f'{year}-01-01 00:00:00')
        dates = pd.date_range(start=start_date, periods=len(df), freq='H')
        
        # 提取物理参数（确保列顺序正确）
        # 根据训练代码，列索引为：
        # 0:year, 1:doy, 2:hour, 3:BX, 4:BY, 5:BZ, 6:SW_speed, 7:Dst, 8:ap, 9:f107, 10:AE
        physics_dict = {
            'BX': df[3].values.astype(np.float32),
            'BY': df[4].values.astype(np.float32),
            'BZ': df[5].values.astype(np.float32),
            'SW_speed': df[6].values.astype(np.float32),
            'Dst': df[7].values.astype(np.float32),
            'ap': df[8].values.astype(np.float32),
            'f107': df[9].values.astype(np.float32),
            'AE': df[10].values.astype(np.float32) if df.shape[1] > 10 else np.zeros(len(df))
        }
        
        df_physics = pd.DataFrame(physics_dict, index=dates)
        # 填充原始数据中可能存在的NaN
        df_physics = df_physics.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 添加 year, doy, hour
        df_physics['year'] = df_physics.index.year
        df_physics['doy'] = df_physics.index.dayofyear
        df_physics['hour'] = df_physics.index.hour
        
        # 特征工程
        df_physics = feature_engineering(df_physics)
        
        all_data.append(df_physics)
        print(f"  {year}年特征数: {df_physics.shape[1]}")
    
    if not all_data:
        print("❌ 没有加载到任何数据")
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data)
    print(f"\n合并后总数据量: {len(combined_df)} 小时")
    print(f"特征数量: {combined_df.shape[1]}")
    
    # 特征名称列表
    feature_names = list(combined_df.columns)
    print(f"\n特征名称 (共{len(feature_names)}个):")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i+1:2d}. {name}")
    print(f"  ...")
    for i, name in enumerate(feature_names[-10:], len(feature_names)-9):
        print(f"  {i:2d}. {name}")
    
    # 创建并拟合scaler
    scaler = StandardScaler()
    scaler.fit(combined_df.values)
    
    print(f"\n✅ scaler拟合完成:")
    print(f"   均值数量: {len(scaler.mean_)}")
    print(f"   标准差数量: {len(scaler.scale_)}")
    
    # 保存scaler
    save_dir = r"E:\daima\2026.3.27"   # 可根据需要修改
    os.makedirs(save_dir, exist_ok=True)
    
    scaler_path = os.path.join(save_dir, 'physics_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ scaler已保存到: {scaler_path}")
    
    # 保存特征名称
    feature_names_path = os.path.join(save_dir, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✅ 特征名称已保存到: {feature_names_path}")
    
    return scaler, feature_names

if __name__ == "__main__":
    create_physics_scaler()