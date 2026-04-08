"""
电离层重构模型：仅基于过去4小时空间环境指数预测当前三维电子密度分布
输入：过去4个时间步的物理特征 (B, 4, F)
输出：当前时间步的电离层密度 (B, nalt, nlat, nlon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import gc
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================ 数据缓存管理 ============================
class DataCacheManager:
    """缓存管理器，确保数据可靠"""
    def __init__(self, cache_dir='./data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_info_path(self, cache_name):
        return os.path.join(self.cache_dir, f"{cache_name}_info.json")

    def is_cache_valid(self, cache_name, expected_shape):
        info_path = self.get_cache_info_path(cache_name)
        data_path = os.path.join(self.cache_dir, f"{cache_name}.dat")
        if not os.path.exists(info_path) or not os.path.exists(data_path):
            return False
        try:
            with open(info_path, 'r') as f:
                cache_info = json.load(f)
            if tuple(cache_info['shape']) != expected_shape:
                return False
            return True
        except Exception:
            return False

    def save_cache_info(self, cache_name, shape, mean=None, std=None):
        info = {
            'shape': list(shape),
            'created_at': datetime.now().isoformat(),
            'mean': float(mean) if mean is not None else None,
            'std': float(std) if std is not None else None
        }
        with open(self.get_cache_info_path(cache_name), 'w') as f:
            json.dump(info, f, indent=2)

    def load_cache_info(self, cache_name):
        with open(self.get_cache_info_path(cache_name), 'r') as f:
            return json.load(f)

# ============================ 新数据集类 ============================
class PhysicsToIonosphereDataset(Dataset):
    """仅使用物理特征（过去4小时窗口）预测当前电离层密度"""
    def __init__(self, ionosphere_data, physics_data, window_size=4, step=1):
        """
        Args:
            ionosphere_data: (T, H, L, W) 归一化电离层密度（memmap或numpy）
            physics_data:    (T, F)       归一化物理特征（numpy）
            window_size:     输入窗口长度（小时），默认4
            step:            采样步长（通常为1）
        """
        self.window_size = window_size
        self.step = step
        self.ionosphere = ionosphere_data
        self.physics = physics_data

        # 有效样本索引：至少需要 window_size 个历史物理数据
        self.indices = np.arange(window_size - 1, len(physics_data), step)
        print(f"数据集创建完成: 总样本数 = {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end_idx = self.indices[idx]                     # 当前时刻索引
        start_idx = end_idx - self.window_size + 1

        # 输入：过去 window_size 个时刻的物理特征 (window_size, F)
        physics_seq = self.physics[start_idx:end_idx+1].copy()
        # 目标：当前时刻的电离层密度 (H, L, W)
        target = self.ionosphere[end_idx].copy()

        # 转换为 Tensor
        physics_seq = torch.FloatTensor(physics_seq.astype(np.float32))
        target = torch.FloatTensor(target.astype(np.float32))

        return physics_seq, target

# ============================ 新模型定义 ============================
class PhysicsToIonosphereModel(nn.Module):
    """物理特征编码器 + 三维网格解码器"""
    def __init__(self, physics_feature_dim, nalt=51, nlat=73, nlon=72, hidden_dim=128):
        super().__init__()
        self.nalt = nalt
        self.nlat = nlat
        self.nlon = nlon

        # LSTM 编码器（处理时间窗口）
        self.lstm = nn.LSTM(
            input_size=physics_feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 解码器：将最后一个时间步的隐状态映射到三维网格
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, nalt * nlat * nlon)
        )

    def forward(self, physics_seq):
        """
        Args:
            physics_seq: (B, T, F)
        Returns:
            iono: (B, nalt, nlat, nlon)
        """
        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(physics_seq)   # h_n: (num_layers, B, hidden_dim)
        last_hidden = h_n[-1]                            # (B, hidden_dim)

        # 解码
        x = self.decoder(last_hidden)                    # (B, nalt*nlat*nlon)
        return x.view(-1, self.nalt, self.nlat, self.nlon)

# ============================ 数据加载函数（保留原有逻辑） ============================
def load_real_data_memory_efficient(folder_paths=None):
    """加载真实的电离层数据和空间环境指数（内存优化版）- 支持多文件夹"""
    print("\n" + "="*60)
    print("开始加载数据（内存优化版）")
    print("="*60)

    cache_manager = DataCacheManager()

    try:
        # 1. 加载空间环境指数 - 使用正确的列索引
        print("1. 加载空间环境指数...")

        # 直接列出所有文件
        excel_files = [
            r"F:\wenjian\2019_231-2019_365.xlsx",
            r"F:\wenjian\2020_001-2020_366.xlsx",
            r"F:\wenjian\2021_001-2021_365.xlsx",
            r"F:\wenjian\2022_001-2022_365.xlsx",
            r"F:\wenjian\2023_001-2023_365.xlsx",
            r"F:\wenjian\2024_001-2024_366.xlsx",
            r"F:\wenjian\2025_001-2025_131.xlsx"
        ]

        # 合并所有文件的数据
        all_dfs = []
        all_time_indices = []

        for file_path in excel_files:
            df = pd.read_excel(file_path, header=None, engine='openpyxl')

            # 提取时间信息
            years = df[0].values      # 第1列：年份
            doys = df[1].values       # 第2列：年积日
            hours = df[2].values      # 第3列：小时

            # 创建时间索引
            file_times = []
            for year, doy, hour in zip(years, doys, hours):
                if pd.isna(year) or pd.isna(doy) or pd.isna(hour):
                    continue
                base_date = datetime(int(year), 1, 1)
                date = base_date + timedelta(days=int(doy) - 1)
                timestamp = date + timedelta(hours=int(hour))
                file_times.append(timestamp)

            all_time_indices.extend(file_times)
            all_dfs.append(df)

        # 合并所有DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # 确保时间索引长度匹配
        if len(all_time_indices) != len(combined_df):
            min_len = min(len(all_time_indices), len(combined_df))
            all_time_indices = all_time_indices[:min_len]
            combined_df = combined_df.iloc[:min_len]

        # 提取空间环境参数
        physics_data = {}
        if combined_df.shape[1] >= 11:
            physics_data = {
                'year': combined_df[0].values,
                'doy': combined_df[1].values,
                'hour': combined_df[2].values,
                'BX': combined_df[3].values,
                'BY': combined_df[4].values,
                'BZ': combined_df[5].values,
                'SW_speed': combined_df[6].values,
                'Dst': combined_df[7].values,
                'ap': combined_df[8].values,
                'f107': combined_df[9].values,
                'AE': combined_df[10].values
            }

        # 创建带正确时间索引的DataFrame
        physics_df = pd.DataFrame(physics_data, index=all_time_indices)

        print(f"  时间范围: {physics_df.index[0]} 到 {physics_df.index[-1]}")
        print(f"  总小时数: {len(physics_df)}")
        print(f"  年份分布: {sorted(set(physics_df['year'].values))}")

        # 2. 特征工程
        print("2. 特征工程...")
        features = physics_df.copy()

        # 基础时间特征
        features['hour_sin'] = np.sin(2 * np.pi * physics_df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * physics_df.index.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * physics_df.index.dayofyear / 365.25)
        features['day_cos'] = np.cos(2 * np.pi * physics_df.index.dayofyear / 365.25)

        # 添加周期特征（太阳活动周期约11年）
        features['year_sin'] = np.sin(2 * np.pi * (physics_df.index.year - 2019) / 11)
        features['year_cos'] = np.cos(2 * np.pi * (physics_df.index.year - 2019) / 11)

        # 组合特征
        features['B_total'] = np.sqrt(features['BX']**2 + features['BY']**2 + features['BZ']**2)
        features['vBs'] = features['SW_speed'] * features['BZ'].clip(upper=0)

        # 滞后特征
        for col in ['BX', 'BY', 'BZ', 'Dst', 'f107', 'AE']:
            features[f'{col}_lag1'] = features[col].shift(1)
            features[f'{col}_lag2'] = features[col].shift(2)
            features[f'{col}_lag3'] = features[col].shift(3)

        # 滚动统计
        for col in ['Dst', 'ap', 'AE', 'f107']:
            features[f'{col}_mean3'] = features[col].rolling(window=3, min_periods=1).mean()
            features[f'{col}_mean6'] = features[col].rolling(window=6, min_periods=1).mean()
            features[f'{col}_std3'] = features[col].rolling(window=3, min_periods=1).std()

        # 差分特征
        for col in ['Dst', 'ap', 'AE']:
            features[f'{col}_diff1'] = features[col].diff(1)
            features[f'{col}_diff3'] = features[col].diff(3)

        # 填充缺失值
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 归一化
        scaler = StandardScaler()
        physics_array = scaler.fit_transform(features.values)

        print(f"  物理特征数量: {physics_array.shape[1]}")
        print(f"  特征列示例: {list(features.columns[:15])}...")

        # 3. 加载电离层数据 - 根据文件名解析时间
        print("3. 加载电离层数据...")

        if folder_paths is None:
            folder_paths = [r"F:\wenjian\2019", r"F:\wenjian\2020", r"F:\wenjian\2021",
                           r"F:\wenjian\2022", r"F:\wenjian\2023", r"F:\wenjian\2024",
                           r"F:\wenjian\2025"]
        elif isinstance(folder_paths, str):
            folder_paths = [folder_paths]

        all_nc_files = []
        nc_file_times = []

        for folder_path in folder_paths:
            if os.path.exists(folder_path):
                nc_files = sorted(glob(os.path.join(folder_path, "*.nc")))
                print(f"  文件夹 {folder_path}: 找到 {len(nc_files)} 个NC文件")

                for nc_file in nc_files:
                    filename = os.path.basename(nc_file)
                    name_without_ext = filename.replace('.nc', '')
                    parts = name_without_ext.split('_')

                    # 文件名格式: GIS_Ne_IRI_RO_GPS_2019_231_00.nc
                    if len(parts) >= 8:
                        try:
                            year = int(parts[5])
                            doy = int(parts[6])
                            hour = int(parts[7])

                            base_date = datetime(year, 1, 1)
                            date = base_date + timedelta(days=doy - 1)
                            timestamp = date + timedelta(hours=hour)

                            nc_file_times.append(timestamp)
                            all_nc_files.append(nc_file)
                        except (ValueError, IndexError) as e:
                            print(f"  警告: 解析文件 {filename} 时出错: {e}")
                            continue
            else:
                print(f"  警告: 文件夹不存在 {folder_path}")

        if not all_nc_files:
            raise FileNotFoundError(f"在指定的文件夹中未找到有效的NC文件: {folder_paths}")

        print(f"\n  总共找到 {len(all_nc_files)} 个有效的NC文件")
        print(f"  电离层数据时间范围: {min(nc_file_times)} 到 {max(nc_file_times)}")

        # 4. 按时间排序NC文件
        print("\n4. 按时间排序NC文件...")
        sorted_pairs = sorted(zip(nc_file_times, all_nc_files), key=lambda x: x[0])
        sorted_times, sorted_files = zip(*sorted_pairs)
        sorted_times = list(sorted_times)
        sorted_files = list(sorted_files)

        print(f"  排序后时间范围: {sorted_times[0]} 到 {sorted_times[-1]}")

        # 5. 验证时间对齐
        print("\n5. 验证时间对齐...")
        physics_start = physics_df.index[0]
        physics_end = physics_df.index[-1]
        iono_start = sorted_times[0]
        iono_end = sorted_times[-1]

        print(f"  物理指数时间范围: {physics_start} - {physics_end}")
        print(f"  电离层数据时间范围: {iono_start} - {iono_end}")

        overlap_start = max(physics_start, iono_start)
        overlap_end = min(physics_end, iono_end)

        if overlap_start <= overlap_end:
            print(f"  ✅ 时间重叠范围: {overlap_start} - {overlap_end}")
            overlap_hours = (overlap_end - overlap_start).days * 24 + (overlap_end - overlap_start).seconds // 3600 + 1
            print(f"  重叠小时数: {overlap_hours}")
        else:
            print(f"  ⚠️ 警告: 时间范围不重叠！")

        # 6. 读取示例文件获取形状
        sample_file = sorted_files[0]
        print(f"\n6. 读取示例文件: {os.path.basename(sample_file)}")
        with xr.open_dataset(sample_file) as sample_ds:
            if 'Ne' in sample_ds:
                sample_shape = sample_ds['Ne'].shape
                print(f"  单样本形状: {sample_shape}")
                if len(sample_shape) == 3:
                    nalt, nlat, nlon = sample_shape
                    print(f"  高度层数: {nalt}, 纬度格点数: {nlat}, 经度格点数: {nlon}")
                else:
                    raise ValueError(f"Ne数据维度不正确: {sample_shape}")
            else:
                raise ValueError("NC文件中没有'Ne'变量")

        total_time_steps = len(sorted_files)
        full_shape = (total_time_steps,) + sample_shape
        print(f"  总数据形状: {full_shape}")

        # 7. 原始数据缓存
        print("\n7. 检查/创建原始数据缓存...")
        raw_cache_name = "ionosphere_raw"
        raw_cache_path = os.path.join(cache_manager.cache_dir, f"{raw_cache_name}.dat")

        if cache_manager.is_cache_valid(raw_cache_name, full_shape):
            print(f"  使用已缓存的原始数据: {raw_cache_path}")
            ionosphere_raw = np.memmap(raw_cache_path, dtype=np.float32, mode='r', shape=full_shape)
        else:
            print(f"  创建新的原始数据缓存...")
            if os.path.exists(raw_cache_path):
                os.remove(raw_cache_path)
            ionosphere_raw = np.memmap(raw_cache_path, dtype=np.float32, mode='w+', shape=full_shape)

            chunk_size = 10
            for chunk_start in range(0, total_time_steps, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_time_steps)
                chunk_files = sorted_files[chunk_start:chunk_end]

                chunk_data = []
                for file_path in chunk_files:
                    try:
                        with xr.open_dataset(file_path) as ds:
                            if 'Ne' in ds:
                                ne_data = ds['Ne'].values.astype(np.float32)
                                ne_data = np.nan_to_num(ne_data, nan=0.0)
                                ne_data = np.clip(ne_data, 0, 1e7)
                                chunk_data.append(ne_data)
                            else:
                                chunk_data.append(np.zeros(sample_shape, dtype=np.float32))
                    except Exception as e:
                        print(f"  警告: 无法读取文件 {os.path.basename(file_path)}: {e}")
                        chunk_data.append(np.zeros(sample_shape, dtype=np.float32))

                if chunk_data:
                    chunk_array = np.array(chunk_data)
                    ionosphere_raw[chunk_start:chunk_end] = chunk_array

                if (chunk_start // chunk_size) % 10 == 0:
                    print(f"    已加载 {chunk_end}/{total_time_steps} 个文件")
                del chunk_data, chunk_array
                gc.collect()

            cache_manager.save_cache_info(raw_cache_name, full_shape)
            ionosphere_raw.flush()
            print("  原始数据缓存完成")

        # 8. 计算归一化统计量
        print("\n8. 计算归一化统计量...")
        stats_cache_name = "ionosphere_stats"
        stats_info_path = cache_manager.get_cache_info_path(stats_cache_name)

        if os.path.exists(stats_info_path):
            stats_info = cache_manager.load_cache_info(stats_cache_name)
            mean_val = stats_info.get('mean', 0)
            std_val = stats_info.get('std', 1)
            print(f"  使用缓存的统计量: mean={mean_val:.4e}, std={std_val:.4e}")
        else:
            print(f"  计算新的统计量（分块计算）...")
            chunk_size = 50
            total_sum = 0
            total_sq_sum = 0
            total_count = 0
            for i in range(0, total_time_steps, chunk_size):
                chunk_end = min(i + chunk_size, total_time_steps)
                chunk = ionosphere_raw[i:chunk_end].copy()
                chunk = np.clip(chunk, 0, 1e7)
                chunk_log = np.log1p(chunk)
                chunk_flat = chunk_log.flatten()
                total_sum += np.sum(chunk_flat)
                total_sq_sum += np.sum(chunk_flat**2)
                total_count += len(chunk_flat)
                del chunk, chunk_log, chunk_flat
                gc.collect()
            mean_val = total_sum / total_count
            std_val = np.sqrt(total_sq_sum / total_count - mean_val**2)
            cache_manager.save_cache_info(stats_cache_name, full_shape, mean_val, std_val)
            print(f"  计算完成: mean={mean_val:.4e}, std={std_val:.4e}")

        # 9. 创建归一化数据缓存
        print("\n9. 创建归一化数据缓存...")
        norm_cache_name = "ionosphere_normalized"
        norm_cache_path = os.path.join(cache_manager.cache_dir, f"{norm_cache_name}.dat")

        if cache_manager.is_cache_valid(norm_cache_name, full_shape):
            print(f"  使用已缓存的归一化数据")
            ionosphere_norm = np.memmap(norm_cache_path, dtype=np.float32, mode='r', shape=full_shape)
        else:
            print(f"  创建新的归一化缓存...")
            if os.path.exists(norm_cache_path):
                os.remove(norm_cache_path)
            ionosphere_norm = np.memmap(norm_cache_path, dtype=np.float32, mode='w+', shape=full_shape)

            chunk_size = 20
            for i in range(0, total_time_steps, chunk_size):
                chunk_end = min(i + chunk_size, total_time_steps)
                chunk = ionosphere_raw[i:chunk_end].copy()
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=1e7, neginf=-1e7)
                chunk = np.clip(chunk, 0, 1e7)
                chunk_log = np.log1p(chunk)
                chunk_norm = (chunk_log - mean_val) / (std_val + 1e-8)
                ionosphere_norm[i:chunk_end] = chunk_norm
                if (i // chunk_size) % 5 == 0:
                    print(f"    已归一化 {chunk_end}/{total_time_steps} 个时间步")
                del chunk, chunk_log, chunk_norm
                gc.collect()

            cache_manager.save_cache_info(norm_cache_name, full_shape)
            ionosphere_norm.flush()
            print("  归一化缓存完成")

        # 10. 数据对齐
        print("\n10. 数据对齐...")

        # 创建时间到索引的映射（电离层数据）
        iono_time_to_idx = {t: i for i, t in enumerate(sorted_times)}

        # 找到物理指数和电离层数据的共同时间点
        common_times = [t for t in physics_df.index if t in iono_time_to_idx]
        common_times.sort()

        print(f"  物理指数总点数: {len(physics_df)}")
        print(f"  电离层数据总点数: {len(sorted_times)}")
        print(f"  共同时间点数: {len(common_times)}")

        if len(common_times) == 0:
            raise ValueError("物理指数和电离层数据没有共同的时间点！")

        # 创建对齐后的数据缓存（保持为memmap）
        print("\n  创建对齐后的数据缓存...")
        aligned_cache_name = "ionosphere_aligned"
        aligned_cache_path = os.path.join(cache_manager.cache_dir, f"{aligned_cache_name}.dat")
        aligned_shape = (len(common_times),) + sample_shape

        if os.path.exists(aligned_cache_path):
            print(f"  使用已对齐的缓存数据: {aligned_cache_path}")
            ionosphere_norm_aligned = np.memmap(aligned_cache_path, dtype=np.float32, mode='r', shape=aligned_shape)

            # 同时对齐物理数据
            physics_time_to_idx = {t: i for i, t in enumerate(physics_df.index)}
            physics_array_aligned = []
            for t in common_times:
                physics_idx = physics_time_to_idx[t]
                physics_array_aligned.append(physics_array[physics_idx])
            physics_array_aligned = np.array(physics_array_aligned)
        else:
            print(f"  创建新的对齐缓存...")
            ionosphere_norm_aligned = np.memmap(aligned_cache_path, dtype=np.float32, mode='w+', shape=aligned_shape)

            physics_time_to_idx = {t: i for i, t in enumerate(physics_df.index)}
            physics_array_aligned = []

            chunk_size = 100
            for chunk_start in range(0, len(common_times), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(common_times))
                chunk_times = common_times[chunk_start:chunk_end]

                chunk_physics = []
                for t in chunk_times:
                    physics_idx = physics_time_to_idx[t]
                    chunk_physics.append(physics_array[physics_idx])

                chunk_ionosphere = []
                for t in chunk_times:
                    iono_idx = iono_time_to_idx[t]
                    chunk_ionosphere.append(ionosphere_norm[iono_idx])

                chunk_physics_array = np.array(chunk_physics)
                chunk_ionosphere_array = np.array(chunk_ionosphere)

                ionosphere_norm_aligned[chunk_start:chunk_end] = chunk_ionosphere_array
                physics_array_aligned.extend(chunk_physics)

                if (chunk_start // chunk_size) % 10 == 0:
                    print(f"    已处理 {chunk_end}/{len(common_times)} 个时间点")

                del chunk_physics, chunk_ionosphere, chunk_physics_array, chunk_ionosphere_array
                gc.collect()

            physics_array_aligned = np.array(physics_array_aligned)
            ionosphere_norm_aligned.flush()
            cache_manager.save_cache_info(aligned_cache_name, aligned_shape)
            print(f"  对齐缓存创建完成")

        print(f"  对齐后数据长度: {len(common_times)}")
        print(f"  对齐后时间范围: {common_times[0]} - {common_times[-1]}")
        print(f"  电离层数据形状: {ionosphere_norm_aligned.shape}")
        print(f"  物理数据形状: {physics_array_aligned.shape}")

        # 11. 数据验证
        print("\n11. 数据验证...")
        ionosphere_sample = ionosphere_norm_aligned[0]
        physics_sample = physics_array_aligned[0]

        print(f"  电离层数据范围: [{ionosphere_sample.min():.4f}, {ionosphere_sample.max():.4f}]")
        print(f"  电离层数据均值: {ionosphere_sample.mean():.4f}")
        print(f"  物理数据范围: [{physics_sample.min():.4f}, {physics_sample.max():.4f}]")

        if np.all(ionosphere_sample == 0):
            print("  ⚠️ 警告: 第一个样本全为0，可能数据有问题")
        if np.isnan(ionosphere_sample).any():
            raise ValueError("电离层数据包含NaN！")
        if np.isnan(physics_sample).any():
            raise ValueError("物理数据包含NaN！")

        print("✅ 数据加载完成！")
        print(f"   电离层数据缓存大小: {ionosphere_norm_aligned.nbytes / 1024**3:.2f} GB")
        print(f"   物理数据内存占用: {physics_array_aligned.nbytes / 1024**2:.1f} MB")
        print(f"   缓存位置: {cache_manager.cache_dir}")

        return ionosphere_norm_aligned, physics_array_aligned, features.columns.tolist()

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============================ 训练函数 ============================
def train_model(continue_training=False, checkpoint_path=None, additional_epochs=60):
    """训练模型主函数，支持继续训练"""
    print("\n" + "="*60)
    if continue_training:
        print("继续训练电离层重构模型（仅物理输入）")
    else:
        print("开始训练电离层重构模型（仅物理输入）")
    print("="*60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # 1. 加载数据
        print("加载数据...")
        ionosphere_data, physics_data, feature_names = load_real_data_memory_efficient()

        # 2. 创建数据集（过去4小时窗口）
        window_size = 4
        dataset = PhysicsToIonosphereDataset(ionosphere_data, physics_data, window_size=window_size)

        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print(f"\n数据集划分:")
        print(f"  训练集: {train_size} 个样本")
        print(f"  验证集: {val_size} 个样本")

        # 3. 数据加载器
        batch_size = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 4. 模型
        physics_feature_dim = physics_data.shape[1]
        model = PhysicsToIonosphereModel(physics_feature_dim=physics_feature_dim)
        model = model.to(device)

        # 5. 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.MSELoss()

        # 6. 训练参数
        start_epoch = 0
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0

        # 7. 继续训练：加载检查点
        if continue_training:
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"\n加载检查点: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                start_epoch = checkpoint.get('epoch', 0)
                print(f"  起始轮次: {start_epoch}, 最佳验证损失: {best_val_loss:.6f}")
            else:
                print(f"❌ 检查点文件不存在: {checkpoint_path}，将开始新的训练")

        # 8. 设置总轮次
        if continue_training:
            num_epochs = start_epoch + additional_epochs
            print(f"\n继续训练: 从第 {start_epoch} 轮开始，共训练 {num_epochs} 轮")
        else:
            num_epochs = 50
            print(f"\n开始新训练，共 {num_epochs} 轮...")

        print(f"优化器: AdamW, 学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 9. 训练循环
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch + 1
            model.train()
            epoch_train_loss = 0

            for physics_seq, targets in train_loader:
                physics_seq = physics_seq.to(device)
                targets = targets.to(device)

                predictions = model(physics_seq)
                loss = criterion(predictions, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for physics_seq, targets in val_loader:
                    physics_seq = physics_seq.to(device)
                    targets = targets.to(device)
                    predictions = model(physics_seq)
                    loss = criterion(predictions, targets)
                    epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # 学习率调整
            scheduler.step(avg_val_loss)

            # 打印进度
            if current_epoch % 5 == 0 or current_epoch == 1:
                print(f"Epoch [{current_epoch:03d}/{num_epochs}] - "
                      f"训练损失: {avg_train_loss:.6f}, "
                      f"验证损失: {avg_val_loss:.6f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # 模型保存策略
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # 保存最佳模型
                checkpoint_data = {
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'feature_names': feature_names,
                    'physics_feature_dim': physics_feature_dim,
                    'window_size': window_size,
                    'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                torch.save(checkpoint_data, 'best_physics_to_iono.pth')
                print(f"  ✅ 保存最佳模型 (验证损失: {best_val_loss:.6f})")

                # 每隔25轮额外保存一个检查点
                if current_epoch % 25 == 0:
                    torch.save(checkpoint_data, f"checkpoint_epoch_{current_epoch}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发，在 {current_epoch} 轮停止训练")
                    break

        # 10. 训练完成后保存最终模型
        final_checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'feature_names': feature_names,
            'physics_feature_dim': physics_feature_dim,
            'window_size': window_size,
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_final': True
        }
        torch.save(final_checkpoint, 'final_physics_to_iono.pth')
        print(f"\n✅ 最终模型已保存: final_physics_to_iono.pth")

        # 11. 训练完成信息
        print(f"\n训练完成!")
        print(f"起始轮次: {start_epoch}")
        print(f"最终轮次: {current_epoch}")
        print(f"总训练轮次: {len(train_losses)}")
        print(f"最佳验证损失: {best_val_loss:.6f}")

        # 12. 可视化训练过程
        visualize_training_progress(train_losses, val_losses)

        # 13. 在验证集上可视化预测结果（采样少量样本）
        visualize_predictions(model, val_dataset, device, num_samples=3)

        # 14. 流式计算验证集指标
        print("\n计算验证集性能指标...")
        model.eval()
        total_mae = 0.0
        total_rmse = 0.0
        total_samples = 0

        with torch.no_grad():
            for physics_seq, targets in val_loader:
                physics_seq = physics_seq.to(device)
                targets = targets.to(device)
                predictions = model(physics_seq)

                batch_mae = F.l1_loss(predictions, targets).item()
                batch_rmse = torch.sqrt(F.mse_loss(predictions, targets)).item()

                batch_size = targets.size(0)
                total_mae += batch_mae * batch_size
                total_rmse += batch_rmse * batch_size
                total_samples += batch_size

        avg_mae = total_mae / total_samples
        avg_rmse = total_rmse / total_samples

        print(f"验证集性能（归一化尺度）:")
        print(f"  MAE: {avg_mae:.6f}")
        print(f"  RMSE: {avg_rmse:.6f}")

        return model, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'mae_norm': avg_mae,
            'rmse_norm': avg_rmse,
            'feature_names': feature_names,
            'start_epoch': start_epoch,
            'final_epoch': current_epoch,
            'total_epochs': len(train_losses)
        }

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================ 可视化函数 ============================
def visualize_training_progress(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('MSE损失')
    plt.title('训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.yscale('log')
    plt.xlabel('轮次')
    plt.ylabel('MSE损失 (对数尺度)')
    plt.title('训练过程 (对数尺度)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress_physics_to_iono.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, val_dataset, device, num_samples=3):
    """可视化模型在验证集上的预测结果（选择一个高度层）"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    for i in range(min(num_samples, len(val_dataset))):
        physics_seq, target = val_dataset[i]
        physics_seq = physics_seq.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(physics_seq).cpu().squeeze().numpy()
        target = target.numpy()

        # 选择一个典型高度层（例如中间层 25）
        height_idx = 25

        ax1 = axes[i, 0]
        im1 = ax1.imshow(pred[height_idx], cmap='jet', aspect='auto')
        ax1.set_title(f"预测 (高度层 {height_idx})")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[i, 1]
        im2 = ax2.imshow(target[height_idx], cmap='jet', aspect='auto')
        ax2.set_title(f"真实 (高度层 {height_idx})")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = axes[i, 2]
        error = pred[height_idx] - target[height_idx]
        vmax = max(abs(error.min()), abs(error.max()))
        im3 = ax3.imshow(error, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax3.set_title(f"误差 (高度层 {height_idx})")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle("验证集预测示例 (归一化尺度)")
    plt.tight_layout()
    plt.savefig('prediction_examples_physics_to_iono.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================ 继续训练交互 ============================
def continue_training_interactive():
    """交互式继续训练"""
    print("\n" + "="*60)
    print("继续训练选项")
    print("="*60)

    # 检查可用的模型文件
    model_files = []
    if os.path.exists('best_physics_to_iono.pth'):
        model_files.append(('best_physics_to_iono.pth', '最佳模型'))
    if os.path.exists('final_physics_to_iono.pth'):
        model_files.append(('final_physics_to_iono.pth', '最终模型'))
    checkpoint_files = glob('checkpoint_epoch_*.pth')
    for file in checkpoint_files:
        model_files.append((file, f'检查点 {os.path.basename(file)}'))

    if not model_files:
        print("❌ 未找到可用的模型文件，请先进行训练")
        return None, None

    print("可用的模型文件:")
    for i, (file_path, desc) in enumerate(model_files):
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            epochs = checkpoint.get('epoch', '未知')
            loss = checkpoint.get('best_val_loss', '未知')
            print(f"  {i+1}. {desc}")
            print(f"     文件: {file_path}, 轮次: {epochs}, 最佳损失: {loss:.6f if isinstance(loss, float) else loss}")
        except:
            print(f"  {i+1}. {desc} (文件可能损坏)")

    # 选择模型文件
    while True:
        try:
            choice = input(f"\n请选择要加载的模型 (1-{len(model_files)}) 或输入 'q' 退出: ").strip()
            if choice.lower() == 'q':
                return None, None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected_file = model_files[choice_idx][0]
                break
            else:
                print(f"❌ 请输入 1 到 {len(model_files)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")

    # 询问继续训练的轮数
    while True:
        try:
            additional_epochs = input("请输入要增加的训练轮数 (默认50): ").strip()
            if not additional_epochs:
                additional_epochs = 50
                break
            additional_epochs = int(additional_epochs)
            if additional_epochs > 0:
                break
            else:
                print("❌ 轮数必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")

    print(f"\n开始继续训练:")
    print(f"  加载模型: {selected_file}")
    print(f"  增加轮数: {additional_epochs}")

    return train_model(continue_training=True, checkpoint_path=selected_file, additional_epochs=additional_epochs)

# ============================ 主函数 ============================
def main():
    """主函数"""
    print(f"设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n训练选项:")
    print("1. 开始新训练")
    print("2. 继续训练")
    print("3. 退出")

    while True:
        choice = input("请选择 (1-3): ").strip()
        if choice == "1":
            print("\n开始新训练...")
            model, training_info = train_model(continue_training=False)
            break
        elif choice == "2":
            model, training_info = continue_training_interactive()
            break
        elif choice == "3":
            print("\n退出程序")
            return
        else:
            print("❌ 无效选择，请重新输入")

    if model is not None:
        print("训练成功完成!")
        print(f"\n训练摘要:")
        print(f"  起始轮次: {training_info.get('start_epoch', 0)}")
        print(f"  最终轮次: {training_info.get('final_epoch', 0)}")
        print(f"  总训练轮次: {training_info.get('total_epochs', 0)}")
        print(f"  最佳验证损失: {training_info.get('best_val_loss', 0):.6f}")
        print(f"  归一化尺度MAE: {training_info.get('mae_norm', 0):.6f}")
        print(f"  归一化尺度RMSE: {training_info.get('rmse_norm', 0):.6f}")

        feature_names = training_info.get('feature_names', [])
        if feature_names:
            print(f"\n使用的物理特征 ({len(feature_names)}个):")
            for i, name in enumerate(feature_names[:10]):
                print(f"  {i+1}. {name}")
            if len(feature_names) > 10:
                print(f"  ... 还有 {len(feature_names)-10} 个特征")

        print(f"\n模型文件:")
        print(f"  best_physics_to_iono.pth - 最佳模型")
        print(f"  final_physics_to_iono.pth - 最终模型")
        print(f"  training_progress_physics_to_iono.png - 训练过程图")
        print(f"  prediction_examples_physics_to_iono.png - 预测示例图")
    else:
        print("\n❌ 训练失败!")

if __name__ == "__main__":
    main()