"""
验证脚本：基于空间环境指数预测的电离层TEC与真实值对比
新增功能：
1. 短时验证（原有功能）
2. 长时无原重构（纯物理驱动，逐日平均，支持平滑）
"""
import gc
import os
import numpy as np
from scipy import stats
import xarray as xr
import pandas as pd
import torch
import torch.nn as nn
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import json
import pickle
from scipy.io import loadmat
from glob import glob
from sklearn.preprocessing import StandardScaler
#注意nc文件的起始经度是-180，终止是180度
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ============================ TEC异常阈值 ============================
TEC_MAX_REASONABLE = 80  # TECU，超过此值视为异常
TEC_MIN_REASONABLE = 0   # TECU，低于此值视为异常（通常不会为负）

# ============================ 模型定义 ============================
class PhysicsToIonosphereModel(nn.Module):
    def __init__(self, physics_feature_dim, nalt=51, nlat=73, nlon=72, hidden_dim=128):
        super().__init__()
        self.nalt = nalt
        self.nlat = nlat
        self.nlon = nlon
        self.lstm = nn.LSTM(
            input_size=physics_feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, nalt * nlat * nlon)
        )

    def forward(self, physics_seq):
        lstm_out, (h_n, c_n) = self.lstm(physics_seq)
        last_hidden = h_n[-1]
        x = self.decoder(last_hidden)
        return x.view(-1, self.nalt, self.nlat, self.nlon)

# ============================ 特征工程 ============================
def feature_engineering(df):
    """特征工程（与训练代码完全一致）"""
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    df['year_sin'] = np.sin(2 * np.pi * (df['year'] - 2019) / 11)
    df['year_cos'] = np.cos(2 * np.pi * (df['year'] - 2019) / 11)
    df['B_total'] = np.sqrt(df['BX']**2 + df['BY']**2 + df['BZ']**2)
    df['vBs'] = df['SW_speed'] * df['BZ'].clip(upper=0)
    for col in ['BX', 'BY', 'BZ', 'Dst', 'f107', 'AE']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_lag3'] = df[col].shift(3)
    for col in ['Dst', 'ap', 'AE', 'f107']:
        df[f'{col}_mean3'] = df[col].rolling(window=3, min_periods=1).mean()
        df[f'{col}_mean6'] = df[col].rolling(window=6, min_periods=1).mean()
        df[f'{col}_std3'] = df[col].rolling(window=3, min_periods=1).std()
    for col in ['Dst', 'ap', 'AE']:
        df[f'{col}_diff1'] = df[col].diff(1)
        df[f'{col}_diff3'] = df[col].diff(3)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df

# ============================ 数据加载 ============================
def load_physics_data(start_time, end_time, scaler, feature_names):
    """加载并标准化物理数据"""
    excel_files = [
        r"F:\wenjian\2019_231-2019_365.xlsx",
        r"F:\wenjian\2020_001-2020_366.xlsx",
        r"F:\wenjian\2021_001-2021_365.xlsx",
        r"F:\wenjian\2022_001-2022_365.xlsx",
        r"F:\wenjian\2023_001-2023_365.xlsx",
        r"F:\wenjian\2024_001-2024_366.xlsx",
        r"F:\wenjian\2025_001-2025_365.xlsx"
    ]
    all_dfs = []
    all_time_indices = []
    for file_path in excel_files:
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
        years = df[0].values
        doys = df[1].values
        hours = df[2].values
        file_times = []
        for year, doy, hour in zip(years, doys, hours):
            if pd.isna(year) or pd.isna(doy) or pd.isna(hour):
                continue
            base_date = datetime(int(year), 1, 1)
            date = base_date + timedelta(days=int(doy)-1)
            timestamp = date + timedelta(hours=int(hour))
            file_times.append(timestamp)
        all_time_indices.extend(file_times)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if len(all_time_indices) != len(combined_df):
        min_len = min(len(all_time_indices), len(combined_df))
        all_time_indices = all_time_indices[:min_len]
        combined_df = combined_df.iloc[:min_len]
    physics_dict = {
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
        'AE': combined_df[10].values if combined_df.shape[1] > 10 else np.zeros(len(combined_df))
    }
    physics_df = pd.DataFrame(physics_dict, index=all_time_indices)
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    physics_df = physics_df.loc[start_dt:end_dt].copy()
    if len(physics_df) == 0:
        raise ValueError(f"指定时间段 {start_time} 至 {end_time} 内没有物理数据")
    physics_feat = feature_engineering(physics_df)
    physics_feat = physics_feat[feature_names]
    physics_norm = scaler.transform(physics_feat.values)
    return physics_df, physics_norm

def load_iono_data(start_time, end_time, folder_paths=None):
    """加载电离层数据，自动识别高度、纬度、经度坐标，跳过损坏文件"""
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
            for nc_file in nc_files:
                filename = os.path.basename(nc_file)
                name_without_ext = filename.replace('.nc', '')
                parts = name_without_ext.split('_')
                if len(parts) >= 8:
                    try:
                        year = int(parts[5])
                        doy = int(parts[6])
                        hour = int(parts[7])
                        base_date = datetime(year, 1, 1)
                        date = base_date + timedelta(days=doy-1)
                        timestamp = date + timedelta(hours=hour)
                        nc_file_times.append(timestamp)
                        all_nc_files.append(nc_file)
                    except:
                        continue
    if not all_nc_files:
        return {}, None, None, None
    sorted_pairs = sorted(zip(nc_file_times, all_nc_files), key=lambda x: x[0])
    sorted_times, sorted_files = zip(*sorted_pairs)
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    iono_dict = {}
    height_coords = None
    lat_coords = None
    lon_coords = None
    coords_initialized = False
    for t, file in zip(sorted_times, sorted_files):
        if start_dt <= t <= end_dt:
            print(f"正在读取文件: {file}")   # 添加这一行
            # 跳过已知损坏文件
            bad_files = [
                r"F:\wenjian\2022\GIS_Ne_IRI_RO_GPS_2022_022_01.nc",
                
            ]
            if file in bad_files:
                print(f"跳过已知损坏文件: {file}")
                continue
            try:
                with xr.open_dataset(file, engine='netcdf4') as ds:
                    # ... 原有代码 ...
                    ne = ds['Ne'].values.astype(np.float32)
                    ne = np.nan_to_num(ne, nan=0.0)
                    ne = np.clip(ne, 0, 1e7)
                    iono_dict[t] = ne
                    if not coords_initialized:
                        # 高度坐标
                        candidate_heights = ['height', 'altitude', 'alt', 'h', 'level', 'z']
                        for name in candidate_heights:
                            if name in ds.coords:
                                height_coords = ds[name].values
                                print(f"高度坐标: {name}, 形状: {height_coords.shape}")
                                break
                        if height_coords is None:
                            for name in candidate_heights:
                                if name in ds.data_vars:
                                    height_coords = ds[name].values
                                    print(f"高度数据变量: {name}, 形状: {height_coords.shape}")
                                    break
                        if height_coords is None:
                            height_dim = ds['Ne'].dims[0]
                            if height_dim in ds:
                                height_coords = ds[height_dim].values
                                print(f"通过Ne的第一维 '{height_dim}' 获取高度")
                        if height_coords is None:
                            nalt = ds['Ne'].shape[0]
                            height_coords = np.linspace(100, 1100, nalt)
                            print(f"使用默认高度: 100-1100km, {nalt}层")
                        # 纬度坐标
                        candidate_lats = ['lat', 'latitude', 'Lat', 'Latitude', 'y']
                        for name in candidate_lats:
                            if name in ds.coords:
                                lat_coords = ds[name].values
                                print(f"纬度坐标: {name}, 形状: {lat_coords.shape}")
                                break
                        if lat_coords is None:
                            for name in candidate_lats:
                                if name in ds.data_vars:
                                    lat_coords = ds[name].values
                                    print(f"纬度数据变量: {name}, 形状: {lat_coords.shape}")
                                    break
                        if lat_coords is None:
                            lat_dim = ds['Ne'].dims[1]
                            if lat_dim in ds:
                                lat_coords = ds[lat_dim].values
                                print(f"通过Ne的第二维 '{lat_dim}' 获取纬度")
                        if lat_coords is None:
                            nlat = ds['Ne'].shape[1]
                            lat_coords = np.linspace(-90, 90, nlat)
                            print(f"使用默认纬度: -90°到90°, {nlat}点")
                        # 经度坐标
                        candidate_lons = ['lon', 'longitude', 'Lon', 'Longitude', 'x']
                        for name in candidate_lons:
                            if name in ds.coords:
                                lon_coords = ds[name].values
                                print(f"经度坐标: {name}, 形状: {lon_coords.shape}")
                                break
                        if lon_coords is None:
                            for name in candidate_lons:
                                if name in ds.data_vars:
                                    lon_coords = ds[name].values
                                    print(f"经度数据变量: {name}, 形状: {lon_coords.shape}")
                                    break
                        if lon_coords is None:
                            lon_dim = ds['Ne'].dims[2]
                            if lon_dim in ds:
                                lon_coords = ds[lon_dim].values
                                print(f"通过Ne的第三维 '{lon_dim}' 获取经度")
                        if lon_coords is None:
                            nlon = ds['Ne'].shape[2]
                            lon_coords = np.linspace(0, 360, nlon, endpoint=False)
                            print(f"使用默认经度: 0°到360°, {nlon}点")
                        coords_initialized = True
            except Exception as e:
                print(f"警告：跳过损坏文件 {file}: {e}")
                continue
    if not coords_initialized:
        print("错误：没有成功读取任何NC文件，无法获取坐标")
    return iono_dict, height_coords, lat_coords, lon_coords

# ============================ TEC计算 ============================
def compute_tec(ne_3d, height_coords):
    """沿高度梯形积分，返回全纬度TEC数组 (nlat, nlon)，单位TECU"""
    ne_m3 = ne_3d * 1e11
    height_m = height_coords * 1000.0
    diff_h = np.diff(height_m)
    tec_elm2 = np.sum((ne_m3[:-1] + ne_m3[1:]) / 2 * diff_h[:, np.newaxis, np.newaxis], axis=0)
    tec_tecu = tec_elm2 / 1e16
    return tec_tecu

def inverse_normalize_iono(norm_data, mean, std):
    """反归一化得到原始电子密度（单位：1e5 el/cm3）"""
    log1p = norm_data * std + mean
    return np.expm1(log1p)

def predict(model, physics_window_norm, mean, std, height_coords, device):
    """返回全纬度TEC数组 (nlat, nlon)，单位TECU"""
    with torch.no_grad():
        physics_tensor = torch.FloatTensor(physics_window_norm).unsqueeze(0).to(device)
        pred_norm = model(physics_tensor).cpu().squeeze().numpy()
    pred_raw = inverse_normalize_iono(pred_norm, mean, std)
    tec = compute_tec(pred_raw, height_coords)
    return tec

# ============================ 短时验证 ============================
def process_period(period_name, start_time, end_time, period_type, use_initial_iono=False):
    """处理单个时间段（短时验证）"""
    print(f"\n处理 {period_type} 期: {period_name} ({start_time} 至 {end_time})")
    print(f"  有原模式: {'启用' if use_initial_iono else '禁用'}")

    # 加载物理数据
    physics_df, physics_norm = load_physics_data(start_time, end_time, scaler, feature_names)
    # 加载电离层数据
    iono_dict, height_coords, lat_coords, lon_coords = load_iono_data(start_time, end_time)
    if height_coords is None:
        raise ValueError("未找到高度坐标")

    # 获取纬度数组
    if lat_coords is None:
        nlat = next(iter(iono_dict.values())).shape[1]
        lat_coords = np.linspace(-90, 90, nlat)
        print(f"使用默认纬度数组: 形状 {lat_coords.shape}")

    # 区域定义（可修改）
    regions = {
        'Global': (-90, 90),
        'Mid-Latitude (±45°)': (-45, 45),
        'North Mid-Latitude (30-75°)': (30, 75),
        'Equatorial (±15°)': (-15, 15)
    }

    # 为每个区域预先计算纬度索引切片
    region_slices = {}
    for region_name, (lat_min, lat_max) in regions.items():
        mask = (lat_coords >= lat_min) & (lat_coords <= lat_max)
        if not np.any(mask):
            print(f"警告: 区域 {region_name} 纬度范围 {lat_min}~{lat_max} 在数据中无覆盖")
            continue
        idx_start = np.where(mask)[0][0]
        idx_end = np.where(mask)[0][-1] + 1
        region_slices[region_name] = slice(idx_start, idx_end)
        print(f"区域 {region_name} 纬度范围 {lat_min}°~{lat_max}° 对应索引 {idx_start}~{idx_end-1}")

    # 时间对齐
    common_times = sorted(set(physics_df.index) & set(iono_dict.keys()))
    if len(common_times) == 0:
        print(f"⚠️ {period_name} 时段没有共同时间点，跳过")
        return None

    # 初始化结果字典
    region_results_std = {region_name: {'tec_true': [], 'tec_pred': []} for region_name in region_slices}
    region_results_init = {region_name: {'tec_true': [], 'tec_pred': []} for region_name in region_slices}
    timestamps = []
    anomaly_count_std = 0
    anomaly_count_init = 0

    window_size = 4  # 与训练时一致
    first_valid = True

    for t in common_times:
        idx = physics_df.index.get_loc(t)
        if idx < window_size - 1:
            continue
        physics_window = physics_norm[idx - window_size + 1 : idx + 1]
        if len(physics_window) != window_size:
            continue

        # 标准预测（纯物理）
        tec_full_std = predict(model, physics_window, mean, std, height_coords, device)

        # 有原预测（如果启用，第一个有效时间步使用真实值）
        if use_initial_iono and first_valid:
            ne_true = iono_dict[t]
            tec_full_init = compute_tec(ne_true, height_coords)
            first_valid = False
            print(f"  第一个有效时间步 {t} 使用真实值")
        else:
            tec_full_init = tec_full_std

        # 真实值
        ne_true = iono_dict[t]
        tec_true_full = compute_tec(ne_true, height_coords)

        # 为每个区域计算平均TEC
        for region_name, lat_slice in region_slices.items():
            tec_pred_std = np.mean(tec_full_std[lat_slice, :])
            tec_pred_init = np.mean(tec_full_init[lat_slice, :])
            tec_true = np.mean(tec_true_full[lat_slice, :])

            # 异常检测
            is_valid_std = (TEC_MIN_REASONABLE <= tec_pred_std <= TEC_MAX_REASONABLE)
            if not is_valid_std:
                anomaly_count_std += 1
                print(f"  ⚠️ 标准异常: {t.strftime('%Y-%m-%d %H:%M')} - {region_name}: "
                      f"预测TEC={tec_pred_std:.2f} TECU")

            is_valid_init = (TEC_MIN_REASONABLE <= tec_pred_init <= TEC_MAX_REASONABLE)
            if not is_valid_init:
                anomaly_count_init += 1
                print(f"  ⚠️ 有原异常: {t.strftime('%Y-%m-%d %H:%M')} - {region_name}: "
                      f"预测TEC={tec_pred_init:.2f} TECU")

            region_results_std[region_name]['tec_true'].append(tec_true)
            region_results_std[region_name]['tec_pred'].append(tec_pred_std if is_valid_std else np.nan)
            region_results_init[region_name]['tec_true'].append(tec_true)
            region_results_init[region_name]['tec_pred'].append(tec_pred_init if is_valid_init else np.nan)

        timestamps.append(t)

    # 打印异常统计
    if anomaly_count_std > 0:
        print(f"\n⚠️ {period_name} 标准模式发现 {anomaly_count_std} 次异常预测")
    if use_initial_iono and anomaly_count_init > 0:
        print(f"⚠️ {period_name} 有原模式发现 {anomaly_count_init} 次异常预测")

    return {
        'period_name': period_name,
        'period_type': period_type,
        'timestamps': timestamps,
        'region_results_std': region_results_std,
        'region_results_init': region_results_init if use_initial_iono else None,
        'anomaly_count_std': anomaly_count_std,
        'anomaly_count_init': anomaly_count_init if use_initial_iono else 0
    }

# ============================ 长时无原重构 ============================
def long_term_reconstruction(period_name, start_date, end_date, regions, save_prefix, smooth_window=3):
    """长时无原重构：逐日平均TEC对比，可选滑动平均平滑"""
    print(f"\n处理长时段: {period_name} ({start_date} 至 {end_date})")
    print(f"  平滑窗口: {smooth_window}天")

    # 加载物理数据
    physics_df, physics_norm = load_physics_data(start_date, end_date, scaler, feature_names)
    # 加载电离层数据
    iono_dict, height_coords, lat_coords, lon_coords = load_iono_data(start_date, end_date)
    if height_coords is None:
        raise ValueError("未找到高度坐标")

    # 获取纬度数组
    if lat_coords is None:
        nlat = next(iter(iono_dict.values())).shape[1]
        lat_coords = np.linspace(-90, 90, nlat)
        print(f"使用默认纬度数组: 形状 {lat_coords.shape}")

    # 为每个区域预先计算纬度索引切片
    region_slices = {}
    for region_name, (lat_min, lat_max) in regions.items():
        mask = (lat_coords >= lat_min) & (lat_coords <= lat_max)
        if not np.any(mask):
            print(f"警告: 区域 {region_name} 纬度范围 {lat_min}~{lat_max} 在数据中无覆盖")
            continue
        idx_start = np.where(mask)[0][0]
        idx_end = np.where(mask)[0][-1] + 1
        region_slices[region_name] = slice(idx_start, idx_end)
        print(f"区域 {region_name} 纬度范围 {lat_min}°~{lat_max}° 对应索引 {idx_start}~{idx_end-1}")

    # 时间对齐
    common_times = sorted(set(physics_df.index) & set(iono_dict.keys()))
    if len(common_times) == 0:
        print(f"⚠️ 时段 {start_date} 至 {end_date} 没有共同时间点，跳过")
        return

    # 初始化存储每日结果
    daily_data = {}  # key: date, value: dict with 'pred' and 'true' for each region (list of hourly values)
    window_size = 4

    # 逐小时预测
    for t in common_times:
        idx = physics_df.index.get_loc(t)
        if idx < window_size - 1:
            continue
        physics_window = physics_norm[idx - window_size + 1 : idx + 1]
        if len(physics_window) != window_size:
            continue

        tec_full_pred = predict(model, physics_window, mean, std, height_coords, device)
        tec_full_true = compute_tec(iono_dict[t], height_coords)

        date_key = t.date()
        if date_key not in daily_data:
            daily_data[date_key] = {'pred': {reg: [] for reg in region_slices},
                                    'true': {reg: [] for reg in region_slices}}

        for region_name, lat_slice in region_slices.items():
            tec_pred = np.mean(tec_full_pred[lat_slice, :])
            tec_true = np.mean(tec_full_true[lat_slice, :])
            # 小时异常值剔除：超出合理范围则丢弃该小时
            if TEC_MIN_REASONABLE <= tec_pred <= TEC_MAX_REASONABLE:
                daily_data[date_key]['pred'][region_name].append(tec_pred)
            if TEC_MIN_REASONABLE <= tec_true <= TEC_MAX_REASONABLE:
                daily_data[date_key]['true'][region_name].append(tec_true)

    # 计算每日平均
    daily_dates = []
    region_avg = {region_name: {'pred': [], 'true': []} for region_name in region_slices}
    for date in sorted(daily_data.keys()):
        daily_dates.append(date)
        for region_name in region_slices:
            pred_list = daily_data[date]['pred'][region_name]
            true_list = daily_data[date]['true'][region_name]
            if pred_list and true_list:
                region_avg[region_name]['pred'].append(np.mean(pred_list))
                region_avg[region_name]['true'].append(np.mean(true_list))
            else:
                region_avg[region_name]['pred'].append(np.nan)
                region_avg[region_name]['true'].append(np.nan)

    # 对日平均序列进行滑动平均平滑
    if smooth_window > 1:
        for region_name in region_slices:
            pred_series = np.array(region_avg[region_name]['pred'])
            true_series = np.array(region_avg[region_name]['true'])
            # 使用pandas的rolling mean，处理NaN
            pred_series_smoothed = pd.Series(pred_series).rolling(window=smooth_window, min_periods=1, center=True).mean().values
            true_series_smoothed = pd.Series(true_series).rolling(window=smooth_window, min_periods=1, center=True).mean().values
            region_avg[region_name]['pred_smooth'] = pred_series_smoothed
            region_avg[region_name]['true_smooth'] = true_series_smoothed

    # 绘图
    n_regions = len(region_slices)
    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for ax, (region_name, data) in zip(axes, region_avg.items()):
        # 使用平滑后的数据（如果smooth_window>1），否则使用原始日平均
        if smooth_window > 1:
            pred_plot = data['pred_smooth']
            true_plot = data['true_smooth']
        else:
            pred_plot = data['pred']
            true_plot = data['true']

        # 过滤NaN
        dates_valid = [d for d, p, t in zip(daily_dates, pred_plot, true_plot) if not np.isnan(p) and not np.isnan(t)]
        pred_valid = [p for p, t in zip(pred_plot, true_plot) if not np.isnan(p) and not np.isnan(t)]
        true_valid = [t for p, t in zip(pred_plot, true_plot) if not np.isnan(p) and not np.isnan(t)]

        ax.plot(dates_valid, true_valid, 'o-', label='Trained Data TEC', markersize=3, linewidth=1.5, color='blue')
        ax.plot(dates_valid, pred_valid, 'x--', label='PTIM TEC', markersize=3, linewidth=1.5, color='red')
        ax.set_title(f'{period_name} - {region_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Mean TEC (TECU)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        y_max_true = max(true_valid) if true_valid else 10
        y_max = min(y_max_true * 1.2, 80)
        ax.set_ylim(0, y_max)

    plt.tight_layout()
    save_name = f'long_term_{save_prefix}_{period_name.replace("-", "_").replace(" ", "_")}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"长时对比图已保存为 {save_name}")

def plot_error_statistics():
    """
    基于所有共同时间点（既有物理指数又有NC文件），按经度每30度为一个区域，
    计算各个区域平均TEC的误差（Trained Data - PTIM），并绘制误差分布直方图。
    每个区域生成一张独立的图片，并添加高斯拟合表达式。强制高斯拟合曲线的横轴范围为-25到25 TECU。
    """
    print("\n" + "="*60)
    print("生成误差分布直方图 - 按经度每30度分区")
    print("="*60)

    # 经度分区：-180 到 180，每30度一个区间，共12个区域
    lon_bins = np.arange(-180, 181, 30)
    lon_regions = []
    for i in range(len(lon_bins)-1):
        region_name = f"{lon_bins[i]:.0f}°~{lon_bins[i+1]:.0f}°E"
        lon_regions.append((region_name, lon_bins[i], lon_bins[i+1]))
    
    print(f"经度分区：共 {len(lon_regions)} 个区域")
    for name, lmin, lmax in lon_regions:
        print(f"  {name}")

    print('输入起始时间（格式 YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DD）：')
    start_date = str(input()).strip()
    print('输入终止时间：')
    end_date = str(input()).strip()
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    years = range(start_dt.year, end_dt.year + 1)
    window_size = 4

    region_errors = {name: [] for name, _, _ in lon_regions}

    height_coords_global = None
    lat_coords_global = None
    lon_coords_global = None

    for year in years:
        print(f"处理年份 {year}...")
        year_start = max(start_dt, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end_dt, pd.Timestamp(f"{year}-12-31"))
        if year_start > year_end:
            continue

        try:
            physics_df, physics_norm = load_physics_data(
                year_start.strftime('%Y-%m-%d %H:%M:%S'),
                year_end.strftime('%Y-%m-%d %H:%M:%S'),
                scaler, feature_names
            )
        except Exception as e:
            print(f"警告：加载 {year} 年物理数据失败: {e}")
            continue

        folder_path = os.path.join(r"F:\wenjian", str(year))
        if not os.path.exists(folder_path):
            print(f"警告：{year} 年文件夹不存在，跳过")
            continue

        nc_files = glob(os.path.join(folder_path, "*.nc"))
        if not nc_files:
            print(f"警告：{year} 年没有NC文件，跳过")
            continue

        file_times = []
        valid_files = []
        for file in nc_files:
            filename = os.path.basename(file)
            name_without_ext = filename.replace('.nc', '')
            parts = name_without_ext.split('_')
            if len(parts) >= 8:
                try:
                    year_f = int(parts[5])
                    doy = int(parts[6])
                    hour = int(parts[7])
                    base_date = datetime(year_f, 1, 1)
                    date = base_date + timedelta(days=doy-1)
                    timestamp = date + timedelta(hours=hour)
                    if year_start <= timestamp <= year_end:
                        file_times.append(timestamp)
                        valid_files.append(file)
                except:
                    continue
        if not valid_files:
            print(f"警告：{year} 年没有有效NC文件，跳过")
            continue

        sorted_pairs = sorted(zip(file_times, valid_files), key=lambda x: x[0])
        sorted_times, sorted_files = zip(*sorted_pairs)

        if height_coords_global is None:
            for t_file, file in zip(sorted_times, sorted_files):
                try:
                    with xr.open_dataset(file, engine='netcdf4') as ds:
                        candidate_heights = ['height', 'altitude', 'alt', 'h', 'level', 'z']
                        for name in candidate_heights:
                            if name in ds.coords:
                                height_coords_global = ds[name].values
                                break
                        if height_coords_global is None:
                            nalt = ds['Ne'].shape[0]
                            height_coords_global = np.linspace(100, 1100, nalt)
                        candidate_lats = ['lat', 'latitude', 'Lat', 'Latitude', 'y']
                        for name in candidate_lats:
                            if name in ds.coords:
                                lat_coords_global = ds[name].values
                                break
                        if lat_coords_global is None:
                            nlat = ds['Ne'].shape[1]
                            lat_coords_global = np.linspace(-90, 90, nlat)
                        if 'lon' in ds.coords:
                            lon_coords_global = ds['lon'].values
                        else:
                            nlon = ds['Ne'].shape[2]
                            lon_coords_global = np.linspace(0, 360, nlon, endpoint=False)
                        lon_coords_global = np.where(lon_coords_global > 180, lon_coords_global - 360, lon_coords_global)
                        break
                except Exception as e:
                    print(f"警告：读取文件 {file} 获取坐标失败: {e}")
                    continue
            if height_coords_global is None:
                print(f"错误：无法获取坐标，跳过 {year} 年")
                continue

        region_masks = {}
        for name, lmin, lmax in lon_regions:
            mask = (lon_coords_global >= lmin) & (lon_coords_global < lmax)
            if lmax == 180:
                mask = (lon_coords_global >= lmin) & (lon_coords_global <= lmax)
            if not np.any(mask):
                print(f"警告：区域 {name} 在数据中无经度覆盖")
                continue
            region_masks[name] = mask
            lon_indices = np.where(mask)[0]
            print(f"区域 {name} 经度索引范围: {lon_indices[0]}~{lon_indices[-1]}")

        for t, file in zip(sorted_times, sorted_files):
            if t not in physics_df.index:
                continue
            idx = physics_df.index.get_loc(t)
            if idx < window_size - 1:
                continue
            physics_window = physics_norm[idx - window_size + 1 : idx + 1]
            if len(physics_window) != window_size:
                continue

            try:
                with xr.open_dataset(file, engine='netcdf4') as ds:
                    ne = ds['Ne'].values.astype(np.float32)
                    ne = np.nan_to_num(ne, nan=0.0)
                    ne = np.clip(ne, 0, 1e7)
            except Exception as e:
                print(f"警告：跳过损坏文件 {file}: {e}")
                continue

            tec_full_true = compute_tec(ne, height_coords_global)
            tec_full_pred = predict(model, physics_window, mean, std, height_coords_global, device)

            for name, mask in region_masks.items():
                tec_true_region = tec_full_true[:, mask]
                tec_pred_region = tec_full_pred[:, mask]
                tec_true_mean = np.mean(tec_true_region)
                tec_pred_mean = np.mean(tec_pred_region)
                if (TEC_MIN_REASONABLE <= tec_pred_mean <= TEC_MAX_REASONABLE and
                    TEC_MIN_REASONABLE <= tec_true_mean <= TEC_MAX_REASONABLE):
                    error = tec_true_mean - tec_pred_mean
                    region_errors[name].append(error)

        del physics_df, physics_norm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    has_data = False
    for name, errors in region_errors.items():
        if len(errors) > 0:
            has_data = True
            break
    if not has_data:
        print("没有有效数据点，无法绘图")
        return

    for name, errors in region_errors.items():
        if len(errors) == 0:
            print(f"区域 {name} 无有效数据，跳过")
            continue

        errors = np.array(errors)
        mu = np.mean(errors)
        sigma = np.std(errors)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        n_points = len(errors)
        a = 1 / (sigma * np.sqrt(2 * np.pi))

        fig, ax = plt.subplots(figsize=(10, 6))
        # 直方图使用全部误差数据
        n, bins, patches = ax.hist(errors, bins=50, density=True, alpha=0.6,
                                   color='skyblue', edgecolor='black', label='Error distribution')
        # 高斯拟合曲线，x_vals 限制在[-25,25]内
        x_vals = np.linspace(-25, 25, 200)
        gaussian = a * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
        ax.plot(x_vals, gaussian, 'r-', linewidth=2, label='Gaussian fit')
        
        ax.set_xlabel('TEC Error (Trained Data - PTIM) (TECU)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Error Distribution in Longitude Region {name}')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(-25, 25)  # 强制横轴范围为 -25 到 25 TECU

        # 添加高斯表达式文本框
        gauss_expr = (
            f"$g(x) = a \\cdot e^{{ -(x-b)^2 / (2\\sigma^2) }}$\n"
            f"$a = {a:.4f}$\n"
            f"$b = \\mu = {mu:.4f}$ TECU\n"
            f"$\\sigma^2 = {sigma**2:.4f}$ TECU²"
        )
        ax.text(0.05, 0.95, gauss_expr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 添加统计信息文本框
        stats_text = f'MAE: {mae:.2f} TECU\nRMSE: {rmse:.2f} TECU\nN = {n_points}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        safe_name = name.replace('°', 'deg').replace('~', '_to_')
        save_name = f'error_histogram_{safe_name}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"误差分布图已保存为 {save_name} (样本数={n_points})")

    print("\n所有经度区域的误差直方图生成完毕！")

def diagnose_longitude():
    import matplotlib.pyplot as plt
    from datetime import datetime
    folder = r"F:\wenjian\2020"
    target_date = datetime(2020, 9, 23)
    hour = 12
    doy = target_date.timetuple().tm_yday
    fname = f"GIS_Ne_IRI_RO_GPS_{target_date.year}_{doy:03d}_{hour:02d}.nc"
    fpath = os.path.join(folder, fname)
    with xr.open_dataset(fpath, engine='netcdf4') as ds:
        ne = ds['Ne'].values
        if 'height' in ds.coords:
            height_coords = ds['height'].values
        else:
            height_coords = np.linspace(100, 1100, 51)
        if 'lat' in ds.coords:
            lat_coords = ds['lat'].values
        else:
            lat_coords = np.linspace(-90, 90, 73)
        if 'lon' in ds.coords:
            lon_coords_orig = ds['lon'].values
        else:
            lon_coords_orig = np.linspace(0, 360, 72, endpoint=False)
    lon_coords = np.where(lon_coords_orig > 180, lon_coords_orig - 360, lon_coords_orig)
    tec = compute_tec(ne, height_coords)  # (lat, lon)
    
    # 检查经度坐标是否单调递增
    print("经度坐标是否递增？", np.all(np.diff(lon_coords) > 0))
    # 检查不同经度列的TEC是否重复
    # 选择几个经度
    test_lons = [-150, -90, -30, 0, 30, 90, 150]
    for lon in test_lons:
        idx = np.argmin(np.abs(lon_coords - lon))
        col = tec[:, idx]
        print(f"经度 {lon}° (索引 {idx}) 的TEC范围: {col.min():.2f} - {col.max():.2f}, 平均值: {col.mean():.2f}")
    # 检查相邻列相关系数
    corrs = [np.corrcoef(tec[:, i], tec[:, i+1])[0,1] for i in range(tec.shape[1]-1)]
    print("相邻列平均相关系数:", np.mean(corrs))
    # 绘制经度-纬度TEC图，观察是否有明显的重复模式
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(lon_coords, lat_coords, tec, cmap='RdBu_r', shading='auto')
    plt.colorbar(label='TEC (TECU)')
    plt.title('Original TEC at UT=12:00')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.show()

def debug_lt_mapping():
    """
    调试地方时映射：读取2020-09-23 14:00 UTC的NC文件，
    计算赤道（纬度0°）每个经度在300km高度的电子密度，
    并输出经度、地方时、电子密度。
    """
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from datetime import datetime
    from glob import glob

    folder = r"F:\wenjian\2020"
    target_date = datetime(2020, 9, 23, 14, 0)  # 14:00 UTC
    doy = target_date.timetuple().tm_yday
    fname = f"GIS_Ne_IRI_RO_GPS_{target_date.year}_{doy:03d}_{target_date.hour:02d}.nc"
    fpath = os.path.join(folder, fname)
    if not os.path.exists(fpath):
        print(f"文件不存在：{fpath}")
        return

    with xr.open_dataset(fpath, engine='netcdf4') as ds:
        ne = ds['Ne'].values  # (height, lat, lon)
        # 获取纬度坐标
        if 'lat' in ds.coords:
            lat_coords = ds['lat'].values
        else:
            lat_coords = np.linspace(-90, 90, ne.shape[1])
        # 获取经度坐标
        if 'lon' in ds.coords:
            lon_coords = ds['lon'].values
        else:
            lon_coords = np.linspace(0, 360, ne.shape[2], endpoint=False)
        # 获取高度坐标
        if 'height' in ds.coords:
            height_coords = ds['height'].values
        else:
            height_coords = np.linspace(100, 1100, ne.shape[0])

    # 选择赤道附近纬度索引
    lat_idx = np.argmin(np.abs(lat_coords - 0))
    # 选择300km高度附近索引
    height_idx = np.argmin(np.abs(height_coords - 300))
    ne_eq = ne[height_idx, lat_idx, :]  # 经度方向

    # 计算每个经度的地方时
    utc_hour = target_date.hour
    # 经度转换到 -180..180 用于计算地方时偏移
    lon_deg = np.where(lon_coords > 180, lon_coords - 360, lon_coords)
    lt = (utc_hour + lon_deg / 15.0) % 24

    # 输出经度、地方时、电子密度
    print("经度(°), 地方时, Ne(1e5 el/cm³) at 300km, 赤道")
    for i in range(len(lon_coords)):
        print(f"{lon_deg[i]:6.1f}, {lt[i]:6.2f}, {ne_eq[i]:.2e}")

    # 绘制经度-电子密度图，并标注地方时14时附近
    plt.figure(figsize=(12, 5))
    plt.plot(lon_deg, ne_eq, 'b-')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Longitude (°)')
    plt.ylabel('Ne (1e5 el/cm³) at 300km, Equator')
    plt.title(f'Electron Density at {target_date} UT={utc_hour}:00')
    # 标记地方时14时的经度
    lt_target = 14.0
    # 找出地方时最接近14的经度
    idx_target = np.argmin(np.abs(lt - lt_target))
    lon_target = lon_deg[idx_target]
    plt.axvline(x=lon_target, color='r', linestyle='--', label=f'LT≈{lt_target} at lon={lon_target:.1f}°')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_peak_tec_by_ut_all_cols():
    """
    对于指定日期范围内每天，生成全球TEC分布图，观测（Trained Data）和重构（PTIM）分开保存。
    输出：
    1. 原始分辨率图 (72x73) 使用 pcolormesh
    2. 双线性插值高分辨率图 (288x292) 使用 imshow
    所有图片均使用固定色标 0~35 TECU，带世界地图投影。
    """
    print("\n" + "="*60)
    print("生成峰值TEC对比图（每个UT贡献3个经度列） - 观测与重构分开保存，固定色标0-35 TECU")
    print("="*60)

    target_lt = 14.0
    start_date_str = input("请输入起始日期 (格式 YYYY-MM-DD): ").strip()
    end_date_str = input("请输入结束日期 (格式 YYYY-MM-DD): ").strip()
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # 获取坐标
    sample_folder = r"F:\wenjian\2020"
    sample_files = glob(os.path.join(sample_folder, "*.nc"))
    if not sample_files:
        print("错误：无法找到样本NC文件获取坐标")
        return
    with xr.open_dataset(sample_files[0], engine='netcdf4') as ds:
        if 'height' in ds.coords:
            height_coords = ds['height'].values
        else:
            height_coords = np.linspace(100, 1100, 51)
        if 'lat' in ds.coords:
            lat_coords = ds['lat'].values
        else:
            lat_coords = np.linspace(-90, 90, 73)
        if 'lon' in ds.coords:
            lon_coords_orig = ds['lon'].values
        else:
            lon_coords_orig = np.linspace(0, 360, 72, endpoint=False)
    lon_coords = np.where(lon_coords_orig > 180, lon_coords_orig - 360, lon_coords_orig)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from scipy.interpolate import RegularGridInterpolator

    # 固定色标范围 0-35
    vmin_fixed, vmax_fixed = 0, 35
    norm_fixed = plt.Normalize(vmin=vmin_fixed, vmax=vmax_fixed)

    current_date = start_date
    while current_date <= end_date:
        print(f"\n处理日期: {current_date.strftime('%Y-%m-%d')}")
        year = current_date.year
        folder = os.path.join(r"F:\wenjian", str(year))
        if not os.path.exists(folder):
            print(f"警告：文件夹 {folder} 不存在，跳过")
            current_date += timedelta(days=1)
            continue

        utc_times = [current_date + timedelta(hours=h) for h in range(24)]

        # ---- 真实数据（观测） ----
        true_tec_by_lon = {}
        for utc in utc_times:
            hour = utc.hour
            doy = utc.timetuple().tm_yday
            fname = f"GIS_Ne_IRI_RO_GPS_{utc.year}_{doy:03d}_{hour:02d}.nc"
            fpath = os.path.join(folder, fname)
            if not os.path.exists(fpath):
                print(f"警告：文件不存在 {fpath}")
                continue
            with xr.open_dataset(fpath, engine='netcdf4') as ds:
                ne = ds['Ne'].values
                tec = compute_tec(ne, height_coords)
            j_center = (78 - 3 * hour) % 72
            for offset in [-3, 0, 3]:
                j = (j_center + offset) % 72
                lon_val = lon_coords[j]
                tec_profile = tec[:, j]
                true_tec_by_lon[lon_val] = tec_profile

        if not true_tec_by_lon:
            print(f"警告：{current_date.strftime('%Y-%m-%d')} 没有有效观测数据，跳过")
            current_date += timedelta(days=1)
            continue

        true_lons = sorted(true_tec_by_lon.keys())
        true_tec_matrix = np.array([true_tec_by_lon[lon] for lon in true_lons]).T

        # ---- 预测数据（PTIM） ----
        phys_start = current_date - timedelta(hours=4)
        phys_end = current_date + timedelta(hours=23)
        try:
            physics_df, physics_norm = load_physics_data(
                phys_start.strftime('%Y-%m-%d %H:%M:%S'),
                phys_end.strftime('%Y-%m-%d %H:%M:%S'),
                scaler, feature_names
            )
        except Exception as e:
            print(f"警告：加载物理数据失败: {e}，跳过 {current_date.strftime('%Y-%m-%d')}")
            current_date += timedelta(days=1)
            continue

        window_size = 4
        pred_tec_by_lon = {}
        for utc in utc_times:
            hour = utc.hour
            if utc not in physics_df.index:
                continue
            idx = physics_df.index.get_loc(utc)
            if idx < window_size - 1:
                continue
            phys_window = physics_norm[idx - window_size + 1 : idx + 1]
            if len(phys_window) != window_size:
                continue
            phys_tensor = torch.FloatTensor(phys_window).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(phys_tensor).cpu().squeeze().numpy()
            pred_log = pred_norm * std + mean
            pred_raw = np.expm1(pred_log)
            pred_raw = np.clip(pred_raw, 0, 1e7)
            tec = compute_tec(pred_raw, height_coords)

            j_center = (78 - 3 * hour) % 72
            for offset in [-3, 0, 3]:
                j = (j_center + offset) % 72
                lon_val = lon_coords[j]
                tec_profile = tec[:, j]
                pred_tec_by_lon[lon_val] = tec_profile

        if not pred_tec_by_lon:
            print(f"警告：{current_date.strftime('%Y-%m-%d')} 没有预测数据，跳过")
            current_date += timedelta(days=1)
            continue

        pred_lons = sorted(pred_tec_by_lon.keys())
        pred_tec_matrix = np.array([pred_tec_by_lon[lon] for lon in pred_lons]).T

        # ========== 1. 原始分辨率图 (72x73) 分开保存 ==========
        # 观测图
        fig_obs, ax_obs = plt.subplots(1, 1, figsize=(7, 6),
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        im_obs = ax_obs.pcolormesh(true_lons, lat_coords, true_tec_matrix,
                                   cmap='RdBu_r', norm=norm_fixed, shading='auto',
                                   transform=ccrs.PlateCarree())
        ax_obs.set_title(f'Trained Data TEC (Orig) at LT≈{target_lt} ({current_date.strftime("%Y-%m-%d")})')
        ax_obs.set_xlabel('Longitude (°)')
        ax_obs.set_ylabel('Latitude (°)')
        ax_obs.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_obs.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_obs.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_obs.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax_obs.xaxis.set_major_formatter(LongitudeFormatter())
        ax_obs.yaxis.set_major_formatter(LatitudeFormatter())
        ax_obs.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        plt.colorbar(im_obs, ax=ax_obs, label='TEC (TECU)', shrink=0.6)
        plt.tight_layout()
        save_obs = f"peak_TEC_{current_date.strftime('%Y-%m-%d')}_LT{int(target_lt)}_orig_obs.png"
        plt.savefig(save_obs, dpi=300, bbox_inches='tight')
        plt.close(fig_obs)
        print(f"原始分辨率观测图已保存为 {save_obs}")

        # 预测图
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(7, 6),
                                         subplot_kw={'projection': ccrs.PlateCarree()})
        im_pred = ax_pred.pcolormesh(pred_lons, lat_coords, pred_tec_matrix,
                                     cmap='RdBu_r', norm=norm_fixed, shading='auto',
                                     transform=ccrs.PlateCarree())
        ax_pred.set_title(f'PTIM TEC (Orig) at LT≈{target_lt} ({current_date.strftime("%Y-%m-%d")})')
        ax_pred.set_xlabel('Longitude (°)')
        ax_pred.set_ylabel('Latitude (°)')
        ax_pred.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_pred.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_pred.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_pred.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax_pred.xaxis.set_major_formatter(LongitudeFormatter())
        ax_pred.yaxis.set_major_formatter(LatitudeFormatter())
        ax_pred.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        plt.colorbar(im_pred, ax=ax_pred, label='TEC (TECU)', shrink=0.6)
        plt.tight_layout()
        save_pred = f"peak_TEC_{current_date.strftime('%Y-%m-%d')}_LT{int(target_lt)}_orig_pred.png"
        plt.savefig(save_pred, dpi=300, bbox_inches='tight')
        plt.close(fig_pred)
        print(f"原始分辨率预测图已保存为 {save_pred}")

        # ========== 2. 高分辨率插值 (288x292) 使用 imshow 分开保存 ==========
        orig_lats = lat_coords
        orig_lons = np.array(true_lons)
        new_lons = np.linspace(orig_lons.min(), orig_lons.max(), 288)
        new_lats = np.linspace(orig_lats.min(), orig_lats.max(), 292)

        interp_true = RegularGridInterpolator((orig_lats, orig_lons), true_tec_matrix,
                                              method='linear', bounds_error=False, fill_value=np.nan)
        interp_pred = RegularGridInterpolator((orig_lats, orig_lons), pred_tec_matrix,
                                              method='linear', bounds_error=False, fill_value=np.nan)

        grid_lat, grid_lon = np.meshgrid(new_lats, new_lons, indexing='ij')
        tec_true_interp = interp_true((grid_lat, grid_lon))
        tec_pred_interp = interp_pred((grid_lat, grid_lon))

        extent = [-180, 180, -90, 90]

        # 高分辨率观测图
        fig_high_obs, ax_high_obs = plt.subplots(1, 1, figsize=(7, 6),
                                                 subplot_kw={'projection': ccrs.PlateCarree()})
        im_high_obs = ax_high_obs.imshow(tec_true_interp, extent=extent, origin='lower',
                                         cmap='RdBu_r', norm=norm_fixed, interpolation='bilinear',
                                         transform=ccrs.PlateCarree())
        ax_high_obs.set_title(f'Trained Data TEC at LT≈{target_lt} ({current_date.strftime("%Y-%m-%d")})')
        ax_high_obs.set_xlabel('Longitude (°)')
        ax_high_obs.set_ylabel('Latitude (°)')
        ax_high_obs.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_high_obs.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_high_obs.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_high_obs.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax_high_obs.xaxis.set_major_formatter(LongitudeFormatter())
        ax_high_obs.yaxis.set_major_formatter(LatitudeFormatter())
        ax_high_obs.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        plt.colorbar(im_high_obs, ax=ax_high_obs, label='TEC (TECU)', shrink=0.6)
        plt.tight_layout()
        save_high_obs = f"peak_TEC_{current_date.strftime('%Y-%m-%d')}_LT{int(target_lt)}_high_obs.png"
        plt.savefig(save_high_obs, dpi=300, bbox_inches='tight')
        plt.close(fig_high_obs)
        print(f"高分辨率观测图已保存为 {save_high_obs}")

        # 高分辨率预测图
        fig_high_pred, ax_high_pred = plt.subplots(1, 1, figsize=(7, 6),
                                                   subplot_kw={'projection': ccrs.PlateCarree()})
        im_high_pred = ax_high_pred.imshow(tec_pred_interp, extent=extent, origin='lower',
                                           cmap='RdBu_r', norm=norm_fixed, interpolation='bilinear',
                                           transform=ccrs.PlateCarree())
        ax_high_pred.set_title(f'PTIM TEC at LT≈{target_lt} ({current_date.strftime("%Y-%m-%d")})')
        ax_high_pred.set_xlabel('Longitude (°)')
        ax_high_pred.set_ylabel('Latitude (°)')
        ax_high_pred.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_high_pred.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_high_pred.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_high_pred.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax_high_pred.xaxis.set_major_formatter(LongitudeFormatter())
        ax_high_pred.yaxis.set_major_formatter(LatitudeFormatter())
        ax_high_pred.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        plt.colorbar(im_high_pred, ax=ax_high_pred, label='TEC (TECU)', shrink=0.6)
        plt.tight_layout()
        save_high_pred = f"peak_TEC_{current_date.strftime('%Y-%m-%d')}_LT{int(target_lt)}_high_pred.png"
        plt.savefig(save_high_pred, dpi=300, bbox_inches='tight')
        plt.close(fig_high_pred)
        print(f"高分辨率预测图已保存为 {save_high_pred}")

        current_date += timedelta(days=1)

    print("\n全部日期处理完成！")

def generate_time_range_plots():
    """
    生成指定时间范围内逐小时的观测与预测电子密度图（分开保存）。
    每个时间步生成两张独立图片：
        - example_YYYYMMDD_HH_obs.png：Trained Data Ne（观测）
        - example_YYYYMMDD_HH_pred.png：PTIM Ne（预测）
    每张图包含200km、400km、600km三个高度层，从上到下排列。
    添加世界地图海岸线背景，并显示经纬度刻度。
    """
    import os
    from datetime import datetime, timedelta
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # 目标时间段（可根据需要修改）
    start_date = datetime(2023, 3, 23, 0, 0, 0)
    end_date   = datetime(2023, 3, 24, 23, 0, 0)

    # 输出目录
    out_dir = 'prediction_examples_physics_to_iono'
    os.makedirs(out_dir, exist_ok=True)

    # 目标高度（km）
    target_heights = [200, 400, 600]

    # 加载物理数据和电离层数据
    print(f"加载物理数据: {start_date} 至 {end_date}")
    physics_df, physics_norm = load_physics_data(
        start_date.strftime('%Y-%m-%d %H:%M:%S'),
        end_date.strftime('%Y-%m-%d %H:%M:%S'),
        scaler, feature_names
    )
    print(f"加载电离层数据...")
    iono_dict, height_coords, lat_coords, lon_coords = load_iono_data(
        start_date.strftime('%Y-%m-%d %H:%M:%S'),
        end_date.strftime('%Y-%m-%d %H:%M:%S')
    )
    if height_coords is None:
        raise ValueError("未找到高度坐标")

    # 获取纬度、经度数组（如果为None则使用默认值）
    if lat_coords is None:
        nlat = next(iter(iono_dict.values())).shape[1]
        lat_coords = np.linspace(-90, 90, nlat)
        print(f"使用默认纬度数组: 形状 {lat_coords.shape}")
    if lon_coords is None:
        nlon = next(iter(iono_dict.values())).shape[2]
        lon_coords = np.linspace(0, 360, nlon, endpoint=False)
        print(f"使用默认经度数组: 形状 {lon_coords.shape}")

    # 找到目标高度在数组中的索引
    height_indices = []
    for h in target_heights:
        diff = np.abs(height_coords - h)
        idx = np.argmin(diff)
        height_indices.append(idx)
        print(f"目标高度 {h} km → 实际高度 {height_coords[idx]:.1f} km (索引 {idx})")

    # 确定共同时间点（物理和电离层数据都有的时间）
    common_times = sorted(set(physics_df.index) & set(iono_dict.keys()))
    if not common_times:
        print("错误：没有共同时间点")
        return

    # 只保留目标时间范围内的点
    target_times = [t for t in common_times if start_date <= t <= end_date]
    print(f"找到 {len(target_times)} 个有效时间点")

    window_size = 4  # 模型需要的窗口大小

    for t in target_times:
        idx = physics_df.index.get_loc(t)
        if idx < window_size - 1:
            print(f"警告：时间 {t} 历史窗口不足，跳过")
            continue

        # 构建输入窗口
        physics_window = physics_norm[idx - window_size + 1 : idx + 1]
        if len(physics_window) != window_size:
            continue

        # 预测
        physics_tensor = torch.FloatTensor(physics_window).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(physics_tensor).cpu().squeeze().numpy()  # (H, L, W)

        # 反归一化预测值
        pred_log = pred_norm * std + mean
        pred_raw = np.expm1(pred_log)
        pred_raw = np.clip(pred_raw, 0, 1e7)

        # 获取真实值（观测）
        true_ne = iono_dict[t]  # 形状 (H, L, W)
        true_raw = true_ne

        # 经度显示范围（-180 到 180）
        lon_display = lon_coords - 180

        # ========== 图1：观测（Trained Data） ==========
        n_rows = len(target_heights)
        fig_obs, axes_obs = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows),
                                         subplot_kw={'projection': ccrs.PlateCarree()})
        if n_rows == 1:
            axes_obs = [axes_obs]

        for i, hi in enumerate(height_indices):
            ax = axes_obs[i]
            im = ax.pcolormesh(lon_display, lat_coords, true_raw[hi, :, :],
                               cmap='RdBu_r', shading='auto',
                               transform=ccrs.PlateCarree())
            ax.set_title(f'Trained Data Ne at {target_heights[i]} km')
            ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            plt.colorbar(im, ax=ax, label='Ne (1e5 el/cm^3)', shrink=0.8)

        plt.suptitle(f'Trained Data Ne - {t.strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout()
        obs_filename = os.path.join(out_dir, f"example_{t.strftime('%Y%m%d_%H')}_obs.png")
        plt.savefig(obs_filename, dpi=150, bbox_inches='tight')
        plt.close(fig_obs)
        print(f"已保存: {obs_filename}")

        # ========== 图2：预测（PTIM） ==========
        fig_pred, axes_pred = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows),
                                           subplot_kw={'projection': ccrs.PlateCarree()})
        if n_rows == 1:
            axes_pred = [axes_pred]

        for i, hi in enumerate(height_indices):
            ax = axes_pred[i]
            im = ax.pcolormesh(lon_display, lat_coords, pred_raw[hi, :, :],
                               cmap='RdBu_r', shading='auto',
                               transform=ccrs.PlateCarree())
            ax.set_title(f'PTIM Ne at {target_heights[i]} km')
            ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            plt.colorbar(im, ax=ax, label='Ne (1e5 el/cm^3)', shrink=0.8)

        plt.suptitle(f'PTIM Ne - {t.strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout()
        pred_filename = os.path.join(out_dir, f"example_{t.strftime('%Y%m%d_%H')}_pred.png")
        plt.savefig(pred_filename, dpi=150, bbox_inches='tight')
        plt.close(fig_pred)
        print(f"已保存: {pred_filename}")

    print(f"全部完成！共生成 {len(target_times)} 个时间步的观测和预测图片，保存在 {out_dir}")

def generate_time_range_tec_plots():
    """
    生成指定时间段的 TEC 重构图与观测图，分开保存为独立图片。
    每个时间步生成两张图片：
        - observe_TEC_YYYYMMDD_HH.png：Trained Data TEC（观测）
        - refactor_TEC_YYYYMMDD_HH.png：PTIM TEC（重构）
    每张图使用固定色标范围 0~30 TECU。
    """
    import os
    from datetime import datetime, timedelta
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    print("\n请输入起止时间（格式：YYYY-MM-DD HH:MM:SS）")
    start_input = input("起始时间: ").strip()
    end_input = input("结束时间: ").strip()
    try:
        start_date = datetime.strptime(start_input, '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime(end_input, '%Y-%m-%d %H:%M:%S')
    except:
        print("格式错误，使用默认时间范围 2023-03-22 00:00:00 至 2023-03-24 23:00:00")
        start_date = datetime(2023, 3, 22, 0, 0, 0)
        end_date = datetime(2023, 3, 24, 23, 0, 0)

    out_dir = 'TEC_prediction_examples'
    os.makedirs(out_dir, exist_ok=True)

    print(f"加载物理数据: {start_date} 至 {end_date}")
    physics_df, physics_norm = load_physics_data(
        start_date.strftime('%Y-%m-%d %H:%M:%S'),
        end_date.strftime('%Y-%m-%d %H:%M:%S'),
        scaler, feature_names
    )
    print(f"加载电离层数据...")
    iono_dict, height_coords, lat_coords, lon_coords = load_iono_data(
        start_date.strftime('%Y-%m-%d %H:%M:%S'),
        end_date.strftime('%Y-%m-%d %H:%M:%S')
    )
    if height_coords is None:
        raise ValueError("未找到高度坐标")

    if lat_coords is None:
        nlat = next(iter(iono_dict.values())).shape[1]
        lat_coords = np.linspace(-90, 90, nlat)
        print(f"使用默认纬度数组: 形状 {lat_coords.shape}")
    if lon_coords is None:
        nlon = next(iter(iono_dict.values())).shape[2]
        lon_coords = np.linspace(0, 360, nlon, endpoint=False)
        print(f"使用默认经度数组: 形状 {lon_coords.shape}")

    common_times = sorted(set(physics_df.index) & set(iono_dict.keys()))
    if not common_times:
        print("错误：没有共同时间点")
        return

    target_times = [t for t in common_times if start_date <= t <= end_date]
    print(f"找到 {len(target_times)} 个有效时间点")

    window_size = 4
    lon_display = lon_coords - 180  # 转换到 -180..180

    # 固定色标范围 0~30 TECU
    vmin_fixed, vmax_fixed = 0, 30
    norm_fixed = plt.Normalize(vmin=vmin_fixed, vmax=vmax_fixed)

    for t in target_times:
        idx = physics_df.index.get_loc(t)
        if idx < window_size - 1:
            print(f"警告：时间 {t} 历史窗口不足，跳过")
            continue

        physics_window = physics_norm[idx - window_size + 1 : idx + 1]
        if len(physics_window) != window_size:
            continue

        # 真实 TEC（观测）
        tec_true = compute_tec(iono_dict[t], height_coords)
        # 预测 TEC
        tec_pred = predict(model, physics_window, mean, std, height_coords, device)

        def plot_tec(ax, tec_data, title):
            im = ax.pcolormesh(lon_display, lat_coords, tec_data,
                               cmap='RdBu_r', norm=norm_fixed, shading='auto',
                               transform=ccrs.PlateCarree())
            ax.set_title(title)
            ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            return im

        # 观测图（Trained Data TEC）
        fig_obs, ax_obs = plt.subplots(1, 1, figsize=(10, 6),
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        im_obs = plot_tec(ax_obs, tec_true, f'Trained Data TEC at {t.strftime("%Y-%m-%d %H:%M")}')
        plt.colorbar(im_obs, ax=ax_obs, orientation='vertical', label='TEC (TECU)', shrink=0.8)
        plt.tight_layout()
        obs_name = os.path.join(out_dir, f"observe_TEC_{t.strftime('%Y%m%d_%H')}.png")
        plt.savefig(obs_name, dpi=150, bbox_inches='tight')
        plt.close(fig_obs)

        # 预测图（PTIM TEC）
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(10, 6),
                                         subplot_kw={'projection': ccrs.PlateCarree()})
        im_pred = plot_tec(ax_pred, tec_pred, f'PTIM TEC at {t.strftime("%Y-%m-%d %H:%M")}')
        plt.colorbar(im_pred, ax=ax_pred, orientation='vertical', label='TEC (TECU)', shrink=0.8)
        plt.tight_layout()
        pred_name = os.path.join(out_dir, f"refactor_TEC_{t.strftime('%Y%m%d_%H')}.png")
        plt.savefig(pred_name, dpi=150, bbox_inches='tight')
        plt.close(fig_pred)

        print(f"已保存: {obs_name} 和 {pred_name}")

    print(f"全部完成！共生成 {len(target_times)} 个时间步的观测和预测 TEC 图片，保存在 {out_dir}")

def plot_fof2_comparison(sta, year, sta_lat, sta_lon, data_dir='.', output_dir='.'):
    global model, scaler, feature_names, mean, std, device

    mat_file = os.path.join(data_dir, f"{sta}{year}fof2.mat")
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"数据文件 {mat_file} 不存在！")
    data = loadmat(mat_file)
    all_obs = data['all_obs']
    all_bgm = data['all_bgm']
    all_doy = data['all_doy'].flatten()
    interp_ut = data['interp_ut'].flatten()

    def fof2_to_ne(fof2):
        fof2 = np.real(fof2)
        ne = (fof2 * 1000) ** 2 / 80.6
        ne = np.where(fof2 > 13, np.nan, ne)
        return ne

    tmp0 = fof2_to_ne(all_obs).T
    tmp1 = fof2_to_ne(all_bgm).T
    print(f"观测电子密度 (el/cm³) 统计: min={np.nanmin(tmp0):.2e}, max={np.nanmax(tmp0):.2e}, mean={np.nanmean(tmp0):.2e}")

    # 网格坐标
    lat_coords = np.linspace(-90, 90, 73)
    lon_coords = np.linspace(0, 360, 72, endpoint=False)
    height_coords = np.linspace(100, 1100, 51)

    sta_lon_360 = sta_lon % 360
    lat_idx = np.argmin(np.abs(lat_coords - sta_lat))
    lon_idx = np.argmin(np.abs(lon_coords - sta_lon_360))
    print(f"台站 {sta} 经纬度 ({sta_lat}, {sta_lon}) -> 网格索引 lat={lat_idx}, lon={lon_idx}")

    # 加载物理数据
    phys_start = datetime(year, 1, 1, 0, 0, 0)
    phys_end = datetime(year, 12, 31, 23, 0, 0)
    physics_df, physics_norm = load_physics_data(
        phys_start.strftime('%Y-%m-%d %H:%M:%S'),
        phys_end.strftime('%Y-%m-%d %H:%M:%S'),
        scaler, feature_names
    )
    time_to_phys_idx = {t: i for i, t in enumerate(physics_df.index)}
    window_size = 4

    n_doy = len(all_doy)
    tmp2_raw = np.full((24, n_doy), np.nan, dtype=np.float32)

    for i_doy, doy in enumerate(all_doy):
        base_date = datetime(year, 1, 1) + timedelta(days=int(doy)-1)
        for hour in range(24):
            target_time = base_date + timedelta(hours=hour)
            if target_time not in time_to_phys_idx:
                continue
            idx = time_to_phys_idx[target_time]
            if idx < window_size - 1:
                continue
            physics_window = physics_norm[idx - window_size + 1 : idx + 1]
            if len(physics_window) != window_size:
                continue

            physics_tensor = torch.FloatTensor(physics_window).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(physics_tensor).cpu().squeeze().numpy()
            pred_log = pred_norm * std + mean
            pred_raw = np.expm1(pred_log)
            # 截断异常高值（上限设为 5e6 el/cm³）
            pred_raw = np.clip(pred_raw, 0, 5e6)

            vertical_profile = pred_raw[:, lat_idx, lon_idx]
            peak_ne = np.max(vertical_profile)           # 单位: 1e5 el/cm³
            # 截断异常高值: 上限 25 (对应 2.5e6 el/cm³)，下限 0
            peak_ne = np.clip(peak_ne, 0, 25)
            peak_ne_elcm3 = peak_ne * 1e5                # 转换为 el/cm³
            tmp2_raw[hour, i_doy] = peak_ne_elcm3
    # 打印截断后的统计信息
    print(f"模型预测峰值电子密度 (el/cm³) 截断后统计: min={np.nanmin(tmp2_raw):.2e}, max={np.nanmax(tmp2_raw):.2e}, mean={np.nanmean(tmp2_raw):.2e}")

    # 时间插值
    from scipy.interpolate import interp1d
    n_ut = len(interp_ut)
    tmp2 = np.full((n_ut, n_doy), np.nan, dtype=np.float32)
    ut_hours_full = np.arange(24)
    for j in range(n_doy):
        y_vals = tmp2_raw[:, j]
        mask = ~np.isnan(y_vals)
        if np.sum(mask) < 2:
            continue
        f = interp1d(ut_hours_full[mask], y_vals[mask], kind='linear',
                     bounds_error=False, fill_value=np.nan)
        tmp2[:, j] = f(interp_ut)
    # 在绘图前添加
    max_idx = np.unravel_index(np.nanargmax(tmp2), tmp2.shape)
    max_ut = interp_ut[max_idx[0]]
    max_doy = all_doy[max_idx[1]]
    print(f"PTIM 最大值位置: UT={max_ut:.2f}, DOY={max_doy}, 值={tmp2[max_idx]:.2e}")
    print(f"对应观测值: {tmp0[max_idx]:.2e}, IRI值: {tmp1[max_idx]:.2e}")
        # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16.9/2.54, 20/2.54),
                             dpi=300, constrained_layout=True)
    vmin, vmax = 0, 2e6
    x = all_doy
    y = interp_ut

    im0 = axes[0].pcolormesh(x, y, tmp0, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title('观测')
    axes[0].set_ylabel('UT')

    im1 = axes[1].pcolormesh(x, y, tmp1, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title('IRI')
    axes[1].set_ylabel('UT')

    im2 = axes[2].pcolormesh(x, y, tmp2, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].set_title('PTIM (模型重构, 异常值截断)')
    axes[2].set_xlabel('DOY')
    axes[2].set_ylabel('UT')

    cbar = fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label('el/cm^3')

    fig.suptitle(f'{sta} {year} fof2')
    out_file = os.path.join(output_dir, f"ssfigure{sta}{year}fof2_PTIM_clipped.jpg")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图片已保存至 {out_file}")


def plot_height_Ne_by_ut():
    """
    对于指定日期（2020-09-14），计算地方时14 LT下不同高度（200,400,600 km）的电子密度。
    观测（Trained Data）和重构（PTIM）拼在一张图上：左列观测，右列重构，共3行。
    所有子图使用统一的colorbar，范围基于400km高度的观测和重构数据的最大值。
    使用双线性插值到288×292高分辨率网格，带世界地图投影。
    """
    print("\n" + "="*60)
    print("生成分高度电子密度图（2020-09-14，地方时14时） - 观测与重构拼图")
    print("="*60)

    target_lt = 14.0
    target_date = datetime(2020, 9, 14)
    year = target_date.year
    folder = os.path.join(r"F:\wenjian", str(year))
    if not os.path.exists(folder):
        print(f"错误：文件夹 {folder} 不存在")
        return

    # 目标高度（km） - 只保留200,400,600，去掉800
    target_heights = [200, 400, 600]

    # 获取坐标（从任意一个NC文件）
    sample_folder = r"F:\wenjian\2020"
    sample_files = glob(os.path.join(sample_folder, "*.nc"))
    if not sample_files:
        print("错误：无法找到样本NC文件获取坐标")
        return
    with xr.open_dataset(sample_files[0], engine='netcdf4') as ds:
        if 'height' in ds.coords:
            height_coords = ds['height'].values
        else:
            height_coords = np.linspace(100, 1100, 51)
        if 'lat' in ds.coords:
            lat_coords = ds['lat'].values
        else:
            lat_coords = np.linspace(-90, 90, 73)
        if 'lon' in ds.coords:
            lon_coords_orig = ds['lon'].values
        else:
            lon_coords_orig = np.linspace(0, 360, 72, endpoint=False)
    lon_coords = np.where(lon_coords_orig > 180, lon_coords_orig - 360, lon_coords_orig)

    # 找到目标高度在数组中的索引
    height_indices = []
    for h in target_heights:
        diff = np.abs(height_coords - h)
        idx = np.argmin(diff)
        height_indices.append(idx)
        print(f"目标高度 {h} km → 实际高度 {height_coords[idx]:.1f} km (索引 {idx})")

    # 获取当天所有UT小时（0-23）
    utc_times = [target_date + timedelta(hours=h) for h in range(24)]

    # ---- 观测数据（Trained Data）----
    print("加载观测电子密度...")
    true_ne_by_height = {h: None for h in target_heights}
    for hi, h_val in enumerate(target_heights):
        lon_to_ne = {}
        for utc in utc_times:
            hour = utc.hour
            doy = utc.timetuple().tm_yday
            fname = f"GIS_Ne_IRI_RO_GPS_{utc.year}_{doy:03d}_{hour:02d}.nc"
            fpath = os.path.join(folder, fname)
            if not os.path.exists(fpath):
                continue
            with xr.open_dataset(fpath, engine='netcdf4') as ds:
                ne = ds['Ne'].values
            ne_level = ne[height_indices[hi], :, :]
            j_center = (78 - 3 * hour) % 72
            for offset in [-3, 0, 3]:
                j = (j_center + offset) % 72
                lon_val = lon_coords[j]
                ne_profile = ne_level[:, j]
                if lon_val not in lon_to_ne:
                    lon_to_ne[lon_val] = [ne_profile, 1]
                else:
                    lon_to_ne[lon_val][0] += ne_profile
                    lon_to_ne[lon_val][1] += 1
        if not lon_to_ne:
            print(f"错误：高度 {h_val} km 没有有效观测数据")
            return
        lons = sorted(lon_to_ne.keys())
        ne_matrix = np.array([lon_to_ne[lon][0] / lon_to_ne[lon][1] for lon in lons]).T
        true_ne_by_height[h_val] = (lons, ne_matrix)

    # ---- 重构数据（PTIM）----
    print("加载模型重构电子密度...")
    pred_ne_by_height = {h: None for h in target_heights}
    phys_start = target_date - timedelta(hours=4)
    phys_end = target_date + timedelta(hours=23)
    try:
        physics_df, physics_norm = load_physics_data(
            phys_start.strftime('%Y-%m-%d %H:%M:%S'),
            phys_end.strftime('%Y-%m-%d %H:%M:%S'),
            scaler, feature_names
        )
    except Exception as e:
        print(f"警告：加载物理数据失败: {e}")
        return

    window_size = 4
    for hi, h_val in enumerate(target_heights):
        lon_to_ne = {}
        for utc in utc_times:
            hour = utc.hour
            if utc not in physics_df.index:
                continue
            idx = physics_df.index.get_loc(utc)
            if idx < window_size - 1:
                continue
            phys_window = physics_norm[idx - window_size + 1 : idx + 1]
            if len(phys_window) != window_size:
                continue
            phys_tensor = torch.FloatTensor(phys_window).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(phys_tensor).cpu().squeeze().numpy()
            pred_log = pred_norm * std + mean
            pred_raw = np.expm1(pred_log)
            pred_raw = np.clip(pred_raw, 0, 1e7)
            ne_level = pred_raw[height_indices[hi], :, :]
            j_center = (78 - 3 * hour) % 72
            for offset in [-3, 0, 3]:
                j = (j_center + offset) % 72
                lon_val = lon_coords[j]
                ne_profile = ne_level[:, j]
                if lon_val not in lon_to_ne:
                    lon_to_ne[lon_val] = [ne_profile, 1]
                else:
                    lon_to_ne[lon_val][0] += ne_profile
                    lon_to_ne[lon_val][1] += 1
        if not lon_to_ne:
            print(f"错误：高度 {h_val} km 没有有效重构数据")
            return
        lons = sorted(lon_to_ne.keys())
        ne_matrix = np.array([lon_to_ne[lon][0] / lon_to_ne[lon][1] for lon in lons]).T
        pred_ne_by_height[h_val] = (lons, ne_matrix)

    # 导入插值和投影工具
    from scipy.interpolate import RegularGridInterpolator
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # 统一插值到高分辨率网格（288经度×292纬度）
    orig_lats = lat_coords
    new_lats = np.linspace(orig_lats.min(), orig_lats.max(), 292)
    new_lons = np.linspace(-180, 180, 288, endpoint=False)
    extent = [-180, 180, -90, 90]

    # 辅助函数：经度重排
    def sort_by_lon(lons, data):
        lons_arr = np.array(lons)
        lons_180 = np.where(lons_arr > 180, lons_arr - 360, lons_arr)
        sort_idx = np.argsort(lons_180)
        return lons_180[sort_idx], data[:, sort_idx]

    # 计算400km高度的观测和重构数据，以确定统一色标范围
    h400 = 400
    obs_lons, obs_ne400 = true_ne_by_height[h400]
    pred_lons, pred_ne400 = pred_ne_by_height[h400]
    obs_lons_sorted, obs_ne400_sorted = sort_by_lon(obs_lons, obs_ne400)
    pred_lons_sorted, pred_ne400_sorted = sort_by_lon(pred_lons, pred_ne400)

    interp_obs400 = RegularGridInterpolator((orig_lats, obs_lons_sorted), obs_ne400_sorted,
                                            method='linear', bounds_error=False, fill_value=np.nan)
    interp_pred400 = RegularGridInterpolator((orig_lats, pred_lons_sorted), pred_ne400_sorted,
                                             method='linear', bounds_error=False, fill_value=np.nan)
    grid_lat, grid_lon = np.meshgrid(new_lats, new_lons, indexing='ij')
    ne_obs400_interp = interp_obs400((grid_lat, grid_lon))
    ne_pred400_interp = interp_pred400((grid_lat, grid_lon))
    vmax_common = max(np.nanmax(ne_obs400_interp), np.nanmax(ne_pred400_interp))
    vmin_common = 0
    norm_common = plt.Normalize(vmin=vmin_common, vmax=vmax_common)
    print(f"统一色标范围: {vmin_common} ~ {vmax_common:.2f} (1e5 el/cm³)")

    # 创建拼图：3行2列
    fig, axes = plt.subplots(len(target_heights), 2, figsize=(14, 12),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    if len(target_heights) == 1:
        axes = axes.reshape(1, -1)

    for i, h_val in enumerate(target_heights):
        # 观测
        lons_obs, ne_obs = true_ne_by_height[h_val]
        lons_obs_sorted, ne_obs_sorted = sort_by_lon(lons_obs, ne_obs)
        interp_obs = RegularGridInterpolator((orig_lats, lons_obs_sorted), ne_obs_sorted,
                                             method='linear', bounds_error=False, fill_value=np.nan)
        ne_obs_interp = interp_obs((grid_lat, grid_lon))
        ax_obs = axes[i, 0]
        im_obs = ax_obs.imshow(ne_obs_interp, extent=extent, origin='lower',
                               cmap='RdBu_r', norm=norm_common, interpolation='bilinear',
                               transform=ccrs.PlateCarree())
        ax_obs.set_title(f'Trained Data Ne at {h_val} km (LT≈{target_lt})')
        ax_obs.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_obs.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_obs.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_obs.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax_obs.xaxis.set_major_formatter(LongitudeFormatter())
        ax_obs.yaxis.set_major_formatter(LatitudeFormatter())
        ax_obs.gridlines(draw_labels=False, linestyle='--', alpha=0.5)

        # 重构
        lons_pred, ne_pred = pred_ne_by_height[h_val]
        lons_pred_sorted, ne_pred_sorted = sort_by_lon(lons_pred, ne_pred)
        interp_pred = RegularGridInterpolator((orig_lats, lons_pred_sorted), ne_pred_sorted,
                                              method='linear', bounds_error=False, fill_value=np.nan)
        ne_pred_interp = interp_pred((grid_lat, grid_lon))
        ax_pred = axes[i, 1]
        im_pred = ax_pred.imshow(ne_pred_interp, extent=extent, origin='lower',
                                 cmap='RdBu_r', norm=norm_common, interpolation='bilinear',
                                 transform=ccrs.PlateCarree())
        ax_pred.set_title(f'PTIM Ne at {h_val} km (LT≈{target_lt})')
        ax_pred.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_pred.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_pred.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_pred.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax_pred.xaxis.set_major_formatter(LongitudeFormatter())
        ax_pred.yaxis.set_major_formatter(LatitudeFormatter())
        ax_pred.gridlines(draw_labels=False, linestyle='--', alpha=0.5)

    # 添加共享colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_obs, cax=cbar_ax, label='Ne (1e5 el/cm^3)')
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle(f'Electron Density Comparison - {target_date.strftime("%Y-%m-%d")} LT {target_lt}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    save_name = f'height_Ne_{target_date.strftime("%Y-%m-%d")}_LT{int(target_lt)}_combined.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"拼图已保存为 {save_name}")
    print("\n分高度电子密度拼图生成完毕！")

# ============================ 主程序 ============================
def main():
    global model, mean, std, scaler, feature_names

    model_path = r"E:\daima\2026.3.27\best_physics_to_iono.pth"
    scaler_path = r"E:\daima\2026.3.27\physics_scaler.pkl"
    feature_names_path = r"E:\daima\2026.3.27\feature_names.json"
    cache_dir = './data_cache'
    stats_info_path = os.path.join(cache_dir, 'ionosphere_stats_info.json')

    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    physics_feature_dim = checkpoint['physics_feature_dim']
    model = PhysicsToIonosphereModel(physics_feature_dim=physics_feature_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"模型加载成功，window_size = {checkpoint['window_size']}")

    # 加载电离层归一化参数
    print("加载电离层归一化参数...")
    with open(stats_info_path, 'r') as f:
        stats_info = json.load(f)
    mean = stats_info['mean']
    std = stats_info['std']
    print(f"mean={mean:.4e}, std={std:.4e}")

    # 加载物理特征scaler
    print("加载物理特征scaler...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    print(f"scaler加载成功，特征数={len(feature_names)}")

    # 定义区域（与短时验证一致）
    regions = {
        'Global': (-90, 90),
        'Mid-Latitude (±45°)': (-45, 45),
        'North Mid-Latitude (30-75°)': (30, 75),
        'Equatorial (±15°)': (-15, 15)
    }

    # 主菜单
    while True:
        print("请选择验证模式:")
        print("1. 短时验证（多时间段，可选用有原模式）")
        print("2. 长时无原重构（单时间段，逐日平均）")
        print("3. 生成指定时间范围预测图（电子密度，逐小时）")
        print("4. 生成误差统计图（分布对比）")
        print("5. 生成峰值密度对比图")
        print("6. 生成指定时间范围 TEC 对比图（全高度积分）")  
        print("7. 生成IRI与PTIM模型重构对比图")
        print("8. 生成分高度电子密度图（2020-09-14，LT14）")
        print("9. 退出")
        choice = input("请输入数字 (1-9): ").strip()
        if choice == '1':
            # 短时验证
            # 定义时间段
            quiet_periods = [
                ('2020-06-01~03', '2020-06-01 00:00:00', '2020-06-03 23:00:00'),
                ('2020-08-10~12', '2020-08-10 00:00:00', '2020-08-12 23:00:00'),
                ('2021-07-01~03', '2021-07-01 00:00:00', '2021-07-03 23:00:00'),
                ('2022-08-01~03', '2022-08-01 00:00:00', '2022-08-03 23:00:00'),
                ('2023-02-25~27', '2023-02-25 00:00:00', '2023-02-27 23:00:00'),
            ]
            storm_periods = [
                ('2020-09-28~30', '2020-09-28 00:00:00', '2020-09-30 23:00:00'),
                ('2022-02-03~05', '2022-02-03 00:00:00', '2022-02-05 23:00:00'),
                ('2021-05-12~14', '2021-05-12 00:00:00', '2021-05-14 23:00:00'), 
                ('2024-02-08~10', '2024-02-08 01:00:00', '2024-02-10 23:00:00'),
                ('2024-12-01~03', '2024-12-01 00:00:00', '2024-12-03 23:00:00'),
            ]

            use_initial = input("\n是否启用有原重构（第一个时间步使用真实NC文件）？(y/n): ").strip().lower() == 'y'

            # 处理平静期
            quiet_results = []
            for period_name, start_time, end_time in quiet_periods:
                result = process_period(period_name, start_time, end_time, 'quiet', use_initial_iono=use_initial)
                if result is not None:
                    quiet_results.append(result)

            # 处理磁暴期
            storm_results = []
            for period_name, start_time, end_time in storm_periods:
                result = process_period(period_name, start_time, end_time, 'storm', use_initial_iono=use_initial)
                if result is not None:
                    storm_results.append(result)

            # 绘图
            plt.style.use('seaborn-v0_8-whitegrid')
            for result in quiet_results + storm_results:
                period_name = result['period_name']
                timestamps = result['timestamps']
                region_results_std = result['region_results_std']
                region_results_init = result['region_results_init']
                anomaly_std = result['anomaly_count_std']
                anomaly_init = result['anomaly_count_init']

                n_regions = len(region_results_std)
                fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
                if n_regions == 1:
                    axes = [axes]

                for ax, (region_name, region_data) in zip(axes, region_results_std.items()):
                    ax.plot(timestamps, region_data['tec_true'], 'o-',
                            label='Trained Data TEC', markersize=3, linewidth=1.5, color='blue')

                    valid_mask_std = ~np.isnan(region_data['tec_pred'])
                    if np.any(valid_mask_std):
                        valid_times = [timestamps[i] for i in range(len(timestamps)) if valid_mask_std[i]]
                        valid_pred_std = [region_data['tec_pred'][i] for i in range(len(timestamps)) if valid_mask_std[i]]
                        ax.plot(valid_times, valid_pred_std, 'x--',
                                label='PTIM (No initial)', markersize=3, linewidth=1.5, color='red')

                    if region_results_init is not None:
                        init_data = region_results_init[region_name]
                        valid_mask_init = ~np.isnan(init_data['tec_pred'])
                        if np.any(valid_mask_init):
                            valid_times_init = [timestamps[i] for i in range(len(timestamps)) if valid_mask_init[i]]
                            valid_pred_init = [init_data['tec_pred'][i] for i in range(len(timestamps)) if valid_mask_init[i]]
                            ax.plot(valid_times_init, valid_pred_init, 's--',
                                    label='PTIM (With initial)', markersize=3, linewidth=1.5, color='green')

                    ax.set_title(f'{period_name} - {region_name}')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Area-Averaged TEC (TECU)')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)

                    y_max_true = max(region_data['tec_true']) if region_data['tec_true'] else 10
                    y_max = min(y_max_true * 1.2, 80)
                    ax.set_ylim(0, y_max)

                title_suffix = f"异常: 标准{anomaly_std}次"
                if region_results_init:
                    title_suffix += f", 有原{anomaly_init}次"
                fig.suptitle(f'{period_name} ({title_suffix})', fontsize=10, y=1.02)

                plt.tight_layout()
                suffix = "_with_initial" if region_results_init else "_no_initial"
                save_name = f'tec_comparison_{result["period_type"]}_{period_name.replace("~", "_")}{suffix}.png'
                plt.savefig(save_name, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"对比图已保存为 {save_name}")

            print("\n短时验证完成！")

        elif choice == '2':
            # 长时无原重构
            print("\n" + "="*60)
            print("长时无原重构（纯物理驱动，逐日平均）")
            print("="*60)

            # 选择持续时间
            print("可选持续时间：90天, 180天, 365天")
            duration_input = input("请输入天数 (90/180/365): ").strip()
            try:
                days = int(duration_input)
                if days not in [90, 180, 365]:
                    print("天数无效，将使用90天")
                    days = 90
            except:
                days = 90
                print("输入无效，将使用90天")

            # 选择起始日期
            start_input = input(f"请输入起始日期 (格式 YYYY-MM-DD, 默认 2020-01-01): ").strip()
            if not start_input:
                start_date = '2020-01-01'
            else:
                start_date = start_input
            # 计算结束日期
            start_dt = pd.to_datetime(start_date)
            end_dt = start_dt + timedelta(days=days - 1)
            end_date = end_dt.strftime('%Y-%m-%d')
            period_name = f"{start_date} 至 {end_date} ({days}天)"

            # 询问是否平滑
            smooth_input = input("是否对日平均序列进行滑动平均平滑？(y/n, 默认 y): ").strip().lower()
            if smooth_input == 'n':
                smooth_window = 1
            else:
                smooth_window = 3  # 3天滑动平均

            # 检查数据可用性
            try:
                long_term_reconstruction(period_name, start_date, end_date, regions, save_prefix=f"{days}d", smooth_window=smooth_window)
                print(f"长时重构完成，图片已保存。")
            except Exception as e:
                print(f"长时重构失败: {e}")
                print("请检查起始日期是否在数据覆盖范围内（2019-08-19 至 2025-05-10）")
        elif choice == '3':
            generate_time_range_plots()
        elif choice == '4':
            plot_error_statistics()
        elif choice == '5':
            plot_peak_tec_by_ut_all_cols()
        elif choice == '6':
            generate_time_range_tec_plots()
        # 在菜单项 7 中调用
        elif choice == '7':
            sta_lat = 30.54
            sta_lon = 114.36
            plot_fof2_comparison(sta='WU430', year=2021, sta_lat=sta_lat, sta_lon=sta_lon,
                                data_dir='.', output_dir='.')
        elif choice == '8':
            plot_height_Ne_by_ut()
        elif choice == '9':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
