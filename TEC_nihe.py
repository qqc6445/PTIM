"""
reconstruct_two_regions.py
使用训练好的模型（仅物理输入）重构电离层密度，计算东八区和西四区的 TEC 热力图。
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import gc
import warnings
import json
import pickle
from tqdm import tqdm  # 可选，用于进度条

warnings.filterwarnings("ignore")

# ==================== 模型定义（必须与训练时一致） ====================
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


# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型路径
model_path = r"E:\daima\2026.3.27\best_physics_to_iono.pth"

# 数据路径
physics_files = {
    2019: r"F:\wenjian\2019_231-2019_365.xlsx",
    2020: r"F:\wenjian\2020_001-2020_366.xlsx",
    2021: r"F:\wenjian\2021_001-2021_365.xlsx",
    2022: r"F:\wenjian\2022_001-2022_365.xlsx",
    2023: r"F:\wenjian\2023_001-2023_365.xlsx",
    2024: r"F:\wenjian\2024_001-2024_366.xlsx",
    2025: r"F:\wenjian\2025_001-2025_131.xlsx",
}

ionosphere_folders = {
    2019: r"F:\wenjian\2019",
    2020: r"F:\wenjian\2020",
    2021: r"F:\wenjian\2021",
    2022: r"F:\wenjian\2022",
    2023: r"F:\wenjian\2023",
    2024: r"F:\wenjian\2024",
    2025: r"F:\wenjian\2025",
}

# 时间范围（与训练数据一致）
start_date = '2019-08-19'
end_date = '2025-05-10'

# 区域定义
regions = {
    'east8': {
        'name': 'East8 (112.5°E-127.5°E)',
        'lon_range': (112.5, 127.5),
        'target_hours': [4, 5, 6],           # UT 4,5,6 对应地方时 12,13,14
        'output_fig': 'TEC_nihe_east8_heatmap.png'
    },
    'west4': {
        'name': 'West4 (60°W-45°W)',
        'lon_range': (-60, -45),
        'target_hours': [16, 17, 18],        # UT 16,17,18 对应地方时 12,13,14
        'output_fig': 'TEC_nihe_west4_heatmap.png'
    }
}

# 网格参数
N_HEIGHT = 51
N_LAT = 73
N_LON = 72

lat_full = np.linspace(-90, 90, N_LAT)          # 原始纬度网格
lon_full = np.linspace(0, 360, N_LON)            # 原始经度网格（0~360）
heights_km = np.arange(100, 100 + N_HEIGHT*20, 20)  # 高度 km

# 绘图纬度网格（间隔1°）
lat_grid = np.arange(-90, 91, 1.0)

# ==================== 加载归一化和特征工程工具 ====================
def replace_outliers_iqr(data, iqr_mult=1.5):
    """
    使用 IQR 检测一维数组的异常值，并用线性插值替换。
    data: 一维 numpy 数组
    iqr_mult: IQR 倍数，默认 1.5
    返回替换后的数组。
    """
    data = data.copy()
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr
    upper = q3 + iqr_mult * iqr
    outliers = (data < lower) | (data > upper)
    if not np.any(outliers):
        return data
    # 用线性插值替换异常值
    x = np.arange(len(data))
    valid = ~outliers & ~np.isnan(data)
    if np.sum(valid) < 2:
        # 有效点太少，用中位数填充
        median_val = np.nanmedian(data)
        data[outliers] = median_val
    else:
        from scipy.interpolate import interp1d
        interp_func = interp1d(x[valid], data[valid], kind='linear',
                                bounds_error=False, fill_value='extrapolate')
        data[outliers] = interp_func(x[outliers])
    return data
def load_normalization_params():
    """加载电离层归一化参数和物理特征scaler"""
    # 电离层归一化参数
    norm_path = r"E:\daima\data_cache\ionosphere_stats_info.json"
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"未找到电离层归一化参数文件: {norm_path}")
    with open(norm_path, 'r') as f:
        stats = json.load(f)
    mean = stats['mean']
    std = stats['std']
    print(f"电离层归一化参数: mean={mean:.6f}, std={std:.6f}")

    # 物理特征scaler
    scaler_path = r"E:\daima\2026.3.27\physics_scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到物理特征scaler文件: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 特征名称
    feat_path = r"E:\daima\2026.3.27\feature_names.json"
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            feature_names = json.load(f)
        print(f"加载特征名称: {len(feature_names)}个")
    else:
        feature_names = None
        print("警告: 未找到特征名称文件，将使用scaler的均值长度")

    return mean, std, scaler, feature_names


def feature_engineering(df):
    """特征工程（与训练时完全一致）"""
    df = df.copy()
    # 时间特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    # 周期特征
    df['year_sin'] = np.sin(2 * np.pi * (df['year'] - 2019) / 11)
    df['year_cos'] = np.cos(2 * np.pi * (df['year'] - 2019) / 11)

    # 组合特征
    df['B_total'] = np.sqrt(df['BX']**2 + df['BY']**2 + df['BZ']**2)
    df['vBs'] = df['SW_speed'] * df['BZ'].clip(upper=0)

    # 滞后特征
    for col in ['BX', 'BY', 'BZ', 'Dst', 'f107', 'AE']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(0)
        df[f'{col}_lag2'] = df[col].shift(2).fillna(0)
        df[f'{col}_lag3'] = df[col].shift(3).fillna(0)

    # 滚动统计
    for col in ['Dst', 'ap', 'AE', 'f107']:
        df[f'{col}_mean3'] = df[col].rolling(3, min_periods=1).mean().fillna(0)
        df[f'{col}_mean6'] = df[col].rolling(6, min_periods=1).mean().fillna(0)
        df[f'{col}_std3'] = df[col].rolling(3, min_periods=1).std().fillna(0)

    # 差分特征
    for col in ['Dst', 'ap', 'AE']:
        df[f'{col}_diff1'] = df[col].diff(1).fillna(0)
        df[f'{col}_diff3'] = df[col].diff(3).fillna(0)

    return df


def load_physics_year(year):
    """加载一年物理指数并生成特征矩阵（归一化后）"""
    filepath = physics_files.get(year)
    if not filepath or not os.path.exists(filepath):
        return None
    df = pd.read_excel(filepath, header=None)
    if df.shape[1] < 11:
        # 列数不足，填充缺失项
        for _ in range(11 - df.shape[1]):
            df[df.shape[1]] = 0
    df.columns = ['year', 'doy', 'hour', 'BX', 'BY', 'BZ', 'SW_speed', 'Dst', 'ap', 'f107', 'AE']
    # 转换为整数
    for col in ['year', 'doy', 'hour']:
        df[col] = df[col].astype(int)

    # 创建时间索引（用于对齐）
    start = pd.Timestamp(f"{year}-01-01") if year != 2019 else pd.Timestamp("2019-08-19")
    dates = pd.date_range(start=start, periods=len(df), freq='H')
    df.index = dates

    # 特征工程
    df = feature_engineering(df)

    # 确保特征顺序与训练时一致
    if 'feature_names' in globals() and feature_names is not None:
        # 只保留训练时用到的特征
        df = df[feature_names]

    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df


def get_physics_for_period(start, end):
    """获取指定日期范围的物理指数（已归一化）"""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    years = range(start.year, end.year + 1)
    all_dfs = []
    for year in years:
        df_year = load_physics_year(year)
        if df_year is None:
            continue
        mask = (df_year.index >= start) & (df_year.index <= end)
        df_slice = df_year[mask]
        if len(df_slice) > 0:
            all_dfs.append(df_slice)
    if not all_dfs:
        raise ValueError(f"未找到 {start} 到 {end} 的物理数据")
    df_combined = pd.concat(all_dfs).sort_index()
    # 补全缺失小时
    full_index = pd.date_range(start=start, end=end, freq='H')
    df_combined = df_combined.reindex(full_index).fillna(method='ffill').fillna(method='bfill').fillna(0)
    # 归一化
    values = df_combined.values
    scaler = globals()['scaler']  # 外部加载的scaler
    return scaler.transform(values)


def denormalize_ne(ne_norm, mean, std):
    """反归一化电子密度（原始单位 1e5 el/cm³）"""
    # 裁剪输入防止异常
    ne_norm = np.clip(ne_norm, -5, 10)
    log_ne = ne_norm * std + mean
    ne = np.expm1(log_ne)
    ne = np.clip(ne, 0, 1e7)   # 物理上限
    return ne


def calculate_tec_trapezoid(ne_profile):
    """梯形积分计算TEC，ne_profile: (height,) 单位 1e5 el/cm³，返回 TECu"""
    # 转换为 el/m³: 1e5 el/cm³ * 1e6 = 1e11
    ne_m3 = ne_profile * 1e11
    # 高度转换为米
    heights_m = heights_km * 1000.0
    delta_h = np.diff(heights_m)
    avg_ne = (ne_m3[:-1] + ne_m3[1:]) / 2.0
    tec_elm2 = np.sum(avg_ne * delta_h)
    return tec_elm2 / 1e16


def compute_tec_for_region(ne_3d, lon_range):
    """
    计算指定经度范围的平均TEC纬度剖面
    ne_3d: (height, lat, lon) 原始单位 1e5 el/cm³
    lon_range: (min, max) 经度范围（度）
    返回: tec_profile (n_lat,) 每个纬度的平均TEC
    """
    # 找到经度索引
    lon_min, lon_max = lon_range
    # 转换为0~360
    lon_min_mod = lon_min % 360
    lon_max_mod = lon_max % 360
    if lon_min_mod <= lon_max_mod:
        lon_mask = (lon_full >= lon_min_mod) & (lon_full <= lon_max_mod)
    else:
        lon_mask = (lon_full >= lon_min_mod) | (lon_full <= lon_max_mod)
    lon_indices = np.where(lon_mask)[0]
    if len(lon_indices) == 0:
        return np.full(N_LAT, np.nan)

    # 提取该区域所有经度的数据
    ne_region = ne_3d[:, :, lon_indices]  # (height, lat, n_lon)
    # 对经度平均
    ne_lat_mean = np.mean(ne_region, axis=2)  # (height, lat)
    # 计算每个纬度的TEC
    tec_profile = []
    for ilat in range(N_LAT):
        ne_profile = ne_lat_mean[:, ilat]
        tec = calculate_tec_trapezoid(ne_profile)
        tec_profile.append(tec)
    return np.array(tec_profile)


def main():
    print("="*60)
    print("开始重构电离层电子密度并计算TEC")
    print("="*60)

    # 1. 加载模型
    print("\n加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    physics_feature_dim = checkpoint['physics_feature_dim']
    model = PhysicsToIonosphereModel(physics_feature_dim=physics_feature_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ 模型加载成功，物理特征维度: {physics_feature_dim}")

    # 2. 加载归一化参数和scaler
    global mean, std, scaler, feature_names
    mean, std, scaler, feature_names = load_normalization_params()

    # 3. 生成所有小时列表
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_hours = int((end - start).total_seconds() / 3600) + 1
    all_datetimes = [start + pd.Timedelta(hours=i) for i in range(total_hours)]
    print(f"时间范围: {start_date} 至 {end_date}，共 {total_hours} 小时")

    # 4. 分年处理，避免一次性加载所有物理数据
    # 初始化每个区域的按日数据收集器
    region_daily_data = {region: {} for region in regions}  # 每个区域: {date: [tec_profiles]}

    # 按年份循环
    years = sorted(set(dt.year for dt in all_datetimes))
    for year in years:
        print(f"\n处理年份 {year}...")
        # 该年的起止时间
        year_start = max(start, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end, pd.Timestamp(f"{year}-12-31"))
        year_hours = int((year_end - year_start).total_seconds() / 3600) + 1

        # 获取该年物理数据（需要覆盖历史窗口4小时）
        phys_start = year_start - pd.Timedelta(hours=4)
        phys_end = year_end
        physics_all = get_physics_for_period(phys_start, phys_end)
        print(f"  物理数据形状: {physics_all.shape}")

        # 初始化历史窗口（全零，归一化值）
        history = np.zeros((4, N_HEIGHT, N_LAT, N_LON), dtype=np.float32)

        # 逐小时预测
        for step in tqdm(range(year_hours), desc=f"  {year}年预测"):
            # 当前绝对小时索引
            current_dt = year_start + pd.Timedelta(hours=step)
            # 物理数据索引：从 step 开始取4小时
            start_idx = step
            end_idx = step + 4
            if end_idx > len(physics_all):
                break
            physics_seq = physics_all[start_idx:end_idx]  # (4, F)
            physics_tensor = torch.FloatTensor(physics_seq).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(physics_tensor).cpu().numpy().squeeze()  # (H, L, W)
            # 反归一化得到真实电子密度
            pred_real = denormalize_ne(pred_norm, mean, std)

            # 对每个区域，如果当前小时属于该区域的目标小时，则计算TEC剖面
            for region_name, reg in regions.items():
                if current_dt.hour in reg['target_hours']:
                    tec_profile = compute_tec_for_region(pred_real, reg['lon_range'])
                    # 异常值剔除与替换
                    tec_profile = replace_outliers_iqr(tec_profile, iqr_mult=1.5)
                    date_key = current_dt.date()
                    if date_key not in region_daily_data[region_name]:
                        region_daily_data[region_name][date_key] = []
                    region_daily_data[region_name][date_key].append(tec_profile)
            # 更新历史（实际不需要，因为模型不依赖历史，但保留以备将来）
            # 这里我们不需要滚动历史，因为模型只依赖物理序列，不依赖电离层历史
            # 所以 history 无用，但为了代码完整性保留，但不使用

        # 释放内存
        del physics_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. 对每个区域生成热力图
    for region_name, reg in regions.items():
        print(f"\n处理区域: {reg['name']}")
        daily_dict = region_daily_data[region_name]
        if not daily_dict:
            print("  无有效数据，跳过")
            continue

        # 收集每日平均TEC剖面
        daily_tec_profiles = []
        daily_dates = []
        for date in sorted(daily_dict.keys()):
            profiles = daily_dict[date]
            if not profiles:
                continue
            avg_profile = np.nanmean(profiles, axis=0)  # 对三个小时平均
            daily_tec_profiles.append(avg_profile)
            daily_dates.append(date)

        if not daily_tec_profiles:
            print("  无有效日均数据，跳过")
            continue

        tec_matrix = np.array(daily_tec_profiles).T  # (n_lat, n_days)
        print(f"  TEC矩阵形状: {tec_matrix.shape}")

        # 插值到统一纬度网格（lat_grid）
        # 原始纬度 lat_full 是73个点，需要插值到 lat_grid (181个点)
        tec_interp = np.zeros((len(lat_grid), tec_matrix.shape[1]))
        for i in range(tec_matrix.shape[1]):
            f = interp1d(lat_full, tec_matrix[:, i], kind='linear', bounds_error=False, fill_value=np.nan)
            tec_interp[:, i] = f(lat_grid)

        # 色标范围：0-99%分位数
        valid_vals = tec_interp[~np.isnan(tec_interp)]
        if len(valid_vals) == 0:
            print("  无有效TEC值，跳过")
            continue
        vmin = 0
        vmax = 25.25
        print(f"  色标范围: {vmin:.2f} - {vmax:.2f} TECu")

        # 绘图
        plt.figure(figsize=(14, 8))
        X, Y = np.meshgrid(np.arange(len(daily_dates)), lat_grid)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        masked_tec = np.ma.masked_invalid(tec_interp)
        im = plt.pcolormesh(X, Y, masked_tec, cmap='RdBu_r', norm=norm, shading='auto')
        # x轴刻度
        # 生成半年刻度（从第一个月开始，每6个月）
        from datetime import datetime

        first_date = daily_dates[0]
        last_date = daily_dates[-1]

        first_dt = datetime(first_date.year, first_date.month, 1)
        last_dt = datetime(last_date.year, last_date.month, 1)

        tick_dts = []
        current = first_dt
        while current <= last_dt:
            tick_dts.append(current)
            year = current.year + (current.month + 6 - 1) // 12
            month = (current.month + 6 - 1) % 12 + 1
            current = datetime(year, month, 1)

        if tick_dts[-1] < last_dt:
            tick_dts.append(last_dt)

        xticks = []
        xticklabels = []
        for d in tick_dts:
            diff = [abs((datetime.combine(dt, datetime.min.time()) - d).total_seconds()) for dt in daily_dates]
            idx = np.argmin(diff)
            xticks.append(idx)
            xticklabels.append(daily_dates[idx].strftime('%Y-%m'))

        unique = {}
        xticks_u = []
        xlabels_u = []
        for pos, lab in zip(xticks, xticklabels):
            if pos not in unique:
                unique[pos] = lab
                xticks_u.append(pos)
                xlabels_u.append(lab)

        plt.xticks(xticks_u, xlabels_u, rotation=0, fontsize=8)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Latitude (°)', fontsize=12)
        plt.title(f'Daily Average TEC over {reg["name"]}\n'
                  f'PTIM,Averaged from LT 12,13,14', fontsize=14)
        cbar = plt.colorbar(im, label='TEC (TECu)')
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(reg['output_fig'], dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 热力图已保存: {reg['output_fig']}")

    print("\n✅ 所有区域处理完成！")


if __name__ == "__main__":
    main()