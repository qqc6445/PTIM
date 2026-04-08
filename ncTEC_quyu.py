import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
# 数据文件夹路径（2019-2025）
base_folders = [f"F:\\wenjian\\{year}" for year in range(2019, 2026)]

# 定义两个区域
regions = {
    'east8': {
        'name': 'East8 (112.5°E-127.5°E)',
        'lon_range': (112.5, 127.5),       # 经度范围
        'target_hours': [4, 5, 6],         # 世界时（对应地方时12,13,14）
        'output_fig': 'TEC_guance_east8_heatmap.png',
        'output_ts': 'TEC_guance_east8_time_series.png',
        'output_lat': 'TEC_guance_east8_latitude_profiles.png'
    },
    'west4': {
        'name': 'West4 (60°W-45°W)',
        'lon_range': (-60, -45),           # 西经60°~45°
        'target_hours': [16, 17, 18],      # 世界时（对应地方时12,13,14）
        'output_fig': 'TEC_guance_west4_heatmap.png',
        'output_ts': 'TEC_guance_west4_time_series.png',
        'output_lat': 'TEC_guance_west4_latitude_profiles.png'
    }
}

# 纬度范围（全纬度 -90° ~ 90°）
lat_min, lat_max = -90.0, 90.0

# 统一纬度网格（间隔1°，用于绘图）
lat_grid = np.arange(lat_min, lat_max + 0.5, 1.0)

# 网格维度信息（根据文件实际形状）
N_HEIGHT = 51  # 高度层数
N_LAT = 73     # 纬度点数
N_LON = 72     # 经度点数

# ==================== 辅助函数 ====================
def parse_filename(filename):
    """从文件名中提取年份、年积日和小时。"""
    basename = os.path.basename(filename)
    name_without_ext = basename.split('.')[0]
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        year = int(parts[-3])
        doy = int(parts[-2])
        hour = int(parts[-1])
        return year, doy, hour
    return None, None, None

def datetime_from_doy(year, doy, hour):
    """根据年积日构建datetime对象"""
    base = datetime(year, 1, 1)
    return base + timedelta(days=doy - 1, hours=hour)

def get_coordinates(ds):
    """从数据集获取经纬度和高度坐标，若缺失则生成默认值"""
    # 尝试获取纬度
    lat_values = None
    for v in ['latitude', 'lat', 'Latitude']:
        if v in ds.coords:
            lat_values = ds[v].values
            break
        elif v in ds.variables:
            lat_values = ds[v].values
            break
    if lat_values is None:
        lat_values = np.linspace(-90, 90, N_LAT)

    # 尝试获取经度
    lon_values = None
    for v in ['longitude', 'lon', 'Longitude']:
        if v in ds.coords:
            lon_values = ds[v].values
            break
        elif v in ds.variables:
            lon_values = ds[v].values
            break
    if lon_values is None:
        lon_values = np.linspace(0, 360, N_LON)

    # 尝试获取高度
    height_values = None
    for v in ['height', 'altitude', 'z']:
        if v in ds.coords:
            height_values = ds[v].values
            break
        elif v in ds.variables:
            height_values = ds[v].values
            break
    if height_values is None:
        # 假设高度从100km到1100km，间隔20km
        height_values = np.linspace(100, 100 + (N_HEIGHT-1)*20, N_HEIGHT)
    return lat_values, lon_values, height_values

def calculate_tec_trapezoid(ne_profile, heights_km):
    """
    使用梯形积分计算单个剖面的TEC
    ne_profile: 电子密度剖面，单位原始文件中的 1e5 el/cm³
    heights_km: 高度数组，单位 km
    返回 TEC (TECu)
    """
    # 1. 转换为 el/m³：原始值 * 1e5 (单位修正) * 1e6 (cm³/m³) = 乘以 1e11
    ne_m3 = ne_profile * 1e11  # el/m³
    
    # 2. 高度转换为米
    heights_m = heights_km * 1000.0
    
    # 3. 梯形积分
    delta_h = np.diff(heights_m)                     # 相邻高度差（米）
    avg_ne = (ne_m3[:-1] + ne_m3[1:]) / 2.0          # 相邻层平均电子密度
    tec_elm2 = np.sum(avg_ne * delta_h)              # el/m²
    
    # 4. 转换为 TECu
    tec = tec_elm2 / 1e16
    return tec

def replace_outliers_iqr(data, iqr_mult=1.5):
    """
    使用 IQR 检测异常值，并用线性插值替换。
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
        # 如果有效点太少，用中位数填充
        median_val = np.nanmedian(data)
        data[outliers] = median_val
    else:
        interp_func = interp1d(x[valid], data[valid], kind='linear', bounds_error=False, fill_value='extrapolate')
        data[outliers] = interp_func(x[outliers])
    return data

def get_tec_for_region(filepath, lon_range, lat_min, lat_max):
    """
    从单个NC文件提取指定经度范围和纬度范围的TEC（对经度平均）
    返回 (latitudes, tec_values, file_time)
    """
    try:
        ds = xr.open_dataset(filepath, decode_times=False)
    except Exception as e:
        return None, None, None

    # 获取时间
    year, doy, hour = parse_filename(filepath)
    if year is None:
        ds.close()
        return None, None, None
    file_time = datetime_from_doy(year, doy, hour)

    # 获取电子密度数据
    if 'Ne' not in ds.variables:
        ds.close()
        return None, None, None
    ne_data = ds['Ne'].values  # 形状 (height, lat, lon)

    # 检查维度
    if ne_data.shape != (N_HEIGHT, N_LAT, N_LON):
        ds.close()
        return None, None, None

    # 获取坐标
    lat_values, lon_values, height_values = get_coordinates(ds)
    ds.close()

    # 处理经度范围
    lon_min, lon_max = lon_range
    if np.min(lon_values) >= 0 and np.max(lon_values) <= 360:
        lon_min_mod = lon_min % 360
        lon_max_mod = lon_max % 360
        if lon_min_mod <= lon_max_mod:
            lon_mask = (lon_values >= lon_min_mod) & (lon_values <= lon_max_mod)
        else:
            lon_mask = (lon_values >= lon_min_mod) | (lon_values <= lon_max_mod)
    else:
        lon_mask = (lon_values >= lon_min) & (lon_values <= lon_max)
    lon_indices = np.where(lon_mask)[0]
    if len(lon_indices) == 0:
        return None, None, None

    # 处理纬度范围
    lat_mask = (lat_values >= lat_min) & (lat_values <= lat_max)
    if not np.any(lat_mask):
        return None, None, None
    selected_lats = lat_values[lat_mask]
    lat_indices = np.where(lat_mask)[0]

    # 对每个纬度，计算该纬度下所有经度的平均 TEC
    tec_values = []
    for ilat in lat_indices:
        ne_at_lat = ne_data[:, ilat, lon_indices]  # (height, n_lon)
        tec_at_lon = []
        for i in range(len(lon_indices)):
            ne_profile = ne_at_lat[:, i]
            tec = calculate_tec_trapezoid(ne_profile, height_values)
            tec_at_lon.append(tec)
        mean_tec = np.mean(tec_at_lon)
        tec_values.append(mean_tec)

    tec_values = np.array(tec_values)

    # 基础过滤：超出物理范围置为 NaN
    tec_values = np.where(tec_values > 200, np.nan, tec_values)
    tec_values = np.where(tec_values < 0, np.nan, tec_values)

    # 异常值检测与替换（基于 IQR）
    tec_values = replace_outliers_iqr(tec_values, iqr_mult=1.5)

    return selected_lats, tec_values, file_time

# ==================== 主处理循环 ====================
# 初始化每个区域的数据存储
region_data = {region: [] for region in regions}

print("开始处理NC文件...")
processed_count = 0
skipped_count = 0
file_count = 0

for folder in base_folders:
    if not os.path.exists(folder):
        print(f"警告：文件夹 {folder} 不存在，跳过")
        continue

    nc_files = glob.glob(os.path.join(folder, "*.nc"))
    print(f"处理文件夹 {folder}，找到 {len(nc_files)} 个NC文件")
    file_count += len(nc_files)

    for filepath in nc_files:
        # 解析小时
        _, _, hour = parse_filename(filepath)
        if hour is None:
            skipped_count += 1
            continue

        # 检查该小时属于哪个区域的目标小时（可属于多个区域，但不同区域的目标小时不同，一般不会重叠）
        matched_regions = []
        for region_name, reg in regions.items():
            if hour in reg['target_hours']:
                matched_regions.append(region_name)

        if not matched_regions:
            continue

        # 对每个匹配的区域，提取TEC数据
        for region_name in matched_regions:
            reg = regions[region_name]
            lats, tec, file_time = get_tec_for_region(filepath, reg['lon_range'], lat_min, lat_max)
            if lats is None or tec is None:
                skipped_count += 1
                continue

            # 插值到统一纬度网格
            interp_func = interp1d(lats, tec, kind='linear', bounds_error=False, fill_value=np.nan)
            tec_interp = interp_func(lat_grid)

            date_key = file_time.date()
            region_data[region_name].append((date_key, tec_interp))

        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"已处理 {processed_count} 个文件...")

print(f"\n处理完成！")
print(f"总共扫描文件: {file_count}")
print(f"成功处理: {processed_count} 个文件")
print(f"跳过: {skipped_count} 个文件")

# ==================== 对每个区域分别处理 ====================
for region_name, reg in regions.items():
    print(f"\n处理区域：{reg['name']}")
    data_list = region_data[region_name]
    if not data_list:
        print(f"  没有有效数据，跳过")
        continue

    # 按日期分组
    daily_data = {}
    for date, profile in data_list:
        if date not in daily_data:
            daily_data[date] = []
        daily_data[date].append(profile)

    # 计算每日平均（允许部分小时）
    daily_tec_profiles = []
    daily_dates = []
    all_tec_values = []

    for date, profiles in sorted(daily_data.items()):
        if not profiles:
            continue
        # 对现有有效小时取平均
        avg_profile = np.nanmean(profiles, axis=0)
        daily_tec_profiles.append(avg_profile)
        daily_dates.append(date)
        all_tec_values.extend(avg_profile[~np.isnan(avg_profile)])

    if not daily_tec_profiles:
        print(f"  没有完整的日均数据，跳过")
        continue

    print(f"  得到 {len(daily_dates)} 个日均数据")

    # 构建TEC矩阵（纬度 × 时间）
    tec_matrix = np.array(daily_tec_profiles).T
    time_axis = daily_dates

    # 确定色标范围（0-99%百分位数）
    if all_tec_values:
        tec_vmin = 0
        tec_vmax = 25.25
        print(f"  色标范围: {tec_vmin:.2f} - {tec_vmax:.2f} TECu")
    else:
        tec_vmin = np.nanmin(tec_matrix)
        tec_vmax = np.nanmax(tec_matrix)

    # 绘制热力图（红蓝配色）
    plt.figure(figsize=(14, 8))
    time_numeric = [datetime.combine(d, datetime.min.time()) for d in time_axis]
    X, Y = np.meshgrid(np.arange(len(time_numeric)), lat_grid)

    norm = plt.Normalize(vmin=tec_vmin, vmax=tec_vmax)
    masked_tec = np.ma.masked_invalid(tec_matrix)
    im = plt.pcolormesh(X, Y, masked_tec, cmap=plt.cm.RdBu_r, norm=norm, shading='auto')

   # 生成半年刻度（从第一个月开始，每6个月）
    first_date = time_numeric[0]
    last_date = time_numeric[-1]

    tick_dates = []
    current = first_date.replace(day=1)
    while current <= last_date:
        tick_dates.append(current)
        year = current.year + (current.month + 6 - 1) // 12
        month = (current.month + 6 - 1) % 12 + 1
        current = current.replace(year=year, month=month)

    if tick_dates[-1] < last_date:
        tick_dates.append(last_date)

    xtick_positions = []
    xtick_labels = []
    for d in tick_dates:
        idx = min(range(len(time_numeric)), key=lambda i: abs((time_numeric[i] - d).total_seconds()))
        xtick_positions.append(idx)
        xtick_labels.append(time_numeric[idx].strftime('%Y-%m'))

    unique = {}
    xticks_unique = []
    xlabels_unique = []
    for pos, lab in zip(xtick_positions, xtick_labels):
        if pos not in unique:
            unique[pos] = lab
            xticks_unique.append(pos)
            xlabels_unique.append(lab)

    plt.xticks(xticks_unique, xlabels_unique, rotation=0, fontsize=8)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Latitude (°)', fontsize=12)
    plt.title(f'Daily Average TEC over {reg["name"]}\n'
              f'Trained Data,Averaged from LT 12,13,14 ', fontsize=14)
    cbar = plt.colorbar(im, label='TEC (TECu)')
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(reg['output_fig'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  热力图已保存: {reg['output_fig']}")

    # 可选：时间序列图
    mean_tec_over_lat = np.nanmean(tec_matrix, axis=0)
    plt.figure(figsize=(12, 5))
    plt.plot(time_numeric, mean_tec_over_lat, 'b-', linewidth=1.5, alpha=0.7)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Mean TEC (TECu)', fontsize=12)
    plt.title(f'Mean TEC Time Series over {reg["name"]}\n(UT {reg["target_hours"]} averaged)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(xtick_positions, xtick_labels, rotation=45, fontsize=8)
    stats_text = f'Mean: {np.nanmean(mean_tec_over_lat):.2f} TECu\n'
    stats_text += f'Std: {np.nanstd(mean_tec_over_lat):.2f} TECu\n'
    stats_text += f'Min: {np.nanmin(mean_tec_over_lat):.2f} TECu\n'
    stats_text += f'Max: {np.nanmax(mean_tec_over_lat):.2f} TECu'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(reg['output_ts'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  时间序列图已保存: {reg['output_ts']}")

    # 可选：纬度剖面图
    plt.figure(figsize=(12, 6))
    spring_dates = [i for i, d in enumerate(time_axis) if d.month in [3,4,5]]
    summer_dates = [i for i, d in enumerate(time_axis) if d.month in [6,7,8]]
    autumn_dates = [i for i, d in enumerate(time_axis) if d.month in [9,10,11]]
    winter_dates = [i for i, d in enumerate(time_axis) if d.month in [12,1,2]]

    if spring_dates:
        plt.plot(lat_grid, np.nanmean(tec_matrix[:, spring_dates], axis=1), label='Spring (MAM)', linewidth=2)
    if summer_dates:
        plt.plot(lat_grid, np.nanmean(tec_matrix[:, summer_dates], axis=1), label='Summer (JJA)', linewidth=2)
    if autumn_dates:
        plt.plot(lat_grid, np.nanmean(tec_matrix[:, autumn_dates], axis=1), label='Autumn (SON)', linewidth=2)
    if winter_dates:
        plt.plot(lat_grid, np.nanmean(tec_matrix[:, winter_dates], axis=1), label='Winter (DJF)', linewidth=2)

    plt.xlabel('Latitude (°)', fontsize=12)
    plt.ylabel('TEC (TECu)', fontsize=12)
    plt.title(f'Seasonal Average TEC Latitudinal Profiles over {reg["name"]}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(reg['output_lat'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  纬度剖面图已保存: {reg['output_lat']}")

print("\n✅ 所有区域处理完成！")