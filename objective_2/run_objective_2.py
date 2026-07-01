import warnings
warnings.filterwarnings("ignore")

from matplotlib.patches import FancyArrowPatch
from scipy.signal import medfilt2d
from scipy.stats import pearsonr
from shapely.geometry import shape
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
import glob
import json
import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio.features
import warnings

# ------------------------------------------------------------------------------
# Phase 1: HCHO Data Loading and Regridding
# ------------------------------------------------------------------------------
def run_phase1_hcho_load():
    warnings.filterwarnings('ignore')
    
    REGRID_DIR = 'data/regridded'
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(os.path.join(REGRID_DIR, 'grid_spec.json')) as f:
        grid_spec = json.load(f)
    
    N_lat = grid_spec['nlat']
    N_lon = grid_spec['nlon']
    lon_min = grid_spec['lon_min']
    lat_max = grid_spec['lat_max']
    
    print("Loading land mask...")
    gadm_path = r"data\gadm41_IND_1.json\gadm41_IND_1.json"
    with open(gadm_path, 'r', encoding='utf-8') as f:
        gadm_data = json.load(f)
    india_geoms = [shape(feature['geometry']) for feature in gadm_data['features']]
    india_geom = unary_union(india_geoms)
    
    transform = rasterio.transform.from_origin(lon_min, lat_max, 0.25, 0.25)
    land_mask = rasterio.features.geometry_mask(
        [india_geom], out_shape=(N_lat, N_lon), transform=transform, invert=True
    )
    ocean_mask = ~land_mask
    
    start_date = np.datetime64('2023-10-01')
    end_date = np.datetime64('2024-01-31')
    date_range = np.arange(start_date, end_date + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    num_days = len(date_range)
    
    hcho_stack = np.full((num_days, N_lat, N_lon), np.nan, dtype=np.float32)
    dates_list = []
    
    print("Loading daily arrays...")
    for i, d in enumerate(date_range):
        d_str = str(d).replace('-', '')
        dates_list.append(str(d))
        file_path = os.path.join(REGRID_DIR, 'HCHO', f'HCHO_{d_str}.npy')
        if os.path.exists(file_path):
            hcho_stack[i] = np.load(file_path)
    
    print("Interpolating missing days in time...")
    def fill_gaps(stack, max_gap=3):
        out = stack.copy()
        for y in range(stack.shape[1]):
            for x in range(stack.shape[2]):
                s = pd.Series(stack[:, y, x])
                s_filled = s.interpolate(method='linear', limit=max_gap, limit_direction='both')
                out[:, y, x] = s_filled.values
        return out
    
    hcho_stack = fill_gaps(hcho_stack, max_gap=3)
    
    print("Applying median filter and computing valid pixel fraction...")
    filtered_stack = np.copy(hcho_stack)
    valid_pixel_fraction = np.zeros(num_days, dtype=np.float32)
    
    for i in range(num_days):
        slice_data = hcho_stack[i]
        
        valid_fraction = np.sum(~np.isnan(slice_data)) / (N_lat * N_lon)
        valid_pixel_fraction[i] = valid_fraction
        
        mean_val = np.nanmean(slice_data)
        if np.isnan(mean_val): mean_val = 0.0
        temp_slice = np.nan_to_num(slice_data, nan=mean_val)
        
        filtered_slice = medfilt2d(temp_slice, kernel_size=3)
        
        filtered_slice[np.isnan(slice_data)] = np.nan
        filtered_stack[i] = filtered_slice
    
    print("Applying land mask...")
    for i in range(num_days):
        filtered_stack[i][ocean_mask] = np.nan
    
    print("Saving results...")
    np.save(os.path.join(RESULTS_DIR, 'HCHO_daily_stack.npy'), filtered_stack)
    np.save(os.path.join(RESULTS_DIR, 'HCHO_valid_pixel_fraction.npy'), valid_pixel_fraction)
    with open(os.path.join(RESULTS_DIR, 'HCHO_dates.json'), 'w') as f:
        json.dump(dates_list, f)
    
    print("Phase 1 complete.")


# ------------------------------------------------------------------------------
# Phase 2: HCHO Baseline Statistics
# ------------------------------------------------------------------------------
def run_phase2_hcho_stats():
    warnings.filterwarnings('ignore')
    
    RESULTS_DIR = 'results'
    REGRID_DIR = 'data/regridded'
    
    print("Loading data...")
    hcho_stack = np.load(os.path.join(RESULTS_DIR, 'HCHO_daily_stack.npy'))
    with open(os.path.join(RESULTS_DIR, 'HCHO_dates.json'), 'r') as f:
        dates = json.load(f)
    valid_frac = np.load(os.path.join(RESULTS_DIR, 'HCHO_valid_pixel_fraction.npy'))
    
    with open(os.path.join(REGRID_DIR, 'grid_spec.json')) as f:
        grid_spec = json.load(f)
    
    # A. Overall Statistics
    print("Computing overall statistics...")
    mean_overall = np.nanmean(hcho_stack, axis=0)
    std_overall = np.nanstd(hcho_stack, axis=0)
    
    valid_pixels = hcho_stack[~np.isnan(hcho_stack)]
    pct_50, pct_75, pct_90, pct_95, pct_99 = np.percentile(valid_pixels, [50, 75, 90, 95, 99])
    global_min = np.nanmin(valid_pixels)
    global_max = np.nanmax(valid_pixels)
    global_mean = np.nanmean(valid_pixels)
    global_std = np.nanstd(valid_pixels)
    
    # B. Monthly means
    print("Computing monthly means...")
    dt_dates = pd.to_datetime(dates)
    months = dt_dates.month
    
    oct_mask = (months == 10)
    nov_mask = (months == 11)
    dec_mask = (months == 12)
    jan_mask = (months == 1)
    
    mean_oct = np.nanmean(hcho_stack[oct_mask], axis=0)
    mean_nov = np.nanmean(hcho_stack[nov_mask], axis=0)
    mean_dec = np.nanmean(hcho_stack[dec_mask], axis=0)
    mean_jan = np.nanmean(hcho_stack[jan_mask], axis=0)
    
    monthly_means = np.stack([mean_oct, mean_nov, mean_dec, mean_jan])
    
    # Save statistics JSON
    stats_dict = {
        'percentiles': {
            '50th': float(pct_50),
            '75th': float(pct_75),
            '90th': float(pct_90),
            '95th': float(pct_95),
            '99th': float(pct_99),
        },
        'global_min': float(global_min),
        'global_max': float(global_max),
        'global_mean': float(global_mean),
        'global_std': float(global_std),
        'monthly_mean_scalars': {
            'Oct': float(np.nanmean(mean_oct)),
            'Nov': float(np.nanmean(mean_nov)),
            'Dec': float(np.nanmean(mean_dec)),
            'Jan': float(np.nanmean(mean_jan)),
        }
    }
    with open(os.path.join(RESULTS_DIR, 'hcho_statistics.json'), 'w') as f:
        json.dump(stats_dict, f, indent=4)
    
    np.save(os.path.join(RESULTS_DIR, 'HCHO_monthly_means.npy'), monthly_means)
    
    # C. Anomalies
    print("Computing anomalies...")
    anomaly_overall = hcho_stack - mean_overall
    
    anomaly_monthly = np.empty_like(hcho_stack)
    anomaly_monthly[oct_mask] = hcho_stack[oct_mask] - mean_oct
    anomaly_monthly[nov_mask] = hcho_stack[nov_mask] - mean_nov
    anomaly_monthly[dec_mask] = hcho_stack[dec_mask] - mean_dec
    anomaly_monthly[jan_mask] = hcho_stack[jan_mask] - mean_jan
    
    np.save(os.path.join(RESULTS_DIR, 'HCHO_anomalies_overall.npy'), anomaly_overall)
    np.save(os.path.join(RESULTS_DIR, 'HCHO_anomalies_monthly.npy'), anomaly_monthly)
    
    # D. IGP time series
    print("Computing IGP time series...")
    lats = np.array(grid_spec['lats'])
    lons = np.array(grid_spec['lons'])
    
    lat_mask = (lats >= 28.0) & (lats <= 32.0)
    lon_mask = (lons >= 73.0) & (lons <= 80.0)
    
    igp_means = []
    for i in range(len(dates)):
        slice_2d = hcho_stack[i]
        igp_box = slice_2d[lat_mask][:, lon_mask]
        igp_means.append(np.nanmean(igp_box))
    
    igp_df = pd.DataFrame({
        'date': dates,
        'hcho_igp_mean': igp_means,
        'valid_pixel_frac': valid_frac
    })
    igp_df.to_csv(os.path.join(RESULTS_DIR, 'HCHO_igp_timeseries.csv'), index=False)
    
    print("Phase 2 complete. Running verification...")
    
    # Verification
    assert monthly_means.shape == (4, 120, 124)
    assert anomaly_overall.shape == (123, 120, 124)
    
    nov_scalar = np.nanmean(monthly_means[1])
    oct_scalar = np.nanmean(monthly_means[0])
    print(f"Oct mean: {oct_scalar:.4e}, Nov mean: {nov_scalar:.4e}")
    print(f"Nov > Oct: {nov_scalar > oct_scalar}")
    
    peak_date = igp_df.loc[igp_df['hcho_igp_mean'].idxmax(), 'date']
    print(f"IGP HCHO peak date: {peak_date}")


# ------------------------------------------------------------------------------
# Phase 3: Fire Data Processing
# ------------------------------------------------------------------------------
def run_phase3_fire_processing():
    warnings.filterwarnings('ignore')
    
    RESULTS_DIR = 'results'
    REGRID_DIR = 'data/regridded'
    FIRE_DIR = 'data/fire_data'
    
    print("Loading grid spec and dates...")
    with open(os.path.join(REGRID_DIR, 'grid_spec.json')) as f:
        grid_spec = json.load(f)
    
    with open(os.path.join(RESULTS_DIR, 'HCHO_dates.json'), 'r') as f:
        hcho_dates = json.load(f)
    
    lon_min = grid_spec['lon_min']
    lon_max = grid_spec['lon_max']
    lat_min = grid_spec['lat_min']
    lat_max = grid_spec['lat_max']
    n_lat = grid_spec['nlat']
    n_lon = grid_spec['nlon']
    
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon + 1)
    
    print("Loading fire CSVs...")
    modis_files = glob.glob(os.path.join(FIRE_DIR, 'fire_archive_M-C61_*.csv'))
    suomi_files = glob.glob(os.path.join(FIRE_DIR, 'fire_archive_SV-C2_*.csv'))
    j1_files = glob.glob(os.path.join(FIRE_DIR, 'fire_archive_J1V-C2_*.csv'))
    
    modis = pd.concat([pd.read_csv(f) for f in modis_files])
    suomi = pd.concat([pd.read_csv(f) for f in suomi_files])
    j1 = pd.concat([pd.read_csv(f) for f in j1_files])
    
    print("Standardising and filtering...")
    modis['acq_date'] = pd.to_datetime(modis['acq_date'], format='mixed', dayfirst=True)
    suomi['acq_date'] = pd.to_datetime(suomi['acq_date'], format='mixed', dayfirst=True)
    j1['acq_date'] = pd.to_datetime(j1['acq_date'], format='mixed', dayfirst=True)
    
    start_date = pd.to_datetime('2023-10-01')
    end_date = pd.to_datetime('2024-01-31')
    
    def filter_base(df, source):
        df = df[(df['acq_date'] >= start_date) & (df['acq_date'] <= end_date)]
        df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
                (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]
        if 'type' in df.columns:
            df = df[df['type'] == 0]
        df['source'] = source
        return df
    
    modis = filter_base(modis, 'MODIS')
    suomi = filter_base(suomi, 'VIIRS_SUOMI')
    j1 = filter_base(j1, 'VIIRS_J1')
    
    modis = modis[modis['confidence'] >= 60]
    
    def map_viirs_conf(c):
        c = str(c).lower()
        if c in ['l', 'low']: return 0
        if c in ['n', 'nominal']: return 1
        if c in ['h', 'high']: return 2
        return -1
    
    suomi['conf_int'] = suomi['confidence'].apply(map_viirs_conf)
    j1['conf_int'] = j1['confidence'].apply(map_viirs_conf)
    
    suomi = suomi[suomi['conf_int'] >= 1]
    j1 = j1[j1['conf_int'] >= 1]
    
    master_fire_df = pd.concat([modis, suomi, j1], ignore_index=True)
    master_fire_df['acq_date_str'] = master_fire_df['acq_date'].dt.strftime('%Y-%m-%d')
    
    print("Gridding to 0.25 deg...")
    fire_counts = np.zeros((123, n_lat, n_lon), dtype=np.float32)
    fire_frps = np.zeros((123, n_lat, n_lon), dtype=np.float32)
    
    igp_timeseries = []
    
    igp_lat_min, igp_lat_max = 28.0, 32.0
    igp_lon_min, igp_lon_max = 73.0, 80.0
    
    for i, d_str in enumerate(hcho_dates):
        day_fires = master_fire_df[master_fire_df['acq_date_str'] == d_str]
        
        counts, _, _ = np.histogram2d(
            day_fires['latitude'], day_fires['longitude'],
            bins=[lat_edges, lon_edges]
        )
        
        frp_sum, _, _ = np.histogram2d(
            day_fires['latitude'], day_fires['longitude'],
            bins=[lat_edges, lon_edges],
            weights=day_fires['frp']
        )
        
        mean_frp = np.zeros_like(counts)
        mask = counts > 0
        mean_frp[mask] = frp_sum[mask] / counts[mask]
        
        counts_flipped = np.flip(counts, axis=0)
        frp_flipped = np.flip(mean_frp, axis=0)
        
        fire_counts[i] = counts_flipped
        fire_frps[i] = frp_flipped
        
        igp_fires = day_fires[(day_fires['latitude'] >= igp_lat_min) & (day_fires['latitude'] <= igp_lat_max) &
                              (day_fires['longitude'] >= igp_lon_min) & (day_fires['longitude'] <= igp_lon_max)]
        
        igp_timeseries.append({
            'date': d_str,
            'fire_count_total': len(igp_fires),
            'fire_count_modis': len(igp_fires[igp_fires['source'] == 'MODIS']),
            'fire_count_viirs_suomi': len(igp_fires[igp_fires['source'] == 'VIIRS_SUOMI']),
            'fire_count_j1': len(igp_fires[igp_fires['source'] == 'VIIRS_J1']),
            'mean_frp': igp_fires['frp'].mean() if len(igp_fires) > 0 else 0.0
        })
    
    np.save(os.path.join(RESULTS_DIR, 'fire_counts_daily.npy'), fire_counts)
    np.save(os.path.join(RESULTS_DIR, 'fire_frp_daily.npy'), fire_frps)
    
    igp_df = pd.DataFrame(igp_timeseries)
    igp_df.to_csv(os.path.join(RESULTS_DIR, 'fire_igp_timeseries.csv'), index=False)
    
    stats = {
        'total_fires': len(master_fire_df),
        'modis_fires': len(modis),
        'suomi_fires': len(suomi),
        'j1_fires': len(j1),
        'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    }
    with open(os.path.join(RESULTS_DIR, 'fire_master_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("Phase 3 complete.")


# ------------------------------------------------------------------------------
# Phase 4: Hotspot Detection Algorithm
# ------------------------------------------------------------------------------
def run_phase4_hotspot_detection():
    RESULTS_DIR = 'results'
    REGRID_DIR = 'data/regridded'
    
    print("Loading HCHO stack...")
    HCHO_stack = np.load(os.path.join(RESULTS_DIR, 'HCHO_daily_stack.npy'))
    
    # A. Compute thresholds
    print("Computing thresholds...")
    all_valid = HCHO_stack[~np.isnan(HCHO_stack)].ravel()
    
    median_val  = np.median(all_valid)
    mad         = np.median(np.abs(all_valid - median_val))
    threshold_B = median_val + 3 * 1.4826 * mad
    
    threshold_C = 4.0e-4
    
    print(f"Threshold B (MAD-based): {threshold_B:.4e} mol/m²")
    print(f"Threshold C (literature): {threshold_C:.4e} mol/m²")
    
    # B. Create daily binary hotspot masks
    print("Creating daily binary hotspot masks...")
    hotspot_A = np.zeros_like(HCHO_stack, dtype=bool)
    thresh_A_daily = np.zeros(123)
    
    for i in range(123):
        day_slice  = HCHO_stack[i]
        valid_vals = day_slice[~np.isnan(day_slice)]
        if len(valid_vals) > 0:
            thresh_day      = np.percentile(valid_vals, 95)
            thresh_A_daily[i] = thresh_day
            hotspot_A[i]    = day_slice > thresh_day
    
    print(f"Threshold A (daily 95th pct mean) : {thresh_A_daily.mean():.4e} mol/m²")
    
    hotspot_B = HCHO_stack > threshold_B
    hotspot_C = HCHO_stack > threshold_C
    
    nan_mask  = np.isnan(HCHO_stack)
    hotspot_A[nan_mask] = False
    hotspot_B[nan_mask] = False
    hotspot_C[nan_mask] = False
    
    # C. Hotspot frequency maps
    print("Computing hotspot frequency maps...")
    freq_A = hotspot_A.sum(axis=0)
    freq_B = hotspot_B.sum(axis=0)
    freq_C = hotspot_C.sum(axis=0)
    
    # D. Consensus hotspot map
    print("Computing consensus hotspot map...")
    ever_A = (freq_A > 0).astype(int)
    ever_B = (freq_B > 0).astype(int)
    ever_C = (freq_C > 0).astype(int)
    
    consensus = (ever_A + ever_B + ever_C) >= 2
    
    # E. Save thresholds
    print("Saving results...")
    thresholds = {
        'method_A_percentile95' : float(thresh_A_daily.mean()),
        'method_B_MAD'          : float(threshold_B),
        'method_C_literature'   : float(threshold_C),
        'median_hcho'           : float(median_val),
        'mad_value'             : float(mad),
    }
    
    with open(os.path.join(RESULTS_DIR, 'hotspot_thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=4)
    
    np.save(os.path.join(RESULTS_DIR, 'hotspot_frequency_A.npy'), freq_A)
    np.save(os.path.join(RESULTS_DIR, 'hotspot_frequency_B.npy'), freq_B)
    np.save(os.path.join(RESULTS_DIR, 'hotspot_frequency_C.npy'), freq_C)
    np.save(os.path.join(RESULTS_DIR, 'hotspot_consensus.npy'), consensus)
    
    print("Phase 4 complete.")


# ------------------------------------------------------------------------------
# Phase 5: Spatial Visualisation
# ------------------------------------------------------------------------------
def run_phase5_hcho_maps():
    matplotlib.use('Agg')
    
    
    
    
    
    warnings.filterwarnings('ignore')
    
    print("Setting up for mapping...")
    with open('data/regridded/grid_spec.json') as f:
        gs = json.load(f)
    
    lats = np.linspace(gs['lat_max'], gs['lat_min'], gs['nlat'])
    lons = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'])
    LON, LAT = np.meshgrid(lons, lats)
    
    
    
    
    with open(r'data\gadm41_IND_1.json\gadm41_IND_1.json', 'r', encoding='utf-8') as f:
        gadm_data = json.load(f)
        
    state_geoms = [shape(feature['geometry']) for feature in gadm_data['features']]
    india_geom = unary_union(state_geoms)
    
    dates    = json.load(open('results/HCHO_dates.json'))
    dates_pd = pd.to_datetime(dates)
    
    HCHO_stack     = np.load('results/HCHO_daily_stack.npy')
    monthly_means  = np.load('results/HCHO_monthly_means.npy')
    monthly_anom   = np.load('results/HCHO_anomalies_monthly.npy')
    freq_B         = np.load('results/hotspot_frequency_B.npy')
    freq_C         = np.load('results/hotspot_frequency_C.npy')
    consensus      = np.load('results/hotspot_consensus.npy')
    fire_counts    = np.load('results/fire_counts_daily.npy')
    
    igp_hcho = pd.read_csv('results/HCHO_igp_timeseries.csv', parse_dates=['date'])
    igp_fire = pd.read_csv('results/fire_igp_timeseries.csv',  parse_dates=['date'])
    
    os.makedirs('obj2_maps/HCHO_daily', exist_ok=True)
    os.makedirs('obj2_maps', exist_ok=True)
    
    valid_all  = HCHO_stack[~np.isnan(HCHO_stack)]
    VMIN_HCHO  = np.percentile(valid_all, 2)
    VMAX_HCHO  = np.percentile(valid_all, 98)
    
    def plot_geom(ax, geom, color, linewidth, alpha=1.0, zorder=5):
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    def add_boundary(ax):
        for geom in state_geoms:
            plot_geom(ax, geom, color='black', linewidth=0.5, alpha=0.7, zorder=5)
        plot_geom(ax, india_geom, color='black', linewidth=1.3, alpha=1.0, zorder=6)
        ax.set_xlim(gs['lon_min'], gs['lon_max'])
        ax.set_ylim(gs['lat_min'], gs['lat_max'])
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
    
    print("Map 0 - Daily HCHO maps...")
    peak_start = pd.Timestamp('2023-10-15')
    peak_end   = pd.Timestamp('2023-11-30')
    
    for i, d in enumerate(dates_pd):
        if not (peak_start <= d <= peak_end):
            continue
        grid = HCHO_stack[i]
        fig, ax = plt.subplots(figsize=(8, 9))
        im = ax.pcolormesh(LON, LAT, grid, cmap='YlOrRd', shading='auto', vmin=VMIN_HCHO, vmax=VMAX_HCHO)
        add_boundary(ax)
        plt.colorbar(im, ax=ax, label='HCHO Column (mol/m²)', shrink=0.7)
        ax.set_title(f'HCHO Column Density — {d.strftime("%d %b %Y")}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'obj2_maps/HCHO_daily/HCHO_{d.strftime("%Y%m%d")}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print("Map 1 - Seasonal mean...")
    mean_overall = np.nanmean(HCHO_stack, axis=0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(LON, LAT, mean_overall, cmap='YlOrRd', shading='auto', vmin=VMIN_HCHO, vmax=VMAX_HCHO)
    add_boundary(ax)
    plt.colorbar(im, ax=ax, label='HCHO Column (mol/m²)', shrink=0.7)
    ax.set_title('Mean HCHO Column Density\nOct 2023 – Jan 2024', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_seasonal_mean.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Map 2 - Monthly means...")
    month_labels = ['October 2023', 'November 2023', 'December 2023', 'January 2024']
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    for idx, (ax, label) in enumerate(zip(axes, month_labels)):
        im = ax.pcolormesh(LON, LAT, monthly_means[idx], cmap='YlOrRd', shading='auto', vmin=VMIN_HCHO, vmax=VMAX_HCHO)
        add_boundary(ax)
        ax.set_title(label, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='mol/m²', shrink=0.75)
    fig.suptitle('Monthly Mean HCHO Column Density', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_monthly_4panel.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Map 3 - Nov fire overlay...")
    nov_idx = [i for i, d in enumerate(dates_pd) if d.month == 11]
    nov_hcho_mean = np.nanmean(HCHO_stack[nov_idx], axis=0)
    nov_fire_mean = np.nanmean(fire_counts[nov_idx], axis=0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(LON, LAT, nov_hcho_mean, cmap='YlOrRd', shading='auto', vmin=VMIN_HCHO, vmax=VMAX_HCHO)
    add_boundary(ax)
    
    fire_lats, fire_lons = [], []
    fire_sizes = []
    for r in range(gs['nlat']):
        for c in range(gs['nlon']):
            val = nov_fire_mean[r, c]
            if val > 0.5:
                fire_lats.append(lats[r])
                fire_lons.append(lons[c])
                fire_sizes.append(val)
    
    fire_sizes = np.array(fire_sizes)
    ax.scatter(fire_lons, fire_lats, s=fire_sizes * 2, c='black', alpha=0.6, zorder=7, label='Fire count (dot size ∝ count)')
    plt.colorbar(im, ax=ax, label='Nov Mean HCHO (mol/m²)', shrink=0.7)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title('November 2023 — HCHO Column vs Fire Activity\nBackground: HCHO | Dots: Fire Count', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_nov_fire_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Map 4 - Hotspot frequency...")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(LON, LAT, freq_C, cmap='hot_r', shading='auto', vmin=0, vmax=freq_C.max())
    add_boundary(ax)
    plt.colorbar(im, ax=ax, label='Number of Hotspot Days', shrink=0.7)
    ax.set_title('HCHO Hotspot Frequency\nDays exceeding 4.0×10⁻⁴ mol/m² (Oct 2023 – Jan 2024)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_hotspot_frequency.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Map 5 - Three method comparison...")
    freq_A = np.load('results/hotspot_frequency_A.npy')
    titles = ['Method A: Daily 95th Percentile', 'Method B: Robust MAD (3.47×10⁻⁴)', 'Method C: Literature (4.0×10⁻⁴)']
    freqs  = [freq_A, freq_B, freq_C]
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, freq, title in zip(axes, freqs, titles):
        im = ax.pcolormesh(LON, LAT, freq, cmap='hot_r', shading='auto', vmin=0, vmax=freq_C.max())
        add_boundary(ax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Hotspot days', shrink=0.7)
    fig.suptitle('Hotspot Detection — Three Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_hotspot_3method.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Map 6 - November anomaly...")
    mean_overall = np.nanmean(HCHO_stack, axis=0)
    nov_anom_mean = monthly_means[1] - mean_overall
    vmax_anom = np.nanpercentile(np.abs(nov_anom_mean), 98)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(LON, LAT, nov_anom_mean, cmap='RdBu_r', shading='auto', vmin=-vmax_anom, vmax=vmax_anom)
    add_boundary(ax)
    plt.colorbar(im, ax=ax, label='HCHO Anomaly from Nov Mean (mol/m²)', shrink=0.7)
    ax.set_title('November 2023 — HCHO Monthly Anomaly\nRed = above monthly mean | Blue = below monthly mean', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_nov_anomaly.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Map 7 - IGP time series...")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    merged = igp_hcho.merge(igp_fire, on='date', how='inner')
    merged = merged.sort_values('date')
    ax1.fill_between(merged['date'], merged['hcho_igp_mean'], alpha=0.35, color='#e74c3c')
    ax1.plot(merged['date'], merged['hcho_igp_mean'], color='#c0392b', linewidth=1.2, label='IGP Mean HCHO')
    ax1.set_ylabel('Mean HCHO Column (mol/m²)', color='#c0392b', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#c0392b')
    ax2 = ax1.twinx()
    ax2.bar(merged['date'], merged['fire_count_total'], color='#2c3e50', alpha=0.5, width=1, label='IGP Fire Count')
    ax2.set_ylabel('Total Fire Count in IGP', color='#2c3e50', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.axvspan(pd.Timestamp('2023-10-15'), pd.Timestamp('2023-11-15'), alpha=0.12, color='orange', label='Peak burning window (Oct 15 – Nov 15)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.set_title('IGP Region — Daily HCHO vs Fire Count\nOct 2023 – Jan 2024', fontsize=13, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_igp_timeseries.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Phase 5 complete.")


# ------------------------------------------------------------------------------
# Phase 6: Temporal Lag Correlation
# ------------------------------------------------------------------------------
def run_phase6_correlation():
    matplotlib.use('Agg')
    
    
    
    
    warnings.filterwarnings('ignore')
    
    RESULTS_DIR = 'results'
    REGRID_DIR = 'data/regridded'
    os.makedirs('obj2_maps', exist_ok=True)
    
    print("Loading data...")
    HCHO_stack = np.load(os.path.join(RESULTS_DIR, 'HCHO_daily_stack.npy'))
    fire_counts = np.load(os.path.join(RESULTS_DIR, 'fire_counts_daily.npy'))
    vpf = np.load(os.path.join(RESULTS_DIR, 'HCHO_valid_pixel_fraction.npy'))
    
    with open(os.path.join(REGRID_DIR, 'grid_spec.json')) as f:
        gs = json.load(f)
    
    lats = np.linspace(gs['lat_max'], gs['lat_min'], gs['nlat'])
    lons = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'])
    LON, LAT = np.meshgrid(lons, lats)
    
    igp_lat_mask = (lats >= 28.0) & (lats <= 32.0)
    igp_lon_mask = (lons >= 73.0) & (lons <= 80.0)
    
    # Setup plotting
    with open(r'data\gadm41_IND_1.json\gadm41_IND_1.json', 'r', encoding='utf-8') as f:
        gadm_data = json.load(f)
    state_geoms = [shape(feature['geometry']) for feature in gadm_data['features']]
    india_geom = unary_union(state_geoms)
    
    def plot_geom(ax, geom, color, linewidth, alpha=1.0, zorder=5):
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    def add_boundary(ax):
        for geom in state_geoms:
            plot_geom(ax, geom, color='black', linewidth=0.5, alpha=0.7, zorder=5)
        plot_geom(ax, india_geom, color='black', linewidth=1.3, alpha=1.0, zorder=6)
        ax.set_xlim(gs['lon_min'], gs['lon_max'])
        ax.set_ylim(gs['lat_min'], gs['lat_max'])
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
    
    # A. IGP time series correlation
    print("Computing IGP time series correlation...")
    hcho_igp = np.array([
        np.nanmean(HCHO_stack[i][np.ix_(igp_lat_mask, igp_lon_mask)])
        for i in range(123)
    ])
    fire_igp = np.array([
        np.nansum(fire_counts[i][np.ix_(igp_lat_mask, igp_lon_mask)])
        for i in range(123)
    ])
    
    quality_mask = vpf > 0.3
    
    with open(os.path.join(RESULTS_DIR, 'HCHO_dates.json'), 'r') as f:
        dates_pd = pd.to_datetime(json.load(f))
    nov_mask = (dates_pd.month == 11) & quality_mask
    
    lag_results = {}
    for lag in range(4):
        if lag == 0:
            fire_lagged = fire_igp[nov_mask]
            hcho_lagged = hcho_igp[nov_mask]
        else:
            nov_indices = np.where(nov_mask)[0]
            valid_pairs = nov_indices[nov_indices + lag < 123]
            fire_lagged = fire_igp[valid_pairs]
            hcho_lagged = hcho_igp[valid_pairs + lag]
    
        if len(fire_lagged) > 5:
            r, p = pearsonr(fire_lagged, hcho_lagged)
            lag_results[f'lag_{lag}'] = {
                'r': float(r),
                'p_value': float(p),
                'significant': bool(p < 0.05),
                'n_days': int(len(fire_lagged))
            }
            print(f"Lag {lag}: r={r:.3f}, p={p:.4f}, {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    
    best_lag = max(lag_results, key=lambda k: abs(lag_results[k]['r']))
    print(f"\nPeak correlation at: {best_lag} (r={lag_results[best_lag]['r']:.3f})")
    
    # B. Spatial correlation map
    print("Computing spatial correlation map...")
    nov_idx = np.where(nov_mask)[0]
    corr_map  = np.full((gs['nlat'], gs['nlon']), np.nan)
    pval_map  = np.full((gs['nlat'], gs['nlon']), np.nan)
    
    for r in range(gs['nlat']):
        for c in range(gs['nlon']):
            hcho_pixel = HCHO_stack[nov_idx, r, c]
            fire_pixel = fire_counts[nov_idx, r, c]
            valid      = ~np.isnan(hcho_pixel)
            if valid.sum() >= 10:
                rval, pval = pearsonr(hcho_pixel[valid], fire_pixel[valid])
                corr_map[r, c] = rval
                pval_map[r, c] = pval
    
    print(f"Spatial corr map: valid cells = {(~np.isnan(corr_map)).sum()}")
    print(f"Significant cells (p<0.05): {(pval_map < 0.05).sum()}")
    print(f"Mean r in IGP: {np.nanmean(corr_map[np.ix_(igp_lat_mask, igp_lon_mask)]):.3f}")
    
    # C. Plot correlation map
    print("Plotting spatial correlation...")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(LON, LAT, corr_map, cmap='RdBu_r', shading='auto', vmin=-1, vmax=1)
    add_boundary(ax)
    
    sig_r, sig_c = np.where(pval_map < 0.05)
    ax.scatter(lons[sig_c], lats[sig_r], s=1.5, c='black', alpha=0.4, zorder=6)
    
    plt.colorbar(im, ax=ax, label='Pearson r (Fire vs HCHO)', shrink=0.7)
    ax.set_title('Spatial Correlation: Fire Count vs HCHO\nNovember 2023 | Stippled = p < 0.05', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_fire_correlation_november.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # D. Plot lag analysis
    print("Plotting lag analysis...")
    lags   = [int(k.split('_')[1]) for k in lag_results]
    r_vals = [lag_results[f'lag_{l}']['r'] for l in lags]
    p_vals = [lag_results[f'lag_{l}']['p_value'] for l in lags]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(lags, r_vals, color=['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_vals], edgecolor='black', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(lags)
    ax.set_xticklabels([f'Lag {l}\n(fire → HCHO+{l}d)' for l in lags])
    ax.set_ylabel('Pearson r', fontsize=11)
    ax.set_title('IGP Fire-HCHO Lagged Correlation\nNovember 2023 | Green = p<0.05', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.2, 1.0)
    for bar, r in zip(bars, r_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{r:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_fire_lag_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'correlation_fire_hcho.json'), 'w') as f:
        json.dump(lag_results, f, indent=4)
    np.save(os.path.join(RESULTS_DIR, 'correlation_map_november.npy'), corr_map)
    np.save(os.path.join(RESULTS_DIR, 'pvalue_map_november.npy'), pval_map)
    
    print("Phase 6 complete.")


# ------------------------------------------------------------------------------
# Phase 7: Transport and Wind Analysis
# ------------------------------------------------------------------------------
def run_phase7_transport():
    matplotlib.use('Agg')
    
    
    
    
    
    os.makedirs('obj2_maps', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    with open('data/regridded/grid_spec.json') as f:
        gs = json.load(f)
    
    lats = np.linspace(gs['lat_max'], gs['lat_min'], gs['nlat'])
    lons = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'])
    LON, LAT = np.meshgrid(lons, lats)
    dlat = abs(lats[1] - lats[0])   # 0.25°
    dlon = abs(lons[1] - lons[0])   # 0.25°
    
    # Delhi receptor
    DELHI_LAT = 28.61
    DELHI_LON = 77.23
    
    # Peak burning window
    dates_pd = pd.to_datetime(json.load(open('results/HCHO_dates.json')))
    peak_mask = (dates_pd >= '2023-10-15') & (dates_pd <= '2023-11-15')
    peak_indices = np.where(peak_mask)[0]
    print(f"Peak burning days: {len(peak_indices)}")
    
    def load_era5_day(date_str, variable):
        """Load one day's ERA5 variable"""
        path = f'data/regridded/ERA5/{variable}_{date_str}.npy'
        return np.load(path)   # shape (120, 124)
    
    def get_wind_at_point(u_grid, v_grid, lat, lon):
        """Bilinear interpolation of wind at a lat/lon point."""
        r = np.argmin(np.abs(lats - lat))
        c = np.argmin(np.abs(lons - lon))
        r = np.clip(r, 0, gs['nlat'] - 1)
        c = np.clip(c, 0, gs['nlon'] - 1)
        return float(u_grid[r, c]), float(v_grid[r, c])
    
    # Run 48-hour backward trajectory for each peak burning day
    DT_HOURS  = 6
    N_STEPS   = 8
    DT_SEC    = DT_HOURS * 3600
    
    all_trajectories  = []
    trajectory_lats   = []
    trajectory_lons   = []
    
    print("Computing trajectories...")
    for idx in peak_indices:
        date_str = dates_pd[idx].strftime('%Y%m%d')
        try:
            u_grid = load_era5_day(date_str, 'U10')
            v_grid = load_era5_day(date_str, 'V10')
        except FileNotFoundError:
            print(f"  Missing ERA5 for {date_str}, skipping")
            continue
    
        cur_lat = DELHI_LAT
        cur_lon = DELHI_LON
        path    = [(cur_lat, cur_lon)]
    
        for step in range(N_STEPS):
            u, v = get_wind_at_point(u_grid, v_grid, cur_lat, cur_lon)
            d_lat = -(v * DT_SEC) / 111000.0
            d_lon = -(u * DT_SEC) / (111000.0 * np.cos(np.radians(cur_lat)))
            cur_lat += d_lat
            cur_lon += d_lon
            cur_lat = np.clip(cur_lat, gs['lat_min'], gs['lat_max'])
            cur_lon = np.clip(cur_lon, gs['lon_min'], gs['lon_max'])
            path.append((cur_lat, cur_lon))
            trajectory_lats.append(cur_lat)
            trajectory_lons.append(cur_lon)
    
        all_trajectories.append({
            'date'  : date_str,
            'path'  : path,
            'origin': path[-1]
        })
    
    print(f"Computed {len(all_trajectories)} trajectories")
    
    with open('results/delhi_trajectories.json', 'w') as f:
        json.dump(all_trajectories, f, indent=2)
    
    print("Computing trajectory frequency...")
    traj_freq = np.zeros((gs['nlat'], gs['nlon']), dtype=float)
    
    lat_edges = np.linspace(gs['lat_min'], gs['lat_max'], gs['nlat'] + 1)
    lon_edges = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'] + 1)
    
    counts, _, _ = np.histogram2d(
        trajectory_lats, trajectory_lons,
        bins=[lat_edges, lon_edges]
    )
    traj_freq = np.flipud(counts)
    traj_freq_pct = traj_freq / len(all_trajectories) * 100
    
    np.save('results/trajectory_frequency_map.npy', traj_freq_pct)
    print(f"Trajectory freq map saved. Max: {traj_freq_pct.max():.1f}%")
    print(f"Peak source area: lat={lats[traj_freq_pct.argmax()//gs['nlon']]:.2f}°N, "
          f"lon={lons[traj_freq_pct.argmax()%gs['nlon']]:.2f}°E")
    
    # BOUNDARY SETUP
    with open(r'data\gadm41_IND_1.json\gadm41_IND_1.json', 'r', encoding='utf-8') as f:
        gadm_data = json.load(f)
    state_geoms = [shape(feature['geometry']) for feature in gadm_data['features']]
    india_geom = unary_union(state_geoms)
    
    def plot_geom(ax, geom, color, linewidth, alpha=1.0, zorder=5):
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    def add_boundary(ax):
        for geom in state_geoms:
            plot_geom(ax, geom, color='black', linewidth=0.5, alpha=0.7, zorder=5)
        plot_geom(ax, india_geom, color='black', linewidth=1.3, alpha=1.0, zorder=6)
        ax.set_xlim(gs['lon_min'], gs['lon_max'])
        ax.set_ylim(gs['lat_min'], gs['lat_max'])
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
    
    print("Plotting trajectory frequency...")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(LON, LAT, traj_freq_pct, cmap='YlOrRd',
                       shading='auto', vmin=0, vmax=traj_freq_pct.max())
    add_boundary(ax)
    ax.scatter([DELHI_LON], [DELHI_LAT], s=120, c='blue',
               marker='*', zorder=10, label='Delhi (receptor)')
    plt.colorbar(im, ax=ax, label='% of Backward Trajectories Passing Through', shrink=0.7)
    ax.legend(fontsize=10)
    ax.set_title('48-Hour Backward Trajectory Frequency\nFrom Delhi | Peak Burning Period (Oct 15 – Nov 15 2023)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj2_maps/trajectory_frequency.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Computing wind rose...")
    wind_dirs  = []
    wind_speeds = []
    
    for idx in peak_indices:
        date_str = dates_pd[idx].strftime('%Y%m%d')
        try:
            u_grid = load_era5_day(date_str, 'U10')
            v_grid = load_era5_day(date_str, 'V10')
            u, v   = get_wind_at_point(u_grid, v_grid, DELHI_LAT, DELHI_LON)
            speed  = np.sqrt(u**2 + v**2)
            direc  = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
            wind_dirs.append(direc)
            wind_speeds.append(speed)
        except FileNotFoundError:
            continue
    
    wind_dirs   = np.array(wind_dirs)
    wind_speeds = np.array(wind_speeds)
    
    # Save wind rose data for verification
    np.save('results/delhi_wind_speeds.npy', wind_speeds)
    np.save('results/delhi_wind_dirs.npy', wind_dirs)
    
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='polar')
    
    n_bins  = 16
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    theta   = np.radians(bin_centers)
    counts  = np.histogram(wind_dirs, bins=bin_edges)[0]
    
    bars = ax.bar(theta, counts, width=2*np.pi/n_bins,
                  bottom=0, alpha=0.7, color='#3498db',
                  edgecolor='black', linewidth=0.5)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_title(f'Wind Rose — Delhi (Standard Meteorological)\nPeak Burning Period (Oct 15 – Nov 15)\nMean speed: {wind_speeds.mean():.1f} m/s',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('obj2_maps/wind_rose_delhi.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Plotting wind field maps...")
    rep_dates = ['20231105', '20231115']
    rep_labels = ['05 Nov 2023 (Peak Burning)', '15 Nov 2023 (Late Burning)']
    
    HCHO_stack = np.load('results/HCHO_daily_stack.npy')
    valid_all  = HCHO_stack[~np.isnan(HCHO_stack)]
    VMIN_HCHO  = np.percentile(valid_all, 2)
    VMAX_HCHO  = np.percentile(valid_all, 98)
    
    for date_str, label in zip(rep_dates, rep_labels):
        idx = np.where(dates_pd == pd.Timestamp(date_str))[0][0]
        hcho_day = HCHO_stack[idx]
        
        try:
            u_grid = load_era5_day(date_str, 'U10')
            v_grid = load_era5_day(date_str, 'V10')
        except FileNotFoundError:
            print(f"Skipping {date_str} due to missing ERA5 data")
            continue
    
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.pcolormesh(LON, LAT, hcho_day, cmap='YlOrRd',
                           shading='auto', vmin=VMIN_HCHO, vmax=VMAX_HCHO)
        add_boundary(ax)
    
        step = 4
        ax.quiver(
            LON[::step, ::step], LAT[::step, ::step],
            u_grid[::step, ::step], v_grid[::step, ::step],
            scale=50, scale_units='inches',
            width=0.003, color='white', alpha=0.8, zorder=7
        )
    
        ax.scatter([DELHI_LON], [DELHI_LAT], s=100, c='blue',
                   marker='*', zorder=10, label='Delhi')
    
        plt.colorbar(im, ax=ax, label='HCHO Column (mol/m²)', shrink=0.7)
        ax.legend(fontsize=10)
        ax.set_title(f'HCHO Column + ERA5 Wind Field\n{label}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'obj2_maps/wind_field_HCHO_{date_str}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved wind field map: {date_str}")
    
    print("Phase 7 complete.")


# ------------------------------------------------------------------------------
# Phase 8: Source Region Identification
# ------------------------------------------------------------------------------
def run_phase8_source_regions():
    matplotlib.use('Agg')
    
    
    
    
    
    
    os.makedirs('obj2_maps', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    with open('data/regridded/grid_spec.json') as f:
        gs = json.load(f)
    
    lats = np.linspace(gs['lat_max'], gs['lat_min'], gs['nlat'])
    lons = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'])
    LON, LAT = np.meshgrid(lons, lats)
    
    REGIONS = {
        'SW_Punjab'  : {'lat': (29.5, 31.5), 'lon': (73.5, 76.0),
                        'label': 'SW Punjab (Bathinda-Muktsar)'},
        'Central_Valley': {'lat': (30.5, 32.0), 'lon': (74.5, 76.5),
                           'label': 'Central Punjab Valley (Ludhiana-Patiala)'},
        'S_Haryana'  : {'lat': (28.5, 30.5), 'lon': (75.5, 77.5),
                        'label': 'S Haryana (Hisar-Karnal)'},
        'W_UP'       : {'lat': (27.5, 30.0), 'lon': (77.0, 80.0),
                        'label': 'W Uttar Pradesh (Saharanpur-Meerut)'},
    }
    
    # Load Data
    dates_pd = pd.to_datetime(json.load(open('results/HCHO_dates.json')))
    peak_mask = (dates_pd >= '2023-10-15') & (dates_pd <= '2023-11-15')
    peak_indices = np.where(peak_mask)[0]
    
    HCHO_stack = np.load('results/HCHO_daily_stack.npy')[peak_indices]
    fire_counts = np.load('results/fire_counts_daily.npy')[peak_indices]
    fire_frp = np.load('results/fire_frp_daily.npy')[peak_indices]
    consensus_hotspot = np.load('results/hotspot_consensus.npy')
    
    HCHO_monthly_means = np.load('results/HCHO_monthly_means.npy')
    nov_hcho = HCHO_monthly_means[1]  # Oct=0, Nov=1, Dec=2, Jan=3
    
    THRESHOLD_C = 4.0e-4
    
    results = []
    
    for r_id, r_info in REGIONS.items():
        lat_min, lat_max = r_info['lat']
        lon_min, lon_max = r_info['lon']
        
        # Create mask
        lat_mask = (LAT >= lat_min) & (LAT <= lat_max)
        lon_mask = (LON >= lon_min) & (LON <= lon_max)
        region_mask = lat_mask & lon_mask
        
        # Subset data spatially
        hcho_region = HCHO_stack[:, region_mask]
        fire_region = fire_counts[:, region_mask]
        frp_region = fire_frp[:, region_mask]
        consensus_region = consensus_hotspot[region_mask]
        
        # Calculate daily spatial means/sums
        daily_mean_hcho = np.nanmean(hcho_region, axis=1)
        daily_fire_sum = np.nansum(fire_region, axis=1)
        
        # Metrics
        mean_hcho = np.nanmean(daily_mean_hcho)
        peak_hcho_day_idx = np.nanargmax(daily_mean_hcho)
        peak_hcho_date = dates_pd[peak_indices[peak_hcho_day_idx]].strftime('%Y-%m-%d')
        total_fire_count = np.nansum(daily_fire_sum)
        
        # Mean FRP (only where fire exists)
        frp_valid = frp_region[fire_region > 0]
        mean_frp = np.nanmean(frp_valid) if len(frp_valid) > 0 else 0
        
        # Hotspot days (days where at least one pixel in region exceeds Threshold C)
        hotspot_days = int(np.sum(np.nanmax(hcho_region, axis=1) > THRESHOLD_C))
        
        # Consensus %
        consensus_pct = np.mean(consensus_region) * 100
        
        # Pearson r
        valid_days = ~np.isnan(daily_mean_hcho) & ~np.isnan(daily_fire_sum)
        if np.sum(valid_days) > 2:
            r, p = pearsonr(daily_fire_sum[valid_days], daily_mean_hcho[valid_days])
        else:
            r, p = np.nan, np.nan
            
        results.append({
            'Region': r_info['label'],
            'Hotspot_Days': hotspot_days,
            'Total_Fire_Count': total_fire_count,
            'Mean_HCHO_mol_m2': mean_hcho,
            'Pearson_r': r,
            'Pearson_p': p,
            'Peak_HCHO_Date': peak_hcho_date,
            'Mean_FRP_MW': mean_frp,
            'Consensus_Hotspot_Pct': consensus_pct
        })
    
    df = pd.DataFrame(results)
    df.to_csv('results/source_regions_summary.csv', index=False)
    print("Saved summary table to results/source_regions_summary.csv")
    print(df[['Region', 'Hotspot_Days', 'Total_Fire_Count', 'Mean_HCHO_mol_m2', 'Pearson_r']])
    
    # Plotting Map
    # Load boundary
    with open(r'data\gadm41_IND_1.json\gadm41_IND_1.json', 'r', encoding='utf-8') as f:
        gadm_data = json.load(f)
    state_geoms = [shape(feature['geometry']) for feature in gadm_data['features']]
    india_geom = unary_union(state_geoms)
    
    def plot_geom(ax, geom, color, linewidth, alpha=1.0, zorder=5):
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    def add_boundary(ax):
        for geom in state_geoms:
            plot_geom(ax, geom, color='black', linewidth=0.5, alpha=0.7, zorder=5)
        plot_geom(ax, india_geom, color='black', linewidth=1.3, alpha=1.0, zorder=6)
        ax.set_xlim(gs['lon_min'], gs['lon_max'])
        ax.set_ylim(gs['lat_min'], gs['lat_max'])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    valid_all = nov_hcho[~np.isnan(nov_hcho)]
    vmin = np.percentile(valid_all, 2)
    vmax = np.percentile(valid_all, 98)
    
    # Zoom in on IGP region for better visibility of source regions
    im = ax.pcolormesh(LON, LAT, nov_hcho, cmap='YlOrRd', shading='auto', vmin=vmin, vmax=vmax)
    add_boundary(ax)
    
    # Draw regions
    colors = ['blue', 'purple', 'green', 'cyan']
    for idx, (r_id, r_info) in enumerate(REGIONS.items()):
        lat_min, lat_max = r_info['lat']
        lon_min, lon_max = r_info['lon']
        
        rect = patches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                 linewidth=2, edgecolor=colors[idx], facecolor='none', zorder=10)
        ax.add_patch(rect)
        
        # Annotation
        row = df[df['Region'] == r_info['label']].iloc[0]
        fires = int(row['Total_Fire_Count'])
        h_days = int(row['Hotspot_Days'])
        text = f"{r_info['label']}\nFires: {fires}\nHotspot Days: {h_days}"
        ax.text(lon_max, lat_max, text, color=colors[idx], fontsize=10, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                ha='left', va='top', zorder=11)
    
    ax.set_xlim(68, 88)
    ax.set_ylim(22, 35)
    ax.set_title("Source Regions overlay on Nov Mean HCHO\nPeak Burning Period (Oct 15 - Nov 15)", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='HCHO Column (mol/m²)', shrink=0.7)
    plt.tight_layout()
    plt.savefig('obj2_maps/source_regions_map.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved source regions map to obj2_maps/source_regions_map.png")


# ------------------------------------------------------------------------------
# Phase 8b: DBSCAN Algorithmic Cluster Validation
# ------------------------------------------------------------------------------
def run_phase8b_dbscan_clustering():
    matplotlib.use('Agg')
    
    
    
    
    
    # Ensure directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('obj2_maps', exist_ok=True)
    os.makedirs('maps', exist_ok=True)
    
    with open('data/regridded/grid_spec.json') as f:
        gs = json.load(f)
    
    # NOTE: Use 'nlat' and 'nlon' which match our grid_spec.json
    lats = np.linspace(gs['lat_max'], gs['lat_min'], gs['nlat'])
    lons = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'])
    
    consensus = np.load('results/hotspot_consensus.npy')   # boolean (120, 124)
    
    # Extract lat/lon coordinates of every hotspot pixel
    rows, cols = np.where(consensus)
    hotspot_coords = np.column_stack([lats[rows], lons[cols]])   # shape (N, 2)
    
    print(f"Total hotspot pixels: {len(hotspot_coords)}")
    
    def simple_dbscan(X, eps, min_samples):
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        for i in range(n):
            if visited[i]: continue
            visited[i] = True
            neighbors = np.where(dist[i] <= eps)[0]
            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                seed_set = list(neighbors)
                if i in seed_set: seed_set.remove(i)
                while len(seed_set) > 0:
                    q = seed_set.pop(0)
                    if not visited[q]:
                        visited[q] = True
                        q_neighbors = np.where(dist[q] <= eps)[0]
                        if len(q_neighbors) >= min_samples:
                            for qn in q_neighbors:
                                if qn not in seed_set and not visited[qn]:
                                    seed_set.append(qn)
                    if labels[q] == -1:
                        labels[q] = cluster_id
                cluster_id += 1
        return labels
    
    labels = simple_dbscan(hotspot_coords, eps=0.5, min_samples=5)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points (unclustered): {n_noise}")
    
    # Build cluster summary table
    cluster_summary = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue   # skip noise
        mask = labels == cluster_id
        cluster_lats = hotspot_coords[mask, 0]
        cluster_lons = hotspot_coords[mask, 1]
        cluster_summary.append({
            'cluster_id'  : int(cluster_id),
            'n_pixels'    : int(mask.sum()),
            'center_lat'  : float(cluster_lats.mean()),
            'center_lon'  : float(cluster_lons.mean()),
            'lat_range'   : f"{cluster_lats.min():.2f}-{cluster_lats.max():.2f}",
            'lon_range'   : f"{cluster_lons.min():.2f}-{cluster_lons.max():.2f}",
        })
    
    df_clusters = pd.DataFrame(cluster_summary).sort_values('n_pixels', ascending=False)
    print("\nCluster Summary:")
    print(df_clusters.to_string(index=False))
    
    df_clusters.to_csv('results/dbscan_clusters.csv', index=False)
    
    # BOUNDARY SETUP
    with open(r'data\gadm41_IND_1.json\gadm41_IND_1.json', 'r', encoding='utf-8') as f:
        gadm_data = json.load(f)
        
    state_geoms = [shape(feature['geometry']) for feature in gadm_data['features']]
    india_geom = unary_union(state_geoms)
    
    def plot_geom(ax, geom, color, linewidth, alpha=1.0, zorder=5):
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    def add_boundary(ax):
        for geom in state_geoms:
            plot_geom(ax, geom, color='black', linewidth=0.5, alpha=0.7, zorder=5)
        plot_geom(ax, india_geom, color='black', linewidth=1.3, alpha=1.0, zorder=6)
        ax.set_xlim(gs['lon_min'], gs['lon_max'])
        ax.set_ylim(gs['lat_min'], gs['lat_max'])
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
    
    
    # Plot the clusters on the map
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot noise points in grey
    noise_mask = labels == -1
    ax.scatter(hotspot_coords[noise_mask, 1], hotspot_coords[noise_mask, 0],
               c='lightgrey', s=8, alpha=0.5, label='Unclustered noise')
    
    # Plot each cluster in a different color
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for cluster_id, color in zip(sorted(set(labels) - {-1}), colors):
        mask = labels == cluster_id
        ax.scatter(hotspot_coords[mask, 1], hotspot_coords[mask, 0],
                   c=[color], s=15, label=f'Cluster {cluster_id} (n={mask.sum()})')
    
    add_boundary(ax)
    ax.set_title('DBSCAN Clustering of Consensus HCHO Hotspots\n'
                 f'{n_clusters} clusters identified | eps=0.5°, min_samples=5',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig('obj2_maps/HCHO_dbscan_clusters.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Verification check
    print(f"\nTop 3 largest clusters:")
    print(df_clusters.head(3).to_string(index=False))
    
    top_cluster = df_clusters.iloc[0]
    print(f"\nLargest cluster center: {top_cluster['center_lat']:.2f}°N, "
          f"{top_cluster['center_lon']:.2f}°E")
    print("Expected: within IGP belt (28-32°N, 73-80°E)")


# ------------------------------------------------------------------------------
# Phase 9: Prep Dashboard Data
# ------------------------------------------------------------------------------
def run_prep_dashboard_data():
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("1. Generating DBSCAN grid (dbscan_grid.npy)...")
    with open('data/regridded/grid_spec.json') as f:
        gs = json.load(f)
    
    lats = np.linspace(gs['lat_max'], gs['lat_min'], gs['nlat'])
    lons = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'])
    consensus = np.load(os.path.join(RESULTS_DIR, 'hotspot_consensus.npy'))
    
    rows, cols = np.where(consensus)
    hotspot_coords = np.column_stack([lats[rows], lons[cols]])
    
    def simple_dbscan(X, eps, min_samples):
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        for i in range(n):
            if visited[i]: continue
            visited[i] = True
            neighbors = np.where(dist[i] <= eps)[0]
            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                seed_set = list(neighbors)
                if i in seed_set: seed_set.remove(i)
                while len(seed_set) > 0:
                    q = seed_set.pop(0)
                    if not visited[q]:
                        visited[q] = True
                        q_neighbors = np.where(dist[q] <= eps)[0]
                        if len(q_neighbors) >= min_samples:
                            for qn in q_neighbors:
                                if qn not in seed_set and not visited[qn]:
                                    seed_set.append(qn)
                    if labels[q] == -1:
                        labels[q] = cluster_id
                cluster_id += 1
        return labels
    
    labels = simple_dbscan(hotspot_coords, eps=0.5, min_samples=5)
    
    dbscan_grid = np.full((120, 124), -1, dtype=int)
    for idx, (r, c) in enumerate(zip(rows, cols)):
        dbscan_grid[r, c] = labels[idx]
    
    np.save(os.path.join(RESULTS_DIR, 'dbscan_grid.npy'), dbscan_grid)
    print("   -> dbscan_grid.npy saved.")
    
    print("2. Generating Disagreement View (hotspot_disagreement.npy)...")
    freq_A = np.load(os.path.join(RESULTS_DIR, 'hotspot_frequency_A.npy'))
    freq_C = np.load(os.path.join(RESULTS_DIR, 'hotspot_frequency_C.npy'))
    # Pixels where A flagged at least once, but C NEVER flagged
    disagreement = (freq_A > 0) & (freq_C == 0)
    np.save(os.path.join(RESULTS_DIR, 'hotspot_disagreement.npy'), disagreement)
    print("   -> hotspot_disagreement.npy saved.")
    
    print("3. Generating Uncertainty Mask (uncertainty_mask.npy)...")
    HCHO_stack = np.load(os.path.join(RESULTS_DIR, 'HCHO_daily_stack.npy'))
    # True where data is NaN (cloud cover / missing)
    uncertainty = np.isnan(HCHO_stack)
    np.save(os.path.join(RESULTS_DIR, 'uncertainty_mask.npy'), uncertainty)
    print("   -> uncertainty_mask.npy saved.")
    
    print("\nPreparation complete!")


if __name__ == '__main__':
    print('================================================================================')
    print('STARTING OBJECTIVE 2 PIPELINE')
    print('================================================================================\n')
    print('\n>>> Running run_phase1_hcho_load...')
    run_phase1_hcho_load()
    print('\n>>> Running run_phase2_hcho_stats...')
    run_phase2_hcho_stats()
    print('\n>>> Running run_phase3_fire_processing...')
    run_phase3_fire_processing()
    print('\n>>> Running run_phase4_hotspot_detection...')
    run_phase4_hotspot_detection()
    print('\n>>> Running run_phase5_hcho_maps...')
    run_phase5_hcho_maps()
    print('\n>>> Running run_phase6_correlation...')
    run_phase6_correlation()
    print('\n>>> Running run_phase7_transport...')
    run_phase7_transport()
    print('\n>>> Running run_phase8_source_regions...')
    run_phase8_source_regions()
    print('\n>>> Running run_phase8b_dbscan_clustering...')
    run_phase8b_dbscan_clustering()
    print('\n>>> Running run_prep_dashboard_data...')
    run_prep_dashboard_data()
    print('\n\nPipeline completed successfully!')
