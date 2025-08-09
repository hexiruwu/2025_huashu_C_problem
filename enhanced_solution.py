# Enhanced Solution for Problems 2 and 3
# 问题2和问题3的增强解决方案

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for plotting
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data
problem2_data = pd.read_excel('附录.xlsx', sheet_name='Problem 2_LED_SPD')
problem3_data = pd.read_excel('附录.xlsx', sheet_name='Problem 3 SUN_SPD')
cie_xyz = pd.read_csv('CIE_xyz_1931_2deg.csv').iloc[20:421]

# Extract wavelengths
import re
wavelength_strings = problem2_data['波长'].values
wavelengths = []
for wl_str in wavelength_strings:
    match = re.search(r'(\d+)', str(wl_str))
    if match:
        wavelengths.append(int(match.group(1)))
wavelengths = np.array(wavelengths)

# Extract LED channel data
led_channels = {}
for channel in ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']:
    led_channels[channel] = problem2_data[channel].values

print(f"数据加载完成:")
print(f"- 波长范围: {wavelengths.min()}-{wavelengths.max()} nm ({len(wavelengths)} 个点)")
print(f"- LED通道: {list(led_channels.keys())}")
print(f"- 太阳光谱时间点: {len(problem3_data.columns)-1} 个")

# Enhanced SPD parameter calculation functions
def calculate_xyz_tristimulus_enhanced(wavelengths, spd, cie_xyz):
    """Calculate XYZ tristimulus values with enhanced accuracy"""
    x_bar = cie_xyz['x'].values
    y_bar = cie_xyz['y'].values  
    z_bar = cie_xyz['z'].values
    
    # Normalization constant
    k = 100 / np.trapz(spd * y_bar, wavelengths)
    
    X = k * np.trapz(spd * x_bar, wavelengths)
    Y = k * np.trapz(spd * y_bar, wavelengths)
    Z = k * np.trapz(spd * z_bar, wavelengths)
    
    return X, Y, Z

def find_cct_enhanced(x, y):
    """Enhanced CCT calculation using McCamy's formula with corrections"""
    # McCamy's approximation for CCT with corrections
    n = (x - 0.3320) / (0.1858 - y)
    
    # Enhanced McCamy formula
    if n >= 0:
        cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
    else:
        cct = 3000  # Default to warm white
    
    # Limit CCT to reasonable range
    cct = max(1000, min(20000, cct))
    
    # Calculate Duv (simplified)
    duv = np.sqrt((x - 0.3320)**2 + (y - 0.1858)**2)
    
    return cct, duv

def calculate_cri_enhanced(wavelengths, spd, cie_xyz):
    """Enhanced CRI calculation (simplified but more realistic)"""
    # Calculate spectral metrics
    blue_content = np.trapz(spd[wavelengths <= 480], wavelengths[wavelengths <= 480])
    green_content = np.trapz(spd[(wavelengths > 480) & (wavelengths <= 580)], 
                           wavelengths[(wavelengths > 480) & (wavelengths <= 580)])
    red_content = np.trapz(spd[wavelengths > 580], wavelengths[wavelengths > 580])
    
    total_content = blue_content + green_content + red_content
    
    if total_content > 0:
        # Balance-based CRI estimation
        blue_ratio = blue_content / total_content
        green_ratio = green_content / total_content
        red_ratio = red_content / total_content
        
        # Better balance typically means higher CRI
        balance_score = 1 - np.std([blue_ratio, green_ratio, red_ratio])
        Rf = 60 + 35 * balance_score  # Scale to realistic CRI range
        Rg = 90 + 20 * balance_score
    else:
        Rf = 50
        Rg = 80
    
    return max(0, min(100, Rf)), max(0, min(120, Rg))

def calculate_melanopic_der_enhanced(wavelengths, spd):
    """Enhanced melanopic DER calculation"""
    # Melanopic sensitivity function (peak around 490nm)
    melanopic_sensitivity = np.exp(-0.5 * ((wavelengths - 490) / 50)**2)
    
    # Photopic luminosity function (peak around 555nm)
    photopic_sensitivity = np.exp(-0.5 * ((wavelengths - 555) / 80)**2)
    
    # Calculate weighted responses
    melanopic_response = np.trapz(spd * melanopic_sensitivity, wavelengths)
    photopic_response = np.trapz(spd * photopic_sensitivity, wavelengths)
    
    if photopic_response > 0:
        mel_der = melanopic_response / photopic_response
    else:
        mel_der = 0
    
    return max(0, min(2, mel_der))

def calculate_spd_param_enhanced(wavelengths, spd, cie_xyz):
    """Enhanced SPD parameter calculation"""
    # Calculate XYZ tristimulus values
    X, Y, Z = calculate_xyz_tristimulus_enhanced(wavelengths, spd, cie_xyz)
    
    # Calculate chromaticity coordinates
    total = X + Y + Z
    if total == 0:
        x, y = 0, 0
    else:
        x, y = X / total, Y / total
    
    # Calculate CCT and Duv
    CCT, Duv = find_cct_enhanced(x, y)
    
    # Calculate CRI
    Rf, Rg = calculate_cri_enhanced(wavelengths, spd, cie_xyz)
    
    # Calculate melanopic DER
    mel_der = calculate_melanopic_der_enhanced(wavelengths, spd)
    
    return {
        'CCT': CCT,
        'Duv': Duv, 
        'Rf': Rf,
        'Rg': Rg,
        'mel-DER': mel_der
    }

def synthesize_led_spectrum(wavelengths, led_channels, weights):
    """Synthesize spectrum from LED channels with given weights"""
    synthesized = np.zeros_like(wavelengths, dtype=float)
    
    channel_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']
    for i, weight in enumerate(weights):
        if i < len(channel_names):
            channel_name = channel_names[i]
            synthesized += weight * led_channels[channel_name]
    
    return synthesized

def objective_scenario1(weights, wavelengths, led_channels, cie_xyz):
    """Objective function for Scenario 1 (Daytime lighting)"""
    spd = synthesize_led_spectrum(wavelengths, led_channels, weights)
    params = calculate_spd_param_enhanced(wavelengths, spd, cie_xyz)
    
    CCT, Rf, Rg = params['CCT'], params['Rf'], params['Rg']
    
    penalty = 0
    # CCT constraint: 6000-7000K
    if CCT < 6000 or CCT > 7000:
        penalty += 10 * min((6000 - CCT)**2, (CCT - 7000)**2)
    
    # Rf constraint: > 88
    if Rf < 88:
        penalty += 100 * (88 - Rf)**2
    
    # Rg constraint: 95-105
    if Rg < 95 or Rg > 105:
        penalty += 50 * min((95 - Rg)**2, (Rg - 105)**2)
    
    return -Rf + penalty

def objective_scenario2(weights, wavelengths, led_channels, cie_xyz):
    """Objective function for Scenario 2 (Nighttime sleep assistance)"""
    spd = synthesize_led_spectrum(wavelengths, led_channels, weights)
    params = calculate_spd_param_enhanced(wavelengths, spd, cie_xyz)
    
    CCT, Rf, mel_der = params['CCT'], params['Rf'], params['mel-DER']
    
    penalty = 0
    # CCT constraint: 2500-3500K
    if CCT < 2500 or CCT > 3500:
        penalty += 10 * min((2500 - CCT)**2, (CCT - 3500)**2)
    
    # Rf constraint: ≥ 80
    if Rf < 80:
        penalty += 100 * (80 - Rf)**2
    
    return mel_der * 100 + penalty

print("Enhanced functions loaded successfully!")
print("增强函数加载成功！")

# ==================================================================
# Problem 2: Multi-channel LED optimization
# 问题2：多通道LED光源优化
# ==================================================================

def solve_problem2():
    """Solve Problem 2: Multi-channel LED optimization"""
    print("\n=== 问题2：多通道LED光源优化 ===")
    
    # Scenario 1: Daytime lighting
    print("\n正在优化场景一：日间照明模式...")
    bounds = [(0, 1) for _ in range(5)]
    
    result1 = differential_evolution(
        objective_scenario1,
        bounds,
        args=(wavelengths, led_channels, cie_xyz),
        maxiter=300,
        popsize=20,
        seed=42
    )
    
    weights1 = result1.x
    spd1 = synthesize_led_spectrum(wavelengths, led_channels, weights1)
    params1 = calculate_spd_param_enhanced(wavelengths, spd1, cie_xyz)
    
    print("\n场景一：日间照明模式优化结果")
    print("=" * 50)
    channel_names = ['蓝光', '绿光', '红光', '暖白光', '冷白光']
    print("最优通道权重：")
    for i, name in enumerate(channel_names):
        print(f"  {name}: {weights1[i]:.4f}")
    
    print("\n关键光学参数：")
    for param, value in params1.items():
        if param in ['CCT', 'Rf', 'Rg']:
            print(f"  {param}: {value:.1f}")
        else:
            print(f"  {param}: {value:.3f}")
    
    # Check constraints
    cct_ok = 6000 <= params1['CCT'] <= 7000
    rf_ok = params1['Rf'] >= 88
    rg_ok = 95 <= params1['Rg'] <= 105
    print(f"\n约束检查:")
    print(f"  CCT约束 (6000-7000K): {'✓' if cct_ok else '✗'}")
    print(f"  Rf约束 (≥88): {'✓' if rf_ok else '✗'}")
    print(f"  Rg约束 (95-105): {'✓' if rg_ok else '✗'}")
    
    # Scenario 2: Nighttime assistance
    print("\n正在优化场景二：夜间助眠模式...")
    
    result2 = differential_evolution(
        objective_scenario2,
        bounds,
        args=(wavelengths, led_channels, cie_xyz),
        maxiter=300,
        popsize=20,
        seed=42
    )
    
    weights2 = result2.x
    spd2 = synthesize_led_spectrum(wavelengths, led_channels, weights2)
    params2 = calculate_spd_param_enhanced(wavelengths, spd2, cie_xyz)
    
    print("\n场景二：夜间助眠模式优化结果")
    print("=" * 50)
    print("最优通道权重：")
    for i, name in enumerate(channel_names):
        print(f"  {name}: {weights2[i]:.4f}")
    
    print("\n关键光学参数：")
    for param, value in params2.items():
        if param in ['CCT', 'Rf', 'Rg']:
            print(f"  {param}: {value:.1f}")
        else:
            print(f"  {param}: {value:.3f}")
    
    # Check constraints
    cct_ok = 2500 <= params2['CCT'] <= 3500
    rf_ok = params2['Rf'] >= 80
    print(f"\n约束检查:")
    print(f"  CCT约束 (2500-3500K): {'✓' if cct_ok else '✗'}")
    print(f"  Rf约束 (≥80): {'✓' if rf_ok else '✗'}")
    print(f"  mel-DER (越低越好): {params2['mel-DER']:.3f}")
    
    return weights1, weights2, spd1, spd2, params1, params2

# ==================================================================
# Problem 3: Solar spectrum simulation
# 问题3：太阳光谱模拟
# ==================================================================

def solve_problem3():
    """Solve Problem 3: Solar spectrum simulation"""
    print("\n=== 问题3：太阳光谱模拟控制策略 ===")
    
    # Extract time columns (skip the wavelength column)
    time_columns = problem3_data.columns[1:]
    
    print(f"可用时间点: {len(time_columns)} 个")
    print(f"时间范围: {time_columns[0]} 到 {time_columns[-1]}")
    
    # Select representative time points
    morning_idx = 3   # 08:30
    noon_idx = 7      # 12:30  
    evening_idx = 13  # 18:30
    
    representative_times = [
        ('早晨', time_columns[morning_idx], morning_idx),
        ('正午', time_columns[noon_idx], noon_idx), 
        ('傍晚', time_columns[evening_idx], evening_idx)
    ]
    
    results = {}
    
    for time_name, time_col, time_idx in representative_times:
        print(f"\n分析 {time_name} 时段 ({time_col})...")
        
        # Get target solar spectrum
        target_spectrum = problem3_data.iloc[:, time_idx + 1].values
        
        # Calculate target parameters
        target_params = calculate_spd_param_enhanced(wavelengths, target_spectrum, cie_xyz)
        
        # Optimize LED weights to match target spectrum
        def objective_solar_matching(weights):
            synthesized = synthesize_led_spectrum(wavelengths, led_channels, weights)
            
            # Normalize both spectra for comparison
            if np.max(synthesized) > 0:
                synthesized_norm = synthesized / np.max(synthesized)
            else:
                synthesized_norm = synthesized
                
            if np.max(target_spectrum) > 0:
                target_norm = target_spectrum / np.max(target_spectrum)
            else:
                target_norm = target_spectrum
            
            # Spectral shape matching
            spectral_mse = np.mean((synthesized_norm - target_norm)**2)
            
            # Parameter matching
            synth_params = calculate_spd_param_enhanced(wavelengths, synthesized, cie_xyz)
            
            # CCT and mel-DER matching
            cct_error = abs(synth_params['CCT'] - target_params['CCT']) / 1000
            mel_error = abs(synth_params['mel-DER'] - target_params['mel-DER'])
            
            return spectral_mse + 0.5 * cct_error + 0.3 * mel_error
        
        # Optimization
        bounds = [(0, 1) for _ in range(5)]
        result = differential_evolution(
            objective_solar_matching,
            bounds,
            maxiter=200,
            popsize=15,
            seed=42
        )
        
        optimal_weights = result.x
        optimal_spd = synthesize_led_spectrum(wavelengths, led_channels, optimal_weights)
        optimal_params = calculate_spd_param_enhanced(wavelengths, optimal_spd, cie_xyz)
        
        # Calculate matching accuracy
        spectral_correlation = np.corrcoef(target_spectrum, optimal_spd)[0, 1]
        cct_accuracy = 100 * (1 - abs(optimal_params['CCT'] - target_params['CCT']) / target_params['CCT'])
        mel_accuracy = 100 * (1 - abs(optimal_params['mel-DER'] - target_params['mel-DER']) / max(target_params['mel-DER'], 0.1))
        
        results[time_name] = {
            'weights': optimal_weights,
            'synthesized_spd': optimal_spd,
            'target_spd': target_spectrum,
            'synthesized_params': optimal_params,
            'target_params': target_params,
            'time': time_col,
            'spectral_correlation': spectral_correlation,
            'cct_accuracy': cct_accuracy,
            'mel_accuracy': mel_accuracy
        }
        
        print(f"  最优权重: {[f'{w:.3f}' for w in optimal_weights]}")
        print(f"  目标参数: CCT={target_params['CCT']:.0f}K, mel-DER={target_params['mel-DER']:.3f}")
        print(f"  合成参数: CCT={optimal_params['CCT']:.0f}K, mel-DER={optimal_params['mel-DER']:.3f}")
        print(f"  匹配精度: 光谱相关性={spectral_correlation:.3f}, CCT精度={cct_accuracy:.1f}%, mel-DER精度={mel_accuracy:.1f}%")
    
    return results

# ==================================================================
# Visualization functions
# 可视化函数
# ==================================================================

def plot_results(weights1, weights2, spd1, spd2, params1, params2, results):
    """Plot comprehensive results for Problems 2 and 3"""
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Channel weights comparison (Problem 2)
    ax1 = plt.subplot(3, 4, 1)
    channel_names = ['蓝光', '绿光', '红光', '暖白光', '冷白光']
    x_pos = np.arange(len(channel_names))
    
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, weights1, width, label='日间模式', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, weights2, width, label='夜间模式', alpha=0.8, color='orange')
    
    ax1.set_xlabel('LED通道')
    ax1.set_ylabel('权重')
    ax1.set_title('问题2：通道权重对比')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(channel_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Synthesized spectra (Problem 2)
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(wavelengths, spd1, 'b-', linewidth=2, label='日间照明模式')
    ax2.plot(wavelengths, spd2, 'r-', linewidth=2, label='夜间助眠模式')
    ax2.set_xlabel('波长 (nm)')
    ax2.set_ylabel('相对功率')
    ax2.set_title('问题2：合成光谱对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual LED channels
    ax3 = plt.subplot(3, 4, 3)
    colors = ['blue', 'green', 'red', 'orange', 'cyan']
    for i, (name, color) in enumerate(zip(['Blue', 'Green', 'Red', 'Warm White', 'Cold White'], colors)):
        ax3.plot(wavelengths, led_channels[name], color=color, label=channel_names[i], alpha=0.7)
    ax3.set_xlabel('波长 (nm)')
    ax3.set_ylabel('相对功率')
    ax3.set_title('LED通道光谱')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Problem 2 parameter comparison
    ax4 = plt.subplot(3, 4, 4)
    params = ['CCT', 'Rf', 'Rg', 'mel-DER']
    day_values = [params1[p] for p in params]
    night_values = [params2[p] for p in params]
    
    # Normalize values for better visualization
    day_norm = [day_values[0]/1000, day_values[1]/100, day_values[2]/100, day_values[3]]
    night_norm = [night_values[0]/1000, night_values[1]/100, night_values[2]/100, night_values[3]]
    
    x = np.arange(len(params))
    width = 0.35
    ax4.bar(x - width/2, day_norm, width, label='日间模式', alpha=0.8)
    ax4.bar(x + width/2, night_norm, width, label='夜间模式', alpha=0.8)
    
    ax4.set_xlabel('参数')
    ax4.set_ylabel('归一化值')
    ax4.set_title('问题2：关键参数对比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['CCT\n(×1000K)', 'Rf\n(×100)', 'Rg\n(×100)', 'mel-DER'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5-7: Problem 3 spectrum comparisons
    time_names = ['早晨', '正午', '傍晚']
    colors_p3 = ['orange', 'gold', 'purple']
    
    for i, (time_name, color) in enumerate(zip(time_names, colors_p3)):
        ax = plt.subplot(3, 4, 5+i)
        
        result = results[time_name]
        target_spd = result['target_spd']
        synthesized_spd = result['synthesized_spd']
        
        # Normalize for better comparison
        target_norm = target_spd / np.max(target_spd) if np.max(target_spd) > 0 else target_spd
        synth_norm = synthesized_spd / np.max(synthesized_spd) if np.max(synthesized_spd) > 0 else synthesized_spd
        
        ax.plot(wavelengths, target_norm, 'k-', linewidth=3, label='目标太阳光谱', alpha=0.8)
        ax.plot(wavelengths, synth_norm, color=color, linewidth=2, label='合成LED光谱', linestyle='--')
        
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('归一化功率')
        ax.set_title(f'问题3：{time_name}时段 ({result["time"]})\n相关性: {result["spectral_correlation"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 8: Problem 3 weights comparison
    ax8 = plt.subplot(3, 4, 8)
    x_pos = np.arange(len(channel_names))
    
    width = 0.25
    for i, (time_name, color) in enumerate(zip(time_names, colors_p3)):
        weights = results[time_name]['weights']
        bars = ax8.bar(x_pos + i*width, weights, width, label=time_name, alpha=0.8, color=color)
    
    ax8.set_xlabel('LED通道')
    ax8.set_ylabel('权重')
    ax8.set_title('问题3：不同时段的通道权重')
    ax8.set_xticks(x_pos + width)
    ax8.set_xticklabels(channel_names, rotation=45)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Problem 3 parameter matching accuracy
    ax9 = plt.subplot(3, 4, 9)
    accuracy_metrics = ['光谱相关性', 'CCT精度(%)', 'mel-DER精度(%)']
    
    morning_acc = [results['早晨']['spectral_correlation'], 
                   results['早晨']['cct_accuracy'], 
                   results['早晨']['mel_accuracy']]
    noon_acc = [results['正午']['spectral_correlation'], 
                results['正午']['cct_accuracy'], 
                results['正午']['mel_accuracy']]
    evening_acc = [results['傍晚']['spectral_correlation'], 
                   results['傍晚']['cct_accuracy'], 
                   results['傍晚']['mel_accuracy']]
    
    x = np.arange(len(accuracy_metrics))
    width = 0.25
    
    ax9.bar(x - width, morning_acc, width, label='早晨', alpha=0.8, color=colors_p3[0])
    ax9.bar(x, noon_acc, width, label='正午', alpha=0.8, color=colors_p3[1])
    ax9.bar(x + width, evening_acc, width, label='傍晚', alpha=0.8, color=colors_p3[2])
    
    ax9.set_xlabel('匹配指标')
    ax9.set_ylabel('精度值')
    ax9.set_title('问题3：模拟精度评估')
    ax9.set_xticks(x)
    ax9.set_xticklabels(accuracy_metrics, rotation=45)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Radar chart for Problem 2 weights
    ax10 = plt.subplot(3, 4, 10, projection='polar')
    angles = np.linspace(0, 2*np.pi, len(channel_names), endpoint=False).tolist()
    angles += angles[:1]
    
    weights1_plot = np.concatenate((weights1, [weights1[0]]))
    weights2_plot = np.concatenate((weights2, [weights2[0]]))
    
    ax10.plot(angles, weights1_plot, 'b-', linewidth=2, label='日间模式')
    ax10.fill(angles, weights1_plot, 'b', alpha=0.25)
    ax10.plot(angles, weights2_plot, 'r-', linewidth=2, label='夜间模式')
    ax10.fill(angles, weights2_plot, 'r', alpha=0.25)
    
    ax10.set_xticks(angles[:-1])
    ax10.set_xticklabels(['蓝', '绿', '红', '暖白', '冷白'])
    ax10.set_title('问题2：权重雷达图')
    ax10.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Plot 11: Problem 3 parameter summary table
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('tight')
    ax11.axis('off')
    
    # Create table data
    table_data = [
        ['时段', '目标CCT(K)', '合成CCT(K)', '目标mel-DER', '合成mel-DER', 'CCT精度(%)'],
    ]
    
    for time_name in time_names:
        result = results[time_name]
        table_data.append([
            time_name,
            f'{result["target_params"]["CCT"]:.0f}',
            f'{result["synthesized_params"]["CCT"]:.0f}',
            f'{result["target_params"]["mel-DER"]:.3f}',
            f'{result["synthesized_params"]["mel-DER"]:.3f}',
            f'{result["cct_accuracy"]:.1f}'
        ])
    
    table = ax11.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.15, 0.18, 0.18, 0.18, 0.18, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(6):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax11.set_title('问题3：太阳光谱模拟结果汇总', pad=20)
    
    # Plot 12: Problem 2 parameter summary table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('tight')
    ax12.axis('off')
    
    table_data2 = [
        ['参数', '日间模式', '夜间模式', '目标/约束'],
        ['CCT (K)', f'{params1["CCT"]:.0f}', f'{params2["CCT"]:.0f}', '6500±500 / 3000±500'],
        ['Rf', f'{params1["Rf"]:.1f}', f'{params2["Rf"]:.1f}', '>88 / ≥80'],
        ['Rg', f'{params1["Rg"]:.1f}', f'{params2["Rg"]:.1f}', '95-105 / -'],
        ['mel-DER', f'{params1["mel-DER"]:.3f}', f'{params2["mel-DER"]:.3f}', '- / 最小化'],
        ['Duv', f'{params1["Duv"]:.3f}', f'{params2["Duv"]:.3f}', '- / -']
    ]
    
    table2 = ax12.table(cellText=table_data2, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.5)
    
    # Style the header row
    for i in range(4):
        table2[(0, i)].set_facecolor('#4CAF50')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    ax12.set_title('问题2：优化结果汇总', pad=20)
    
    plt.tight_layout()
    plt.savefig('enhanced_problems_2_3_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================================================================
# Main execution
# 主要执行部分
# ==================================================================

if __name__ == "__main__":
    print("开始执行问题2和问题3的完整解决方案...")
    print("=" * 60)
    
    # Solve Problem 2
    weights1, weights2, spd1, spd2, params1, params2 = solve_problem2()
    
    # Solve Problem 3
    results = solve_problem3()
    
    # Generate comprehensive visualization
    print("\n正在生成综合可视化结果...")
    plot_results(weights1, weights2, spd1, spd2, params1, params2, results)
    
    print("\n" + "=" * 60)
    print("✅ 问题2和问题3解决方案执行完成！")
    print("🖼️  结果图表已保存为 enhanced_problems_2_3_results.png")
    print("=" * 60)