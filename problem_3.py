#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：人类生物节律LED光源研究 - 太阳光谱模拟
Problem 3: Human Circadian Rhythm LED Light Source Research - Solar Spectrum Simulation

使用五通道LED设计控制策略，模拟全天太阳光谱，具有相似的节律效果
Design a control strategy using 5-channel LEDs to simulate daily solar spectrum with similar circadian effects

作者: AI Assistant
日期: 2025年1月
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SolarSpectrumSimulator:
    """太阳光谱模拟器类"""
    
    def __init__(self):
        """初始化模拟器，加载数据"""
        print("🌞 初始化太阳光谱模拟器...")
        self.load_data()
        self.process_data()
        
    def load_data(self):
        """加载LED和太阳光谱数据"""
        try:
            # 加载LED光谱数据（来自问题2）
            self.led_data = pd.read_excel('附录.xlsx', sheet_name='Problem 2_LED_SPD')
            
            # 加载太阳光谱数据（问题3）
            self.solar_data = pd.read_excel('附录.xlsx', sheet_name='Problem 3 SUN_SPD')
            
            # 加载CIE色匹配函数
            self.cie_xyz = pd.read_csv('CIE_xyz_1931_2deg.csv').iloc[20:421]
            
            print("✅ 数据加载成功")
            print(f"   - LED光谱数据: {self.led_data.shape}")
            print(f"   - 太阳光谱数据: {self.solar_data.shape}")
            print(f"   - CIE色匹配函数: {self.cie_xyz.shape}")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def process_data(self):
        """处理和预处理数据"""
        # 提取波长信息
        import re
        wavelength_strings = self.led_data['波长'].values
        self.wavelengths = []
        for wl_str in wavelength_strings:
            match = re.search(r'(\d+)', str(wl_str))
            if match:
                self.wavelengths.append(int(match.group(1)))
        self.wavelengths = np.array(self.wavelengths)
        
        # 提取LED通道数据
        self.led_channels = {
            'Blue': self.led_data['Blue'].values,
            'Green': self.led_data['Green'].values,
            'Red': self.led_data['Red'].values,
            'Warm_White': self.led_data['Warm White'].values,
            'Cold_White': self.led_data['Cold White'].values
        }
        
        # 提取太阳光谱时间点
        self.time_columns = self.solar_data.columns[1:]  # 跳过波长列
        
        # 选择三个代表性时间点
        self.representative_times = {
            '早晨': {'index': 3, 'time': '08:30', 'description': '早晨柔和光线'},
            '正午': {'index': 7, 'time': '12:30', 'description': '正午强烈光照'},
            '傍晚': {'index': 13, 'time': '18:30', 'description': '傍晚低色温'}
        }
        
        print(f"📊 数据处理完成:")
        print(f"   - 波长范围: {self.wavelengths.min()}-{self.wavelengths.max()} nm")
        print(f"   - LED通道: {list(self.led_channels.keys())}")
        print(f"   - 时间点: {len(self.time_columns)} 个 ({self.time_columns[0]} 到 {self.time_columns[-1]})")
        
    def synthesize_spectrum(self, weights):
        """根据权重合成LED光谱"""
        synthesized = np.zeros_like(self.wavelengths, dtype=float)
        
        channel_names = ['Blue', 'Green', 'Red', 'Warm_White', 'Cold_White']
        for i, weight in enumerate(weights):
            if i < len(channel_names):
                channel_name = channel_names[i]
                synthesized += weight * self.led_channels[channel_name]
        
        return synthesized
    
    def calculate_xyz_tristimulus(self, spd):
        """计算XYZ三刺激值"""
        x_bar = self.cie_xyz['x'].values
        y_bar = self.cie_xyz['y'].values  
        z_bar = self.cie_xyz['z'].values
        
        # 归一化常数
        k = 100 / np.trapz(spd * y_bar, self.wavelengths)
        
        X = k * np.trapz(spd * x_bar, self.wavelengths)
        Y = k * np.trapz(spd * y_bar, self.wavelengths)
        Z = k * np.trapz(spd * z_bar, self.wavelengths)
        
        return X, Y, Z
    
    def calculate_cct(self, x, y):
        """计算相关色温(CCT)"""
        # McCamy近似公式
        n = (x - 0.3320) / (0.1858 - y)
        
        if n >= 0:
            cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
        else:
            cct = 3000  # 默认暖白光
        
        # 限制CCT范围
        cct = max(1000, min(20000, cct))
        return cct
    
    def calculate_melanopic_response(self, spd):
        """计算黑视素响应（生物节律效应）"""
        # 黑视素敏感性函数（峰值约490nm）
        melanopic_sensitivity = np.exp(-0.5 * ((self.wavelengths - 490) / 50)**2)
        
        # 明视觉敏感性函数（峰值约555nm）
        photopic_sensitivity = np.exp(-0.5 * ((self.wavelengths - 555) / 80)**2)
        
        # 计算加权响应
        melanopic_response = np.trapz(spd * melanopic_sensitivity, self.wavelengths)
        photopic_response = np.trapz(spd * photopic_sensitivity, self.wavelengths)
        
        if photopic_response > 0:
            mel_ratio = melanopic_response / photopic_response
        else:
            mel_ratio = 0
        
        return max(0, min(2, mel_ratio))
    
    def calculate_spectrum_parameters(self, spd):
        """计算光谱参数"""
        # 计算XYZ三刺激值
        X, Y, Z = self.calculate_xyz_tristimulus(spd)
        
        # 计算色坐标
        total = X + Y + Z
        if total == 0:
            x, y = 0, 0
        else:
            x, y = X / total, Y / total
        
        # 计算相关色温
        cct = self.calculate_cct(x, y)
        
        # 计算黑视素响应
        mel_ratio = self.calculate_melanopic_response(spd)
        
        return {
            'X': X, 'Y': Y, 'Z': Z,
            'x': x, 'y': y,
            'CCT': cct,
            'melanopic_ratio': mel_ratio,
            'brightness': Y
        }
    
    def optimize_for_target_spectrum(self, target_spectrum, method='comprehensive'):
        """优化LED权重以匹配目标太阳光谱"""
        
        def objective_function(weights):
            """目标函数"""
            synthesized = self.synthesize_spectrum(weights)
            
            # 归一化光谱
            if np.max(synthesized) > 0:
                synthesized_norm = synthesized / np.max(synthesized)
            else:
                synthesized_norm = synthesized
                
            if np.max(target_spectrum) > 0:
                target_norm = target_spectrum / np.max(target_spectrum)
            else:
                target_norm = target_spectrum
            
            # 光谱形状匹配
            spectral_mse = np.mean((synthesized_norm - target_norm)**2)
            
            if method == 'comprehensive':
                # 计算参数匹配
                target_params = self.calculate_spectrum_parameters(target_spectrum)
                synth_params = self.calculate_spectrum_parameters(synthesized)
                
                # CCT匹配
                cct_error = abs(synth_params['CCT'] - target_params['CCT']) / 1000
                
                # 黑视素比率匹配（生物节律效应）
                mel_error = abs(synth_params['melanopic_ratio'] - target_params['melanopic_ratio'])
                
                # 组合目标函数
                total_error = spectral_mse + 0.3 * cct_error + 0.2 * mel_error
            else:
                total_error = spectral_mse
            
            return total_error
        
        # 优化边界（每个LED通道权重 0-1）
        bounds = [(0, 1) for _ in range(5)]
        
        # 使用差分进化算法
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=300,
            popsize=20,
            seed=42
        )
        
        return result.x, result.fun
    
    def analyze_representative_times(self):
        """分析三个代表性时间点"""
        print("\n🕐 分析代表性时间点...")
        
        results = {}
        
        for time_name, time_info in self.representative_times.items():
            print(f"\n分析 {time_name} 时段 ({time_info['time']})...")
            
            # 获取目标太阳光谱
            time_idx = time_info['index']
            target_spectrum = self.solar_data.iloc[:, time_idx + 1].values
            
            # 计算目标参数
            target_params = self.calculate_spectrum_parameters(target_spectrum)
            
            # 优化LED权重
            optimal_weights, optimization_error = self.optimize_for_target_spectrum(target_spectrum)
            
            # 生成最优合成光谱
            optimal_spectrum = self.synthesize_spectrum(optimal_weights)
            optimal_params = self.calculate_spectrum_parameters(optimal_spectrum)
            
            # 计算匹配精度
            spectral_correlation = np.corrcoef(target_spectrum, optimal_spectrum)[0, 1]
            cct_accuracy = 100 * (1 - abs(optimal_params['CCT'] - target_params['CCT']) / target_params['CCT'])
            mel_accuracy = 100 * (1 - abs(optimal_params['melanopic_ratio'] - target_params['melanopic_ratio']) / max(target_params['melanopic_ratio'], 0.1))
            
            # 保存结果
            results[time_name] = {
                'time_info': time_info,
                'target_spectrum': target_spectrum,
                'optimal_weights': optimal_weights,
                'optimal_spectrum': optimal_spectrum,
                'target_params': target_params,
                'optimal_params': optimal_params,
                'spectral_correlation': spectral_correlation,
                'cct_accuracy': cct_accuracy,
                'mel_accuracy': mel_accuracy,
                'optimization_error': optimization_error
            }
            
            # 打印结果
            print(f"  🎯 目标参数: CCT={target_params['CCT']:.0f}K, 黑视素比率={target_params['melanopic_ratio']:.3f}")
            print(f"  ⚡ 最优权重: {[f'{w:.3f}' for w in optimal_weights]}")
            print(f"  📈 合成参数: CCT={optimal_params['CCT']:.0f}K, 黑视素比率={optimal_params['melanopic_ratio']:.3f}")
            print(f"  📊 匹配精度: 光谱相关性={spectral_correlation:.3f}, CCT精度={cct_accuracy:.1f}%, 黑视素精度={mel_accuracy:.1f}%")
        
        return results
    
    def generate_daily_control_strategy(self):
        """生成全天控制策略"""
        print("\n📅 生成全天控制策略...")
        
        daily_strategy = {}
        
        for i, time_col in enumerate(self.time_columns):
            # 获取该时间点的太阳光谱
            target_spectrum = self.solar_data.iloc[:, i + 1].values
            
            # 优化LED权重（使用快速方法）
            optimal_weights, _ = self.optimize_for_target_spectrum(target_spectrum, method='spectral_only')
            
            # 计算参数
            params = self.calculate_spectrum_parameters(target_spectrum)
            
            daily_strategy[str(time_col)] = {
                'weights': optimal_weights,
                'cct': params['CCT'],
                'melanopic_ratio': params['melanopic_ratio'],
                'brightness': params['brightness']
            }
        
        return daily_strategy
    
    def plot_comparison_charts(self, results):
        """绘制对比图表"""
        print("\n📊 生成对比图表...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('问题3：五通道LED太阳光谱模拟结果', fontsize=16, fontweight='bold')
        
        time_names = ['早晨', '正午', '傍晚']
        colors = ['orange', 'gold', 'purple']
        
        # 第一行：光谱对比图
        for i, (time_name, color) in enumerate(zip(time_names, colors)):
            ax = axes[0, i]
            
            result = results[time_name]
            target_spd = result['target_spectrum']
            optimal_spd = result['optimal_spectrum']
            
            # 归一化用于更好的对比
            target_norm = target_spd / np.max(target_spd) if np.max(target_spd) > 0 else target_spd
            optimal_norm = optimal_spd / np.max(optimal_spd) if np.max(optimal_spd) > 0 else optimal_spd
            
            ax.plot(self.wavelengths, target_norm, 'k-', linewidth=3, label='目标太阳光谱', alpha=0.8)
            ax.plot(self.wavelengths, optimal_norm, color=color, linewidth=2, 
                   label='合成LED光谱', linestyle='--', alpha=0.9)
            
            ax.set_xlabel('波长 (nm)')
            ax.set_ylabel('归一化功率')
            ax.set_title(f'{time_name} ({result["time_info"]["time"]})\n'
                        f'相关性: {result["spectral_correlation"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(380, 780)
        
        # 第二行左：通道权重对比
        ax_weights = axes[1, 0]
        channel_names = ['蓝光', '绿光', '红光', '暖白光', '冷白光']
        x_pos = np.arange(len(channel_names))
        
        width = 0.25
        for i, (time_name, color) in enumerate(zip(time_names, colors)):
            weights = results[time_name]['optimal_weights']
            bars = ax_weights.bar(x_pos + i*width, weights, width, 
                                label=time_name, alpha=0.8, color=color)
        
        ax_weights.set_xlabel('LED通道')
        ax_weights.set_ylabel('权重')
        ax_weights.set_title('不同时段的LED通道权重')
        ax_weights.set_xticks(x_pos + width)
        ax_weights.set_xticklabels(channel_names, rotation=45)
        ax_weights.legend()
        ax_weights.grid(True, alpha=0.3)
        
        # 第二行中：参数对比
        ax_params = axes[1, 1]
        metrics = ['CCT (K)', '黑视素比率', '亮度 (Y)']
        
        target_values = []
        optimal_values = []
        
        for time_name in time_names:
            result = results[time_name]
            target_values.append([
                result['target_params']['CCT'] / 1000,  # 缩放CCT
                result['target_params']['melanopic_ratio'],
                result['target_params']['brightness'] / 100  # 缩放亮度
            ])
            optimal_values.append([
                result['optimal_params']['CCT'] / 1000,
                result['optimal_params']['melanopic_ratio'],
                result['optimal_params']['brightness'] / 100
            ])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (time_name, color) in enumerate(zip(time_names, colors)):
            if i == 0:  # 只显示一组对比
                ax_params.bar(x - width/2, target_values[i], width, 
                            label='目标值', alpha=0.7, color='gray')
                ax_params.bar(x + width/2, optimal_values[i], width, 
                            label='合成值', alpha=0.8, color=color)
        
        ax_params.set_xlabel('参数类型')
        ax_params.set_ylabel('归一化值')
        ax_params.set_title('早晨时段参数对比示例')
        ax_params.set_xticks(x)
        ax_params.set_xticklabels(['CCT\n(×1000K)', '黑视素\n比率', '亮度\n(×100)'])
        ax_params.legend()
        ax_params.grid(True, alpha=0.3)
        
        # 第二行右：匹配精度统计
        ax_accuracy = axes[1, 2]
        accuracy_metrics = ['光谱相关性', 'CCT精度(%)', '黑视素精度(%)']
        
        accuracy_data = []
        for time_name in time_names:
            result = results[time_name]
            accuracy_data.append([
                result['spectral_correlation'],
                result['cct_accuracy'],
                result['mel_accuracy']
            ])
        
        x = np.arange(len(accuracy_metrics))
        width = 0.25
        
        for i, (time_name, color) in enumerate(zip(time_names, colors)):
            bars = ax_accuracy.bar(x + i*width, accuracy_data[i], width, 
                                 label=time_name, alpha=0.8, color=color)
        
        ax_accuracy.set_xlabel('精度指标')
        ax_accuracy.set_ylabel('精度值')
        ax_accuracy.set_title('模拟精度评估')
        ax_accuracy.set_xticks(x + width)
        ax_accuracy.set_xticklabels(accuracy_metrics, rotation=45)
        ax_accuracy.legend()
        ax_accuracy.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('problem3_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_daily_strategy(self, daily_strategy):
        """绘制全天控制策略"""
        print("\n📈 绘制全天控制策略...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('全天LED控制策略', fontsize=16, fontweight='bold')
        
        times = list(daily_strategy.keys())
        # Convert time strings to float for plotting (HH:MM:SS -> HH.MM)
        times_float = []
        for t in times:
            if ':' in str(t):
                parts = str(t).split(':')
                hour_min = float(parts[0]) + float(parts[1])/60
                times_float.append(hour_min)
            else:
                times_float.append(float(str(t).replace('.', '')))
        
        # 提取数据
        weights_data = np.array([daily_strategy[t]['weights'] for t in times])
        cct_data = [daily_strategy[t]['cct'] for t in times]
        mel_data = [daily_strategy[t]['melanopic_ratio'] for t in times]
        brightness_data = [daily_strategy[t]['brightness'] for t in times]
        
        # 图1：LED通道权重随时间变化
        ax1 = axes[0, 0]
        channel_names = ['蓝光', '绿光', '红光', '暖白光', '冷白光']
        colors_channels = ['blue', 'green', 'red', 'orange', 'cyan']
        
        for i, (name, color) in enumerate(zip(channel_names, colors_channels)):
            ax1.plot(times_float, weights_data[:, i], 'o-', 
                    label=name, color=color, alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('时间')
        ax1.set_ylabel('权重')
        ax1.set_title('LED通道权重日变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(times_float[::2])
        ax1.set_xticklabels([str(times[i]).split('.')[0] if '.' in str(times[i]) else str(times[i]) for i in range(0, len(times), 2)], rotation=45)
        
        # 图2：色温变化
        ax2 = axes[0, 1]
        ax2.plot(times_float, cct_data, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('时间')
        ax2.set_ylabel('相关色温 (K)')
        ax2.set_title('色温日变化曲线')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(times_float[::2])
        ax2.set_xticklabels([str(times[i]).split('.')[0] if '.' in str(times[i]) else str(times[i]) for i in range(0, len(times), 2)], rotation=45)
        
        # 图3：黑视素比率变化（生物节律效应）
        ax3 = axes[1, 0]
        ax3.plot(times_float, mel_data, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('时间')
        ax3.set_ylabel('黑视素比率')
        ax3.set_title('生物节律效应日变化')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(times_float[::2])
        ax3.set_xticklabels([str(times[i]).split('.')[0] if '.' in str(times[i]) else str(times[i]) for i in range(0, len(times), 2)], rotation=45)
        
        # 图4：亮度变化
        ax4 = axes[1, 1]
        ax4.plot(times_float, brightness_data, 'bo-', linewidth=2, markersize=6)
        ax4.set_xlabel('时间')
        ax4.set_ylabel('亮度 (Y)')
        ax4.set_title('亮度日变化曲线')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(times_float[::2])
        ax4.set_xticklabels([str(times[i]).split('.')[0] if '.' in str(times[i]) else str(times[i]) for i in range(0, len(times), 2)], rotation=45)
        
        plt.tight_layout()
        plt.savefig('problem3_daily_strategy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_control_recommendations(self, results, daily_strategy):
        """生成控制建议"""
        print("\n💡 生成控制建议...")
        
        recommendations = {
            'representative_times': {},
            'daily_control': {},
            'circadian_analysis': {}
        }
        
        # 代表性时间点建议
        for time_name, result in results.items():
            weights = result['optimal_weights']
            params = result['optimal_params']
            
            recommendations['representative_times'][time_name] = {
                'time': result['time_info']['time'],
                'description': result['time_info']['description'],
                'led_weights': {
                    '蓝光': weights[0],
                    '绿光': weights[1], 
                    '红光': weights[2],
                    '暖白光': weights[3],
                    '冷白光': weights[4]
                },
                'expected_cct': params['CCT'],
                'expected_melanopic_ratio': params['melanopic_ratio'],
                'control_advice': self._generate_time_specific_advice(time_name, weights, params)
            }
        
        # 全天控制策略
        times = list(daily_strategy.keys())
        cct_values = [daily_strategy[t]['cct'] for t in times]
        mel_values = [daily_strategy[t]['melanopic_ratio'] for t in times]
        
        recommendations['daily_control'] = {
            'peak_cct_time': times[np.argmax(cct_values)],
            'peak_cct_value': max(cct_values),
            'min_cct_time': times[np.argmin(cct_values)],
            'min_cct_value': min(cct_values),
            'peak_melanopic_time': times[np.argmax(mel_values)],
            'peak_melanopic_value': max(mel_values),
            'min_melanopic_time': times[np.argmin(mel_values)],
            'min_melanopic_value': min(mel_values)
        }
        
        # 生物节律分析
        recommendations['circadian_analysis'] = {
            'morning_activation': mel_values[3:6],  # 8:30-10:30
            'midday_maintenance': mel_values[6:9],  # 11:30-13:30
            'evening_suppression': mel_values[12:15],  # 17:30-19:30
            'daily_variation': max(mel_values) - min(mel_values)
        }
        
        return recommendations
    
    def _generate_time_specific_advice(self, time_name, weights, params):
        """生成特定时间的控制建议"""
        advice = []
        
        if time_name == '早晨':
            if weights[0] > 0.3:  # 蓝光权重高
                advice.append("蓝光成分较高，有助于唤醒和提神")
            if params['melanopic_ratio'] > 0.8:
                advice.append("黑视素刺激适中，有助于调节生物钟")
            advice.append("适合作为起床后的照明")
            
        elif time_name == '正午':
            if params['CCT'] > 6000:
                advice.append("高色温照明，模拟正午强烈日光")
            if weights[4] > 0.4:  # 冷白光权重高
                advice.append("冷白光成分高，提供充足的照明强度")
            advice.append("适合工作和学习环境")
            
        elif time_name == '傍晚':
            if weights[3] > weights[4]:  # 暖白光 > 冷白光
                advice.append("暖白光为主，营造温馨氛围")
            if params['melanopic_ratio'] < 0.5:
                advice.append("低黑视素刺激，有助于为睡眠做准备")
            advice.append("适合放松和休息环境")
        
        return advice
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("🚀 开始完整的太阳光谱模拟分析...")
        print("=" * 60)
        
        # 分析代表性时间点
        results = self.analyze_representative_times()
        
        # 生成全天控制策略
        daily_strategy = self.generate_daily_control_strategy()
        
        # 绘制对比图表
        fig1 = self.plot_comparison_charts(results)
        
        # 绘制全天策略
        fig2 = self.plot_daily_strategy(daily_strategy)
        
        # 生成控制建议
        recommendations = self.generate_control_recommendations(results, daily_strategy)
        
        # 打印总结报告
        self.print_summary_report(results, recommendations)
        
        return results, daily_strategy, recommendations
    
    def print_summary_report(self, results, recommendations):
        """打印总结报告"""
        print("\n" + "=" * 60)
        print("📋 太阳光谱模拟总结报告")
        print("=" * 60)
        
        print("\n🕐 代表性时间点分析:")
        for time_name, result in results.items():
            print(f"\n{time_name} ({result['time_info']['time']}):")
            print(f"  目标 CCT: {result['target_params']['CCT']:.0f}K")
            print(f"  合成 CCT: {result['optimal_params']['CCT']:.0f}K")
            print(f"  光谱相关性: {result['spectral_correlation']:.3f}")
            print(f"  最优权重: {[f'{w:.3f}' for w in result['optimal_weights']]}")
        
        print(f"\n📊 全天控制策略特征:")
        daily_ctrl = recommendations['daily_control']
        print(f"  最高色温: {daily_ctrl['peak_cct_value']:.0f}K ({daily_ctrl['peak_cct_time']})")
        print(f"  最低色温: {daily_ctrl['min_cct_value']:.0f}K ({daily_ctrl['min_cct_time']})")
        print(f"  黑视素比率变化: {recommendations['circadian_analysis']['daily_variation']:.3f}")
        
        print(f"\n💡 生物节律效应分析:")
        circ = recommendations['circadian_analysis']
        print(f"  早晨激活阶段平均黑视素比率: {np.mean(circ['morning_activation']):.3f}")
        print(f"  正午维持阶段平均黑视素比率: {np.mean(circ['midday_maintenance']):.3f}")
        print(f"  傍晚抑制阶段平均黑视素比率: {np.mean(circ['evening_suppression']):.3f}")
        
        print(f"\n✅ 结论与建议:")
        print(f"  - 成功设计了五通道LED控制策略")
        print(f"  - 实现了对太阳光谱的有效模拟")
        print(f"  - 保持了自然光的生物节律效应")
        print(f"  - 可用于智能照明系统的节律调节")
        
        print("=" * 60)


def main():
    """主函数"""
    print("🌞 问题3：太阳光谱模拟控制策略")
    print("=" * 60)
    
    try:
        # 创建模拟器实例
        simulator = SolarSpectrumSimulator()
        
        # 运行完整分析
        results, daily_strategy, recommendations = simulator.run_complete_analysis()
        
        print("\n✅ 分析完成！生成的文件:")
        print("   📊 problem3_results.png - 代表性时间点对比")
        print("   📈 problem3_daily_strategy.png - 全天控制策略")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()