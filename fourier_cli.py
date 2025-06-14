#!/usr/bin/env python3
"""
フーリエ変換視覚化CLI
時間領域の信号を周波数領域に変換して視覚的に理解するためのツール
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from typing import Tuple, Callable
import matplotlib as mpl

# 日本語フォントの設定
import platform
if platform.system() == 'Darwin':  # macOS
    mpl.rcParams['font.family'] = 'Hiragino Sans'
elif platform.system() == 'Windows':
    mpl.rcParams['font.family'] = 'MS Gothic'
else:  # Linux
    mpl.rcParams['font.family'] = 'Noto Sans CJK JP'

# マイナス記号の文字化けを防ぐ
mpl.rcParams['axes.unicode_minus'] = False

class FourierVisualizer:
    def __init__(self, duration: float = 2.0, sample_rate: float = 1000.0):
        """
        Args:
            duration: 信号の継続時間（秒）
            sample_rate: サンプリングレート（Hz）
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
    def generate_signal(self, signal_type: str, **kwargs) -> np.ndarray:
        """各種信号を生成"""
        if signal_type == "sine":
            freq = kwargs.get('frequency', 5)
            amplitude = kwargs.get('amplitude', 1)
            phase = kwargs.get('phase', 0)
            return amplitude * np.sin(2 * np.pi * freq * self.t + phase)
            
        elif signal_type == "composite":
            # 複数の正弦波の合成
            freqs = kwargs.get('frequencies', [5, 10, 15])
            amps = kwargs.get('amplitudes', [1, 0.5, 0.3])
            signal = np.zeros_like(self.t)
            for f, a in zip(freqs, amps):
                signal += a * np.sin(2 * np.pi * f * self.t)
            return signal
            
        elif signal_type == "square":
            freq = kwargs.get('frequency', 5)
            return np.sign(np.sin(2 * np.pi * freq * self.t))
            
        elif signal_type == "sawtooth":
            freq = kwargs.get('frequency', 5)
            return 2 * (self.t * freq % 1) - 1
            
        elif signal_type == "noise":
            return np.random.normal(0, 1, len(self.t))
            
        elif signal_type == "chirp":
            f0 = kwargs.get('start_freq', 1)
            f1 = kwargs.get('end_freq', 20)
            return np.sin(2 * np.pi * (f0 + (f1 - f0) * self.t / self.duration) * self.t)
            
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
    
    def compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """FFTを計算して周波数成分を取得"""
        # FFT計算
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        
        # 正の周波数のみ取得（実信号なので対称）
        positive_freq_idx = fft_freq >= 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_vals = fft_vals[positive_freq_idx]
        
        # 振幅スペクトル（正規化）
        amplitude = 2 * np.abs(fft_vals) / len(signal)
        amplitude[0] = amplitude[0] / 2  # DC成分
        
        # 位相スペクトル
        phase = np.angle(fft_vals)
        
        return fft_freq, amplitude, phase
    
    def plot_analysis(self, signal: np.ndarray, title: str = "Signal Analysis"):
        """信号と周波数成分を表示"""
        fft_freq, amplitude, phase = self.compute_fft(signal)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)
        
        # 時間領域の信号
        axes[0].plot(self.t, signal, 'b-', linewidth=1.5)
        axes[0].set_xlabel('時間 [秒]')
        axes[0].set_ylabel('振幅')
        axes[0].set_title('時間領域の信号')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, min(1, self.duration))  # 最初の1秒を表示
        
        # 周波数スペクトル（振幅）
        axes[1].stem(fft_freq[:100], amplitude[:100], basefmt='b-')
        axes[1].set_xlabel('周波数 [Hz]')
        axes[1].set_ylabel('振幅')
        axes[1].set_title('周波数スペクトル（振幅）')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 50)  # 0-50Hzを表示
        
        # 主要な周波数成分を表示
        threshold = 0.1 * np.max(amplitude)
        significant_freqs = fft_freq[amplitude > threshold]
        significant_amps = amplitude[amplitude > threshold]
        
        for freq, amp in zip(significant_freqs[:5], significant_amps[:5]):
            if freq > 0:  # DC成分を除く
                axes[1].annotate(f'{freq:.1f}Hz\n{amp:.2f}', 
                               xy=(freq, amp), 
                               xytext=(freq, amp + 0.1),
                               ha='center',
                               fontsize=9,
                               color='red')
        
        # 位相スペクトル
        axes[2].plot(fft_freq[:100], phase[:100], 'g-', linewidth=1.5)
        axes[2].set_xlabel('周波数 [Hz]')
        axes[2].set_ylabel('位相 [rad]')
        axes[2].set_title('位相スペクトル')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, 50)
        axes[2].set_ylim(-np.pi, np.pi)
        
        plt.tight_layout()
        plt.show()
    
    def animate_decomposition(self, signal: np.ndarray, n_components: int = 5):
        """信号を周波数成分に分解してアニメーション表示"""
        fft_freq, amplitude, phase = self.compute_fft(signal)
        
        # 主要な周波数成分を抽出
        sorted_idx = np.argsort(amplitude)[::-1]
        main_freqs = fft_freq[sorted_idx[:n_components]]
        main_amps = amplitude[sorted_idx[:n_components]]
        main_phases = phase[sorted_idx[:n_components]]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # アニメーション用の設定
        lines = []
        for i in range(n_components):
            line, = ax1.plot([], [], label=f'{main_freqs[i]:.1f}Hz')
            lines.append(line)
        
        sum_line, = ax1.plot([], [], 'k-', linewidth=2, label='合成波')
        original_line, = ax1.plot(self.t[:1000], signal[:1000], 'r--', 
                                 alpha=0.5, label='元の信号')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.set_xlabel('時間 [秒]')
        ax1.set_ylabel('振幅')
        ax1.set_title('周波数成分への分解')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 周波数スペクトル
        ax2.stem(main_freqs, main_amps, basefmt='b-')
        ax2.set_xlabel('周波数 [Hz]')
        ax2.set_ylabel('振幅')
        ax2.set_title('主要な周波数成分')
        ax2.grid(True, alpha=0.3)
        
        def animate(frame):
            t_display = self.t[:1000]  # 最初の1秒
            
            # 各周波数成分を描画
            sum_signal = np.zeros_like(t_display)
            for i, (freq, amp, ph) in enumerate(zip(main_freqs[:frame+1], 
                                                   main_amps[:frame+1], 
                                                   main_phases[:frame+1])):
                if freq == 0:  # DC成分
                    component = amp * np.ones_like(t_display)
                else:
                    component = amp * np.sin(2 * np.pi * freq * t_display + ph)
                
                if i <= frame:
                    lines[i].set_data(t_display, component)
                    sum_signal += component
            
            sum_line.set_data(t_display, sum_signal)
            return lines[:frame+1] + [sum_line]
        
        anim = FuncAnimation(fig, animate, frames=n_components, 
                           interval=1000, blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()
        return anim

def main():
    parser = argparse.ArgumentParser(description='フーリエ変換視覚化ツール')
    parser.add_argument('--signal', type=str, default='composite',
                       choices=['sine', 'composite', 'square', 'sawtooth', 'noise', 'chirp'],
                       help='信号の種類')
    parser.add_argument('--animate', action='store_true',
                       help='周波数成分への分解をアニメーション表示')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='信号の継続時間（秒）')
    parser.add_argument('--sample-rate', type=float, default=1000.0,
                       help='サンプリングレート（Hz）')
    
    # 信号タイプ別のパラメータ
    parser.add_argument('--frequency', type=float, default=5.0,
                       help='基本周波数（Hz）')
    parser.add_argument('--frequencies', type=float, nargs='+',
                       default=[5, 10, 15],
                       help='合成波の周波数リスト')
    parser.add_argument('--amplitudes', type=float, nargs='+',
                       default=[1, 0.5, 0.3],
                       help='合成波の振幅リスト')
    
    args = parser.parse_args()
    
    # 視覚化器を初期化
    visualizer = FourierVisualizer(args.duration, args.sample_rate)
    
    # 信号生成のパラメータ
    signal_params = {
        'frequency': args.frequency,
        'frequencies': args.frequencies,
        'amplitudes': args.amplitudes,
    }
    
    # 信号を生成
    print(f"信号タイプ: {args.signal}")
    signal = visualizer.generate_signal(args.signal, **signal_params)
    
    if args.animate and args.signal in ['composite', 'square', 'sawtooth']:
        print("周波数成分への分解をアニメーション表示中...")
        anim = visualizer.animate_decomposition(signal)
    else:
        print("信号解析を表示中...")
        visualizer.plot_analysis(signal, f"{args.signal.capitalize()} Signal Analysis")
    
    print("\n主な使い方:")
    print("1. 単一正弦波: python fourier_cli.py --signal sine --frequency 10")
    print("2. 合成波: python fourier_cli.py --signal composite --frequencies 5 10 20 --amplitudes 1 0.7 0.3")
    print("3. 矩形波: python fourier_cli.py --signal square --frequency 5 --animate")
    print("4. のこぎり波: python fourier_cli.py --signal sawtooth --frequency 3 --animate")
    print("5. チャープ信号: python fourier_cli.py --signal chirp")

if __name__ == "__main__":
    main()
