# Fourier Visualization CLI

時間領域信号を周波数領域表現に変換することで、フーリエ変換を可視化し理解するためのコマンドラインツールです。

<img width="1440" alt="スクリーンショット 2025-06-14 19 20 29" src="https://github.com/user-attachments/assets/adc82824-a909-407d-ae5e-b44f720cf4fb" />

## 機能

- **複数の信号タイプ**: 正弦波、複合信号、方形波、のこぎり波、ノイズ、チャープ信号など、様々な信号タイプを生成・解析
- **視覚的解析**: 時間領域信号と周波数スペクトラム（振幅・位相）を同時表示
- **インタラクティブアニメーション**: 複素信号が周波数成分からどのように構築されるかを示すアニメーション分解表示
- **日本語サポート**: 日本語テキストと適切なフォントレンダリングに対応したインターフェース

## インストール

1. このリポジトリをクローンまたはダウンロード
2. 仮想環境の作成（推奨）:
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
3. 必要な依存関係のインストール:
```bash
pip install numpy matplotlib
```

## 使用方法

### 基本的な使用方法

```bash
python fourier_cli.py [オプション]
```

### 信号タイプ

- `sine`: 単一正弦波
- `composite`: 複数の正弦波を合成
- `square`: 方形波
- `sawtooth`: のこぎり波
- `noise`: ランダムノイズ
- `chirp`: 周波数スイープチャープ信号

### 使用例

1. **10Hzの単一正弦波**:
```bash
python fourier_cli.py --signal sine --frequency 10
```

2. **複数周波数の複合信号**:
```bash
python fourier_cli.py --signal composite --frequencies 5 10 20 --amplitudes 1 0.7 0.3
```

3. **アニメーション付き方形波**:
```bash
python fourier_cli.py --signal square --frequency 5 --animate
```

4. **アニメーション付きのこぎり波**:
```bash
python fourier_cli.py --signal sawtooth --frequency 3 --animate
```

5. **チャープ信号（周波数スイープ）**:
```bash
python fourier_cli.py --signal chirp
```

### コマンドラインオプション

- `--signal`: 信号タイプ（sine, composite, square, sawtooth, noise, chirp）
- `--animate`: アニメーション周波数分解表示
- `--duration`: 信号継続時間（秒）（デフォルト: 2.0）
- `--sample-rate`: サンプリングレート（Hz）（デフォルト: 1000.0）
- `--frequency`: 基本周波数（Hz）（デフォルト: 5.0）
- `--frequencies`: 複合信号用の周波数リスト
- `--amplitudes`: 複合信号用の振幅リスト

### 出力

このツールは3つのプロットを表示します:
1. **時間領域**: 時間軸上の元信号
2. **周波数スペクトラム（振幅）**: 各周波数成分の強度
3. **位相スペクトラム**: 周波数成分の位相関係

`--animate`フラグを使用すると、周波数成分がどのように組み合わされて元の信号を形成するかを示すアニメーション可視化が表示されます。

## 必要要件

- Python 3.6+
- NumPy
- Matplotlib

## プラットフォームサポート

- **macOS**: Hiragino Sansフォントを使用
- **Windows**: MS Gothicフォントを使用
- **Linux**: Noto Sans CJK JPフォントを使用

ツールは自動的にプラットフォームを検出し、適切な日本語フォントレンダリングを設定します。
