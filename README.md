# Fourier Visualizer

A command-line tool for visualizing and understanding the Fourier transform by converting time-domain signals to frequency-domain representations.

## Features

- **Multiple Signal Types**: Generate and analyze various signal types including sine waves, composite signals, square waves, sawtooth waves, noise, and chirp signals
- **Visual Analysis**: Display time-domain signals alongside their frequency spectrum (amplitude and phase)
- **Interactive Animation**: Animated decomposition showing how complex signals are built from frequency components
- **Japanese Language Support**: Interface with Japanese text and proper font rendering

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Basic Usage

```bash
python fourier_cli.py [options]
```

### Signal Types

- `sine`: Single sine wave
- `composite`: Multiple sine waves combined
- `square`: Square wave
- `sawtooth`: Sawtooth wave  
- `noise`: Random noise
- `chirp`: Frequency-swept chirp signal

### Examples

1. **Single sine wave at 10Hz**:
```bash
python fourier_cli.py --signal sine --frequency 10
```

2. **Composite signal with multiple frequencies**:
```bash
python fourier_cli.py --signal composite --frequencies 5 10 20 --amplitudes 1 0.7 0.3
```

3. **Square wave with animation**:
```bash
python fourier_cli.py --signal square --frequency 5 --animate
```

4. **Sawtooth wave with animation**:
```bash
python fourier_cli.py --signal sawtooth --frequency 3 --animate
```

5. **Chirp signal (frequency sweep)**:
```bash
python fourier_cli.py --signal chirp
```

### Command Line Options

- `--signal`: Signal type (sine, composite, square, sawtooth, noise, chirp)
- `--animate`: Show animated frequency decomposition
- `--duration`: Signal duration in seconds (default: 2.0)
- `--sample-rate`: Sampling rate in Hz (default: 1000.0)
- `--frequency`: Base frequency in Hz (default: 5.0)
- `--frequencies`: List of frequencies for composite signals
- `--amplitudes`: List of amplitudes for composite signals

### Output

The tool displays three plots:
1. **Time Domain**: The original signal over time
2. **Frequency Spectrum (Amplitude)**: Shows the strength of each frequency component
3. **Phase Spectrum**: Shows the phase relationship of frequency components

With the `--animate` flag, you'll see an animated visualization showing how frequency components combine to form the original signal.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Platform Support

- **macOS**: Uses Hiragino Sans font
- **Windows**: Uses MS Gothic font  
- **Linux**: Uses Noto Sans CJK JP font

The tool automatically detects your platform and configures appropriate Japanese font rendering.