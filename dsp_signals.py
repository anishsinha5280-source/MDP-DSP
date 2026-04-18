"""
dsp_signals.py — Signal Generators for Systolic Array Demo
Provides multi-tone, chirp, noisy sinusoid, and ECG-like waveforms.
"""

import numpy as np


# ──────────────────────────────────────────────
# Core generators
# ──────────────────────────────────────────────

def multi_tone(freqs_hz: list, amplitudes: list, duration_s: float,
               sample_rate: float, phase_offsets: list = None) -> tuple:
    """
    Sum of sinusoids.
    Returns (time_array, signal_array).
    """
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    if phase_offsets is None:
        phase_offsets = [0.0] * len(freqs_hz)
    sig = np.zeros_like(t)
    for f, a, phi in zip(freqs_hz, amplitudes, phase_offsets):
        sig += a * np.sin(2 * np.pi * f * t + phi)
    return t, sig


def chirp_signal(f_start: float, f_end: float, duration_s: float,
                 sample_rate: float, amplitude: float = 1.0) -> tuple:
    """
    Linear frequency sweep (chirp).
    Returns (time_array, signal_array).
    """
    from scipy.signal import chirp
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    sig = amplitude * chirp(t, f0=f_start, f1=f_end, t1=duration_s, method='linear')
    return t, sig


def noisy_sinusoid(freq_hz: float, snr_db: float, duration_s: float,
                   sample_rate: float, amplitude: float = 1.0,
                   seed: int = 42) -> tuple:
    """
    Single sinusoid + white Gaussian noise at specified SNR.
    Returns (time_array, clean_signal, noisy_signal).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    clean = amplitude * np.sin(2 * np.pi * freq_hz * t)
    signal_power = np.mean(clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(t))
    return t, clean, clean + noise


def ecg_like(duration_s: float, sample_rate: float, heart_rate_bpm: float = 72,
             noise_level: float = 0.05, seed: int = 7) -> tuple:
    """
    Synthetic ECG-like waveform (P-QRS-T morphology approximation).
    Returns (time_array, signal_array).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    sig = np.zeros(n)

    beat_period = 60.0 / heart_rate_bpm  # seconds per beat
    beat_samples = int(beat_period * sample_rate)

    def gaussian(x, mu, sigma, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    beat_t = np.linspace(0, beat_period, beat_samples)
    # P wave
    beat = gaussian(beat_t, 0.10, 0.02, 0.15)
    # Q dip
    beat += gaussian(beat_t, 0.20, 0.008, -0.10)
    # R peak
    beat += gaussian(beat_t, 0.22, 0.010, 1.00)
    # S dip
    beat += gaussian(beat_t, 0.24, 0.008, -0.25)
    # T wave
    beat += gaussian(beat_t, 0.38, 0.035, 0.30)

    # Tile beats across signal
    for start in range(0, n, beat_samples):
        end = min(start + beat_samples, n)
        sig[start:end] += beat[:end - start]

    sig += rng.normal(0, noise_level, n)
    return t, sig


def step_response_input(duration_s: float, sample_rate: float,
                        step_time_s: float = 0.1) -> tuple:
    """Unit step for observing filter transient response."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    sig = np.zeros(n)
    step_idx = int(step_time_s * sample_rate)
    sig[step_idx:] = 1.0
    return t, sig


def impulse_input(duration_s: float, sample_rate: float,
                  impulse_time_s: float = 0.0) -> tuple:
    """Unit impulse — reveals the filter's impulse response."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    sig = np.zeros(n)
    idx = int(impulse_time_s * sample_rate)
    if 0 <= idx < n:
        sig[idx] = 1.0
    return t, sig


# ──────────────────────────────────────────────
# Preset demo configurations
# ──────────────────────────────────────────────

def demo_lowpass_scenario(sample_rate: float = 1000.0) -> dict:
    """
    Classic low-pass demo: 50 Hz tone + 300 Hz interference.
    Filter should keep 50 Hz and kill 300 Hz.
    """
    duration = 0.5
    t, signal = multi_tone(
        freqs_hz=[50, 300],
        amplitudes=[1.0, 0.8],
        duration_s=duration,
        sample_rate=sample_rate,
    )
    return {
        "name": "Low-Pass: 50Hz signal + 300Hz noise",
        "sample_rate": sample_rate,
        "duration": duration,
        "time": t,
        "signal": signal,
        "filter_type": "lowpass",
        "cutoff_hz": 150.0,
        "num_taps": 31,
        "target_freqs": [50],
        "noise_freqs": [300],
    }


def demo_bandpass_scenario(sample_rate: float = 1000.0) -> dict:
    """
    Band-pass demo: keep 200–350 Hz band from a 3-tone mix.
    """
    duration = 0.5
    t, signal = multi_tone(
        freqs_hz=[50, 250, 400],
        amplitudes=[1.0, 1.0, 1.0],
        duration_s=duration,
        sample_rate=sample_rate,
    )
    return {
        "name": "Band-Pass: isolate 200–350Hz",
        "sample_rate": sample_rate,
        "duration": duration,
        "time": t,
        "signal": signal,
        "filter_type": "bandpass",
        "low_hz": 180.0,
        "high_hz": 320.0,
        "num_taps": 63,
        "target_freqs": [250],
        "noise_freqs": [50, 400],
    }


def demo_ecg_scenario(sample_rate: float = 500.0) -> dict:
    """
    ECG denoising demo: 60 Hz power-line interference removal.
    """
    duration = 3.0
    t, ecg = ecg_like(duration, sample_rate, heart_rate_bpm=75, noise_level=0.03)
    # Add 60 Hz power-line noise
    power_noise = 0.3 * np.sin(2 * np.pi * 60 * t)
    noisy_ecg = ecg + power_noise
    return {
        "name": "ECG: 60Hz power-line removal",
        "sample_rate": sample_rate,
        "duration": duration,
        "time": t,
        "signal": noisy_ecg,
        "clean_signal": ecg,
        "filter_type": "bandpass",
        "low_hz": 0.5,
        "high_hz": 45.0,
        "num_taps": 127,
    }


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    scenario = demo_lowpass_scenario()
    print(f"Demo: {scenario['name']}")
    print(f"Signal shape: {scenario['signal'].shape}, "
          f"Sample rate: {scenario['sample_rate']} Hz")

    t2, clean, noisy = noisy_sinusoid(100, snr_db=10, duration_s=0.1,
                                      sample_rate=1000)
    from systolic_array import compute_snr
    print(f"SNR check: {compute_snr(clean, noisy):.1f} dB (target ~10 dB)")
