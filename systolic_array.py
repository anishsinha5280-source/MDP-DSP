"""
systolic_array.py — DSP Core + Architecture
Systolic Array engine for FIR filtering.
Each PE holds one tap, does multiply-accumulate, passes partial sums rightward.
"""

import numpy as np
from scipy.signal import firwin, freqz


# ──────────────────────────────────────────────
# Processing Element
# ──────────────────────────────────────────────

class ProcessingElement:
    """
    One cell in the systolic array.
    Holds a fixed FIR coefficient (tap weight).
    Each cycle: receives a sample, multiplies by its weight,
    adds to incoming partial sum, passes result right.
    """

    def __init__(self, pe_id: int, weight: float):
        self.pe_id = pe_id
        self.weight = weight          # FIR tap coefficient
        self.sample = 0.0             # current sample held in register
        self.partial_sum = 0.0        # accumulated partial sum received from left
        self.output = 0.0             # output sent to next PE (or collected)
        self.active = False           # was this PE active this cycle?
        self.activity_count = 0       # total cycles active

    def process(self, sample: float, partial_sum_in: float) -> float:
        """Multiply-accumulate. Returns partial_sum_out."""
        self.sample = sample
        self.partial_sum = partial_sum_in
        self.output = partial_sum_in + (self.weight * sample)
        self.active = True
        self.activity_count += 1
        return self.output

    def idle(self):
        self.active = False

    def reset(self):
        self.sample = 0.0
        self.partial_sum = 0.0
        self.output = 0.0
        self.active = False
        self.activity_count = 0

    def __repr__(self):
        return (f"PE[{self.pe_id}](w={self.weight:.4f}, "
                f"s={self.sample:.4f}, out={self.output:.4f})")


# ──────────────────────────────────────────────
# Systolic Array Engine
# ──────────────────────────────────────────────

class SystolicArray:
    """
    1-D systolic array for FIR filtering.

    Architecture:
      PE_0 — PE_1 — PE_2 — … — PE_{N-1}
      Data (samples) shifts LEFT→RIGHT through a delay pipeline.
      Partial sums accumulate LEFT→RIGHT each cycle.
      Output is read from the rightmost PE.

    Each clock cycle:
      1. Feed new sample into leftmost stage.
      2. Every PE_i processes (delay_register[i], partial_sum_i).
      3. partial_sum_{i+1} = partial_sum_i + weight_i * delay_register[i]
      4. delay registers shift: delay[i] ← delay[i-1]
    """

    def __init__(self, coefficients: np.ndarray):
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.num_taps = len(coefficients)
        self.pes = [ProcessingElement(i, float(coefficients[i]))
                    for i in range(self.num_taps)]
        # Shift register — one delay element per PE
        self.delay_line = np.zeros(self.num_taps, dtype=float)
        self.cycle_count = 0
        self.cycle_snapshots = []   # for animation

    def reset(self):
        for pe in self.pes:
            pe.reset()
        self.delay_line[:] = 0.0
        self.cycle_count = 0
        self.cycle_snapshots = []

    def clock_cycle(self, new_sample: float) -> float:
        """
        Advance one clock cycle with a new input sample.
        Returns the filter output for this cycle.
        """
        # Shift delay line: oldest sample falls off the right
        self.delay_line = np.roll(self.delay_line, 1)
        self.delay_line[0] = new_sample

        # Propagate partial sums left → right
        partial_sum = 0.0
        active_mask = np.zeros(self.num_taps, dtype=bool)

        for i, pe in enumerate(self.pes):
            sample_at_pe = self.delay_line[i]
            if sample_at_pe != 0.0 or self.cycle_count >= i:
                partial_sum = pe.process(sample_at_pe, partial_sum)
                active_mask[i] = True
            else:
                pe.idle()

        output = partial_sum  # collected from rightmost PE
        self.cycle_count += 1

        # Record snapshot for animation
        self.cycle_snapshots.append({
            "cycle": self.cycle_count,
            "input": new_sample,
            "output": output,
            "delay_line": self.delay_line.copy(),
            "partial_sums": [pe.partial_sum for pe in self.pes],
            "active_mask": active_mask.copy(),
            "weights": self.coefficients.copy(),
        })

        return output

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Run the full signal through the systolic array. Returns filtered output."""
        self.reset()
        output = np.zeros(len(signal))
        for n, sample in enumerate(signal):
            output[n] = self.clock_cycle(float(sample))
        return output

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """Alias for .filter() expected by some runner scripts."""
        return self.filter(signal)

    def get_activity_matrix(self) -> np.ndarray:
        """
        Returns a 2-D matrix of shape (num_cycles, num_taps).
        Entry [t, i] = 1 if PE_i was active at cycle t, else 0.
        """
        if not self.cycle_snapshots:
            return np.zeros((0, self.num_taps))
        return np.array([snap["active_mask"].astype(float)
                         for snap in self.cycle_snapshots])

    def verify_against_numpy(self, signal: np.ndarray) -> dict:
        """Numerical verification: compare output to scipy/numpy convolution."""
        sa_out = self.filter(signal)
        np_out = np.convolve(signal, self.coefficients, mode='full')[:len(signal)]
        max_err = np.max(np.abs(sa_out - np_out))
        rms_err = np.sqrt(np.mean((sa_out - np_out) ** 2))
        return {
            "systolic_output": sa_out,
            "numpy_output": np_out,
            "max_absolute_error": max_err,
            "rms_error": rms_err,
            "pass": max_err < 1e-10,
        }

    def benchmark(self, signal_length: int = 1000, num_runs: int = 5) -> dict:
        """Simple throughput benchmark."""
        import time
        signal = np.random.randn(signal_length)
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.filter(signal)
            times.append(time.perf_counter() - t0)
        mean_t = np.mean(times)
        return {
            "signal_length": signal_length,
            "num_taps": self.num_taps,
            "mean_time_s": mean_t,
            "samples_per_second": signal_length / mean_t,
            "cycles_per_sample": self.num_taps,
        }


# ──────────────────────────────────────────────
# FIR Filter Design
# ──────────────────────────────────────────────

def design_lowpass(num_taps: int, cutoff_hz: float, sample_rate: float,
                   window: str = "hamming") -> np.ndarray:
    """Windowed-sinc low-pass FIR filter."""
    nyq = sample_rate / 2.0
    return firwin(num_taps, cutoff_hz / nyq, window=window)

def design_lowpass_fir(num_taps: int, cutoff_ratio: float, window: str = "hamming") -> np.ndarray:
    """Wrapper for design_lowpass using cutoff ratio (0 to 1)."""
    return firwin(num_taps, cutoff_ratio, window=window)


def design_bandpass(num_taps: int, low_hz: float, high_hz: float,
                    sample_rate: float, window: str = "hamming") -> np.ndarray:
    """Windowed-sinc band-pass FIR filter."""
    nyq = sample_rate / 2.0
    return firwin(num_taps, [low_hz / nyq, high_hz / nyq],
                  pass_zero=False, window=window)


def design_highpass(num_taps: int, cutoff_hz: float, sample_rate: float,
                    window: str = "hamming") -> np.ndarray:
    """Windowed-sinc high-pass FIR filter (num_taps must be odd)."""
    nyq = sample_rate / 2.0
    n = num_taps if num_taps % 2 == 1 else num_taps + 1
    return firwin(n, cutoff_hz / nyq, pass_zero=False, window=window)


def frequency_response(coefficients: np.ndarray, sample_rate: float,
                       n_points: int = 512) -> tuple:
    """Returns (frequencies_hz, magnitude_dB)."""
    w, h = freqz(coefficients, worN=n_points)
    freqs = w * sample_rate / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    return freqs, mag_db


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    if noise_power < 1e-20:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def verify_against_numpy(signal, coefficients):
    """Top-level helper for run_all.py."""
    sa = SystolicArray(coefficients)
    res = sa.verify_against_numpy(signal)
    return {
        "max_error": res["max_absolute_error"],
        "match": res["pass"]
    }


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    fs = 1000.0
    taps = design_lowpass(31, cutoff_hz=100.0, sample_rate=fs)
    sa = SystolicArray(taps)

    t = np.arange(512) / fs
    sig = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    result = sa.verify_against_numpy(sig)

    print(f"Max error vs NumPy: {result['max_absolute_error']:.2e}")
    print(f"PASS: {result['pass']}")
    bm = sa.benchmark(signal_length=2000)
    print(f"Throughput: {bm['samples_per_second']:.0f} samples/sec")
