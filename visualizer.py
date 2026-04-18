"""
visualizer.py — Visualization Dashboard
Generates all static plots for the Systolic Array FIR demo.
Exports: signal_plot.png, tap_coefficients.png, freq_response.png, pe_heatmap.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

from systolic_array import SystolicArray, design_lowpass, design_bandpass, frequency_response
from dsp_signals import demo_lowpass_scenario

# ──────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────

DARK_BG     = "#0d1117"
PANEL_BG    = "#161b22"
GRID_COLOR  = "#21262d"
ACCENT_BLUE = "#58a6ff"
ACCENT_CYAN = "#39d0d8"
ACCENT_ORANGE = "#f0883e"
ACCENT_GREEN  = "#3fb950"
ACCENT_PURPLE = "#bc8cff"
TEXT_MAIN   = "#e6edf3"
TEXT_DIM    = "#7d8590"

def _apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT_MAIN)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.8)

def _save(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────
# 1. Input vs Filtered Output
# ──────────────────────────────────────────────

def plot_signal(time: np.ndarray, input_signal: np.ndarray,
                filtered_signal: np.ndarray, sample_rate: float,
                title: str = "FIR Filter: Input vs Output",
                out_path: str = "signal_plot.png"):
    """Two-panel time-domain plot: raw input (top) and filtered output (bottom)."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, color=TEXT_MAIN, fontsize=14, fontweight='bold', y=1.01)
    _apply_dark_style(fig, [ax1, ax2])

    # Show only first 300 samples for clarity
    n_show = min(300, len(time))
    t_ms = time[:n_show] * 1000  # ms

    ax1.plot(t_ms, input_signal[:n_show], color=ACCENT_ORANGE,
             linewidth=1.2, alpha=0.9, label="Input")
    ax1.set_ylabel("Amplitude", color=TEXT_DIM)
    ax1.set_title("Input Signal", color=TEXT_MAIN, fontsize=11)
    ax1.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_MAIN, fontsize=9)

    ax2.plot(t_ms, filtered_signal[:n_show], color=ACCENT_CYAN,
             linewidth=1.4, alpha=0.95, label="Filtered Output")
    ax2.set_xlabel("Time (ms)", color=TEXT_DIM)
    ax2.set_ylabel("Amplitude", color=TEXT_DIM)
    ax2.set_title("Systolic Array Filtered Output", color=TEXT_MAIN, fontsize=11)
    ax2.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_MAIN, fontsize=9)

    plt.tight_layout()
    _save(fig, out_path)


# ──────────────────────────────────────────────
# 2. FIR Tap Coefficients (stem plot)
# ──────────────────────────────────────────────

def plot_tap_coefficients(coefficients: np.ndarray,
                          title: str = "FIR Tap Coefficients",
                          out_path: str = "tap_coefficients.png"):
    """Stem plot of filter tap weights."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark_style(fig, [ax])
    fig.patch.set_facecolor(DARK_BG)

    n = np.arange(len(coefficients))
    markerline, stemlines, baseline = ax.stem(n, coefficients, linefmt='-',
                                              markerfmt='o', basefmt='-')

    plt.setp(stemlines, color=ACCENT_BLUE, linewidth=1.5, alpha=0.8)
    plt.setp(markerline, color=ACCENT_CYAN, markersize=5, zorder=5)
    plt.setp(baseline, color=GRID_COLOR, linewidth=1.0)

    ax.set_xlabel("Tap Index (PE Number)", color=TEXT_DIM)
    ax.set_ylabel("Coefficient Value", color=TEXT_DIM)
    ax.set_title(title, color=TEXT_MAIN, fontsize=13, fontweight='bold')

    # Annotate peak
    peak_idx = np.argmax(np.abs(coefficients))
    ax.annotate(f"Peak: {coefficients[peak_idx]:.4f}",
                xy=(peak_idx, coefficients[peak_idx]),
                xytext=(peak_idx + len(n) * 0.08, coefficients[peak_idx] * 0.85),
                color=ACCENT_ORANGE, fontsize=8,
                arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE, lw=1.2))

    plt.tight_layout()
    _save(fig, out_path)


# ──────────────────────────────────────────────
# 3. Frequency Response
# ──────────────────────────────────────────────

def plot_frequency_response(coefficients: np.ndarray, sample_rate: float,
                             title: str = "FIR Frequency Response",
                             passband_hz: tuple = None,
                             out_path: str = "freq_response.png"):
    """Magnitude response in dB vs Hz with passband annotation."""
    freqs, mag_db = frequency_response(coefficients, sample_rate, n_points=1024)

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, [ax])
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(freqs, mag_db, color=ACCENT_GREEN, linewidth=1.8, label="Magnitude (dB)")
    ax.axhline(-3, color=ACCENT_ORANGE, linestyle='--', linewidth=1,
               alpha=0.7, label="-3 dB cutoff")
    ax.axhline(-60, color=TEXT_DIM, linestyle=':', linewidth=0.8, alpha=0.5)

    if passband_hz:
        ax.axvspan(passband_hz[0], passband_hz[1], alpha=0.08,
                   color=ACCENT_GREEN, label="Passband")

    ax.set_xlim(0, sample_rate / 2)
    ax.set_ylim(-90, 5)
    ax.set_xlabel("Frequency (Hz)", color=TEXT_DIM)
    ax.set_ylabel("Magnitude (dB)", color=TEXT_DIM)
    ax.set_title(title, color=TEXT_MAIN, fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_MAIN, fontsize=9)

    plt.tight_layout()
    _save(fig, out_path)


# ──────────────────────────────────────────────
# 4. PE Activity Heatmap
# ──────────────────────────────────────────────

def plot_pe_heatmap(activity_matrix: np.ndarray,
                   title: str = "Processing Element Activity Heatmap",
                   out_path: str = "pe_heatmap.png",
                   max_cycles: int = 80):
    """
    Heatmap: rows = clock cycles, columns = PE index.
    Bright = active, dark = idle.
    """
    mat = activity_matrix[:max_cycles]  # trim for readability

    cmap = LinearSegmentedColormap.from_list(
        "pe_activity",
        [DARK_BG, "#0a2a4a", ACCENT_BLUE, ACCENT_CYAN, "#ffffff"],
        N=256
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_dark_style(fig, [ax])
    fig.patch.set_facecolor(DARK_BG)

    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   interpolation='nearest', origin='upper')

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
    cbar.set_label("Active", color=TEXT_DIM, fontsize=9)

    ax.set_xlabel("Processing Element (PE / Tap) Index", color=TEXT_DIM)
    ax.set_ylabel("Clock Cycle", color=TEXT_DIM)
    ax.set_title(title, color=TEXT_MAIN, fontsize=13, fontweight='bold')

    # Draw grid lines between cells
    num_cycles, num_pes = mat.shape
    for x in range(num_pes + 1):
        ax.axvline(x - 0.5, color=DARK_BG, linewidth=0.4)
    for y in range(0, num_cycles, 10):
        ax.axhline(y - 0.5, color=DARK_BG, linewidth=0.4)

    # Annotate wavefront diagonal
    diag_x = np.arange(min(num_pes, num_cycles))
    ax.plot(diag_x, diag_x, color=ACCENT_ORANGE, linewidth=1.5,
            linestyle='--', alpha=0.7, label="Wavefront")
    ax.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_MAIN, fontsize=9)

    plt.tight_layout()
    _save(fig, out_path)


# ──────────────────────────────────────────────
# 5. Combined overview figure
# ──────────────────────────────────────────────

def plot_overview(time, input_signal, filtered_signal, coefficients,
                  activity_matrix, sample_rate, scenario_name,
                  out_path="overview.png"):
    """4-panel overview on one figure."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"Systolic Array FIR Filter — {scenario_name}",
                 color=TEXT_MAIN, fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.96, top=0.93, bottom=0.08)

    ax_sig = fig.add_subplot(gs[0, 0])
    ax_tap = fig.add_subplot(gs[0, 1])
    ax_fr  = fig.add_subplot(gs[1, 0])
    ax_hm  = fig.add_subplot(gs[1, 1])
    _apply_dark_style(fig, [ax_sig, ax_tap, ax_fr, ax_hm])

    n_show = min(300, len(time))
    t_ms = time[:n_show] * 1000

    # Signal
    ax_sig.plot(t_ms, input_signal[:n_show], color=ACCENT_ORANGE,
                linewidth=1.0, alpha=0.7, label="Input")
    ax_sig.plot(t_ms, filtered_signal[:n_show], color=ACCENT_CYAN,
                linewidth=1.4, label="Filtered")
    ax_sig.set_title("Time Domain", color=TEXT_MAIN, fontsize=10)
    ax_sig.set_xlabel("Time (ms)", color=TEXT_DIM, fontsize=8)
    ax_sig.set_ylabel("Amplitude", color=TEXT_DIM, fontsize=8)
    ax_sig.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_MAIN, fontsize=8)

    # Taps
    n_c = np.arange(len(coefficients))
    markerline, stemlines, baseline = ax_tap.stem(n_c, coefficients,
                                                  linefmt='-', markerfmt='o', basefmt='-')
    plt.setp(stemlines, color=ACCENT_BLUE, linewidth=1.2)
    plt.setp(markerline, color=ACCENT_CYAN, markersize=4)
    plt.setp(baseline, color=GRID_COLOR)
    ax_tap.set_title("Tap Coefficients", color=TEXT_MAIN, fontsize=10)
    ax_tap.set_xlabel("PE Index", color=TEXT_DIM, fontsize=8)

    # Frequency response
    freqs, mag_db = frequency_response(coefficients, sample_rate)
    ax_fr.plot(freqs, mag_db, color=ACCENT_GREEN, linewidth=1.6)
    ax_fr.axhline(-3, color=ACCENT_ORANGE, linestyle='--', linewidth=1, alpha=0.8)
    ax_fr.set_ylim(-90, 5)
    ax_fr.set_title("Frequency Response", color=TEXT_MAIN, fontsize=10)
    ax_fr.set_xlabel("Frequency (Hz)", color=TEXT_DIM, fontsize=8)
    ax_fr.set_ylabel("dB", color=TEXT_DIM, fontsize=8)

    # Heatmap
    mat = activity_matrix[:60]
    cmap = LinearSegmentedColormap.from_list(
        "pa", [DARK_BG, "#0a2a4a", ACCENT_BLUE, ACCENT_CYAN], N=256)
    ax_hm.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                 interpolation='nearest', origin='upper')
    ax_hm.set_title("PE Activity Heatmap", color=TEXT_MAIN, fontsize=10)
    ax_hm.set_xlabel("PE Index", color=TEXT_DIM, fontsize=8)
    ax_hm.set_ylabel("Clock Cycle", color=TEXT_DIM, fontsize=8)

    _save(fig, out_path, dpi=150)


# ──────────────────────────────────────────────
# Main — generate all plots
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(output_dir, exist_ok=True)

    print("Building scenario...")
    scenario = demo_lowpass_scenario(sample_rate=1000.0)

    coefficients = design_lowpass(
        num_taps=scenario["num_taps"],
        cutoff_hz=scenario["cutoff_hz"],
        sample_rate=scenario["sample_rate"],
    )

    sa = SystolicArray(coefficients)
    filtered = sa.filter(scenario["signal"])
    activity = sa.get_activity_matrix()

    t   = scenario["time"]
    sig = scenario["signal"]
    fs  = scenario["sample_rate"]

    print("Plotting...")
    plot_signal(t, sig, filtered, fs,
                title=f"FIR Filter — {scenario['name']}",
                out_path=os.path.join(output_dir, "signal_plot.png"))

    plot_tap_coefficients(coefficients,
                          title=f"FIR Coefficients ({scenario['num_taps']} taps)",
                          out_path=os.path.join(output_dir, "tap_coefficients.png"))

    plot_frequency_response(coefficients, fs,
                             title="Low-Pass Frequency Response",
                             passband_hz=(0, scenario["cutoff_hz"]),
                             out_path=os.path.join(output_dir, "freq_response.png"))

    plot_pe_heatmap(activity,
                    out_path=os.path.join(output_dir, "pe_heatmap.png"))

    plot_overview(t, sig, filtered, coefficients, activity, fs,
                  scenario_name=scenario["name"],
                  out_path=os.path.join(output_dir, "overview.png"))

    # Numerical verification
    result = sa.verify_against_numpy(sig)
    print(f"\nVerification vs NumPy:")
    print(f"  Max absolute error: {result['max_absolute_error']:.2e}")
    print(f"  PASS: {result['pass']}")
    print("\nAll plots saved.")
