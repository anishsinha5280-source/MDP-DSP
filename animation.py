"""
animation.py — Systolic Array PE Animation
Animates PE cells cycling through active/idle states.
Exports: systolic_animation.gif
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patheffects as pe
import os
import sys

from systolic_array import SystolicArray, design_lowpass
from dsp_signals import demo_lowpass_scenario

# ──────────────────────────────────────────────
# Style constants
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

PE_IDLE_FACE   = "#1c2a3a"
PE_IDLE_EDGE   = "#2d4a6a"
PE_ACTIVE_FACE = "#0a3a5a"
PE_ACTIVE_EDGE = ACCENT_CYAN
PE_ACTIVE_GLOW = "#39d0d8"

ARROW_COLOR = ACCENT_ORANGE
DATA_COLOR  = ACCENT_ORANGE
SUM_COLOR   = ACCENT_PURPLE


# ──────────────────────────────────────────────
# Helper: draw a single PE box
# ──────────────────────────────────────────────

def _draw_pe_box(ax, x, y, width, height, pe_idx, weight,
                 sample_val, partial_sum, active, alpha=1.0):
    """Draw one PE as a styled rectangle with internal labels."""
    face = PE_ACTIVE_FACE if active else PE_IDLE_FACE
    edge = PE_ACTIVE_EDGE if active else PE_IDLE_EDGE
    lw   = 2.2 if active else 0.8

    rect = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02",
        facecolor=face, edgecolor=edge, linewidth=lw,
        alpha=alpha, zorder=3
    )
    ax.add_patch(rect)

    # Glow effect when active
    if active:
        glow = mpatches.FancyBboxPatch(
            (x - width/2 - 0.02, y - height/2 - 0.02),
            width + 0.04, height + 0.04,
            boxstyle="round,pad=0.04",
            facecolor="none", edgecolor=PE_ACTIVE_GLOW,
            linewidth=4, alpha=0.25, zorder=2
        )
        ax.add_patch(glow)

    # PE index label
    ax.text(x, y + height*0.26, f"PE{pe_idx}",
            ha='center', va='center', fontsize=7.5, fontweight='bold',
            color=TEXT_MAIN if active else TEXT_DIM, zorder=4)

    # Weight label
    ax.text(x, y + 0.00, f"w={weight:.3f}",
            ha='center', va='center', fontsize=6.5,
            color=ACCENT_BLUE if active else TEXT_DIM,
            fontfamily='monospace', zorder=4)

    # Sample value
    sample_color = DATA_COLOR if active else TEXT_DIM
    ax.text(x, y - height*0.26, f"x={sample_val:.3f}",
            ha='center', va='center', fontsize=6,
            color=sample_color, fontfamily='monospace', zorder=4)


# ──────────────────────────────────────────────
# Main animation builder
# ──────────────────────────────────────────────

def build_animation(cycle_snapshots: list, coefficients: np.ndarray,
                    num_pe_display: int = 8,
                    num_cycles_animate: int = 40,
                    fps: int = 4,
                    out_path: str = "systolic_animation.gif"):
    """
    Build and export the systolic array animation as a GIF.

    Parameters
    ----------
    cycle_snapshots : list of dicts from SystolicArray.cycle_snapshots
    coefficients    : FIR tap coefficients
    num_pe_display  : how many PEs to show (clip for readability)
    num_cycles_animate : number of animation frames
    fps             : frames per second
    out_path        : output file path
    """
    n_pes  = min(num_pe_display, len(coefficients))
    n_cyc  = min(num_cycles_animate, len(cycle_snapshots))
    snaps  = cycle_snapshots[:n_cyc]

    # Layout constants
    PE_W, PE_H = 0.85, 0.70
    PE_SPACING = 1.05
    ROW_Y      = 2.5
    ARROW_Y    = ROW_Y - PE_H / 2 - 0.25

    fig_w = n_pes * PE_SPACING + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-0.8, n_pes * PE_SPACING + 0.5)
    ax.set_ylim(0.2, 4.8)
    ax.axis('off')

    # Static title
    title_txt = ax.text(
        (n_pes * PE_SPACING - 0.3) / 2, 4.5,
        "Systolic Array — FIR Filter Execution",
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=TEXT_MAIN
    )

    # Clock cycle counter
    cycle_txt = ax.text(
        0.01, 0.98, "Cycle 0",
        ha='left', va='top', transform=ax.transAxes,
        fontsize=10, color=ACCENT_CYAN, fontfamily='monospace'
    )

    # Input / output labels
    input_txt = ax.text(
        -0.55, ROW_Y, "IN →",
        ha='center', va='center', fontsize=9,
        color=ACCENT_ORANGE, fontweight='bold'
    )

    output_label = ax.text(
        n_pes * PE_SPACING + 0.2, ROW_Y, "→ OUT",
        ha='left', va='center', fontsize=9,
        color=ACCENT_PURPLE, fontweight='bold'
    )

    output_val_txt = ax.text(
        n_pes * PE_SPACING + 0.2, ROW_Y - 0.35, "",
        ha='left', va='center', fontsize=7.5,
        color=ACCENT_PURPLE, fontfamily='monospace'
    )

    # Partial sum flow arrows (static decoration)
    for i in range(n_pes - 1):
        x_mid = (i + 1) * PE_SPACING - PE_SPACING / 2 + PE_W / 2 + 0.02
        ax.annotate("",
                    xy=((i + 1) * PE_SPACING - PE_W / 2 + 0.02, ROW_Y - 0.05),
                    xytext=(i * PE_SPACING + PE_W / 2 + 0.02, ROW_Y - 0.05),
                    arrowprops=dict(arrowstyle='->', color=SUM_COLOR,
                                   lw=1.0, alpha=0.4),
                    zorder=1)

    # Data flow label
    ax.text((n_pes * PE_SPACING) / 2, 1.3,
            "← Data shifts right each clock cycle   |   Partial sums accumulate →",
            ha='center', va='center', fontsize=7.5,
            color=TEXT_DIM, style='italic')

    # Waveform mini-plot axes (bottom strip)
    ax_wave = fig.add_axes([0.07, 0.06, 0.88, 0.16])
    ax_wave.set_facecolor(PANEL_BG)
    ax_wave.tick_params(colors=TEXT_DIM, labelsize=7)
    for spine in ax_wave.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax_wave.set_xlim(0, n_cyc)
    all_outputs = [s['output'] for s in cycle_snapshots[:n_cyc]]
    y_range = max(abs(min(all_outputs)), abs(max(all_outputs))) + 0.1
    ax_wave.set_ylim(-y_range, y_range)
    ax_wave.axhline(0, color=GRID_COLOR, linewidth=0.5)
    ax_wave.set_ylabel("Output", color=TEXT_DIM, fontsize=7)
    waveform_line, = ax_wave.plot([], [], color=ACCENT_CYAN, linewidth=1.2)
    vline = ax_wave.axvline(0, color=ACCENT_ORANGE, linewidth=1.0, alpha=0.8)
    wave_xs, wave_ys = [], []

    # PE patch handles (rebuilt each frame)
    pe_patches = []
    pe_texts   = []

    def init():
        waveform_line.set_data([], [])
        return []

    def animate(frame_idx):
        snap = snaps[frame_idx]
        cycle_num  = snap['cycle']
        active_mask = snap['active_mask'][:n_pes]
        delay_line  = snap['delay_line'][:n_pes]
        partial_sums = snap['partial_sums'][:n_pes]
        output_val  = snap['output']
        input_val   = snap['input']

        # Remove old PE patches
        for p in pe_patches:
            p.remove()
        pe_patches.clear()
        for t in pe_texts:
            t.remove()
        pe_texts.clear()

        # Draw PEs
        for i in range(n_pes):
            x = i * PE_SPACING
            active = bool(active_mask[i]) if i < len(active_mask) else False
            sample  = float(delay_line[i]) if i < len(delay_line) else 0.0
            psum    = float(partial_sums[i]) if i < len(partial_sums) else 0.0
            weight  = float(coefficients[i])

            face = PE_ACTIVE_FACE if active else PE_IDLE_FACE
            edge = PE_ACTIVE_EDGE if active else PE_IDLE_EDGE
            lw   = 2.2 if active else 0.8

            rect = mpatches.FancyBboxPatch(
                (x - PE_W/2, ROW_Y - PE_H/2), PE_W, PE_H,
                boxstyle="round,pad=0.03",
                facecolor=face, edgecolor=edge, linewidth=lw,
                alpha=1.0, zorder=3
            )
            ax.add_patch(rect)
            pe_patches.append(rect)

            if active:
                glow = mpatches.FancyBboxPatch(
                    (x - PE_W/2 - 0.03, ROW_Y - PE_H/2 - 0.03),
                    PE_W + 0.06, PE_H + 0.06,
                    boxstyle="round,pad=0.05",
                    facecolor="none", edgecolor=PE_ACTIVE_GLOW,
                    linewidth=4, alpha=0.3, zorder=2
                )
                ax.add_patch(glow)
                pe_patches.append(glow)

            # Labels
            t1 = ax.text(x, ROW_Y + 0.19, f"PE{i}",
                         ha='center', va='center', fontsize=7.5, fontweight='bold',
                         color=TEXT_MAIN if active else TEXT_DIM, zorder=4)
            t2 = ax.text(x, ROW_Y, f"w={weight:.3f}",
                         ha='center', va='center', fontsize=6,
                         color=ACCENT_BLUE if active else TEXT_DIM,
                         fontfamily='monospace', zorder=4)
            t3 = ax.text(x, ROW_Y - 0.19, f"x={sample:.2f}",
                         ha='center', va='center', fontsize=5.8,
                         color=DATA_COLOR if active else TEXT_DIM,
                         fontfamily='monospace', zorder=4)
            pe_texts.extend([t1, t2, t3])

            # Active indicator dot
            if active:
                dot = ax.plot(x, ROW_Y + PE_H/2 - 0.08, 'o',
                              color=ACCENT_CYAN, markersize=4, zorder=5)
                pe_patches.extend(dot)

        # Input arrow glow
        input_txt.set_text(f"IN\n{input_val:.3f}")

        # Output value
        output_val_txt.set_text(f"{output_val:.4f}")

        # Update cycle counter
        cycle_txt.set_text(f"Cycle {cycle_num:3d}")

        # Waveform
        wave_xs.append(frame_idx)
        wave_ys.append(output_val)
        waveform_line.set_data(wave_xs, wave_ys)
        vline.set_xdata([frame_idx, frame_idx])

        return pe_patches + pe_texts + [waveform_line, cycle_txt, input_txt, output_val_txt]

    anim = FuncAnimation(fig, animate, frames=n_cyc, init_func=init,
                          interval=1000 // fps, blit=False, repeat=True)

    writer = PillowWriter(fps=fps, metadata={"title": "Systolic Array FIR"})
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(output_dir, exist_ok=True)

    print("Building scenario and running systolic array...")
    scenario = demo_lowpass_scenario(sample_rate=1000.0)
    coefficients = design_lowpass(
        num_taps=scenario["num_taps"],
        cutoff_hz=scenario["cutoff_hz"],
        sample_rate=scenario["sample_rate"],
    )

    sa = SystolicArray(coefficients)
    sa.filter(scenario["signal"])

    print("Building animation (this takes ~30s)...")
    build_animation(
        cycle_snapshots=sa.cycle_snapshots,
        coefficients=coefficients,
        num_pe_display=8,
        num_cycles_animate=40,
        fps=4,
        out_path=os.path.join(output_dir, "systolic_animation.gif"),
    )
    print("Animation complete.")
