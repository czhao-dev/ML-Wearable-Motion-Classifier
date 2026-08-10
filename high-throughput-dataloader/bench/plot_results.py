"""Generates the benchmark plots embedded in the project README.

Data points below are hardcoded from specific runs of the bench/ tools on
one development machine (see the README's "Reproducing these numbers"
section for the exact commands) -- this script only renders them, it does
not re-run the benchmarks. Re-run the underlying tools and update the
arrays below to regenerate with fresh numbers.

Usage: python3 bench/plot_results.py
Requires: matplotlib (see bench/requirements.txt)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- palette (validated: see project session notes on the dataviz skill) ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # categorical slot 1 / sequential hue
ORANGE = "#eb6834"  # categorical slot 2

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.edgecolor": AXIS,
        "font.family": "sans-serif",
        "font.size": 11,
    }
)


def _style_axes(ax, ylabel: str):
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def plot_throughput_vs_workers():
    workers = [1, 2, 4, 8]
    as_ready = [25.35, 53.94, 96.22, 161.12]
    ordered = [24.58, 53.80, 94.58, 163.23]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, as_ready, color=BLUE, linewidth=2, marker="o", markersize=8, label="as-ready")
    ax.plot(workers, ordered, color=ORANGE, linewidth=2, marker="o", markersize=8, label="ordered")
    ax.set_xticks(workers)
    ax.set_xlabel("worker count", color=INK_SECONDARY)
    _style_axes(ax, "samples/sec (thousands)")
    ax.set_title("Throughput vs. worker count", color=INK_PRIMARY, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/plots/throughput_vs_workers.png", dpi=150)
    plt.close(fig)


def plot_starvation_vs_workers():
    workers = [1, 2, 4, 8]
    idle_pct = [94.48, 89.33, 52.77, 53.01]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, idle_pct, color=BLUE, linewidth=2, marker="o", markersize=8)
    ax.set_xticks(workers)
    ax.set_xlabel("worker count", color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    _style_axes(ax, "consumer idle time (%)")
    ax.set_title("Starvation vs. worker count", color=INK_PRIMARY, loc="left")
    fig.tight_layout()
    fig.savefig("docs/plots/starvation_vs_workers.png", dpi=150)
    plt.close(fig)


def plot_starvation_vs_depth():
    depth = [1, 4, 16, 64]
    idle_pct = [90.74, 90.03, 90.26, 90.64]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depth, idle_pct, color=BLUE, linewidth=2, marker="o", markersize=8)
    ax.set_xscale("log", base=4)
    ax.set_xticks(depth)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("prefetch depth (samples)", color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    _style_axes(ax, "consumer idle time (%)")
    ax.set_title("Starvation vs. prefetch depth", color=INK_PRIMARY, loc="left")
    fig.tight_layout()
    fig.savefig("docs/plots/starvation_vs_depth.png", dpi=150)
    plt.close(fig)


def plot_cost_sweep():
    cost = [1000, 5000, 20000, 100000, 300000, 1000000]
    throughput = [297.4, 208.4, 96.5, 25.0, 8.79, 2.74]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cost, throughput, color=BLUE, linewidth=2, marker="o", markersize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("generator cost knob (busy-loop iterations, log scale)", color=INK_SECONDARY)
    _style_axes(ax, "samples/sec (thousands)")
    ax.set_title("Throughput vs. augmentation cost", color=INK_PRIMARY, loc="left")
    fig.tight_layout()
    fig.savefig("docs/plots/cost_sweep.png", dpi=150)
    plt.close(fig)


def plot_memory_vs_depth():
    depth = [1, 2, 4, 8, 16, 32, 64]
    peak_rss_mb = [7.89, 12.42, 12.31, 20.83, 32.86, 61.97, 94.13]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depth, peak_rss_mb, color=BLUE, linewidth=2, marker="o", markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(depth)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("prefetch depth (batches)", color=INK_SECONDARY)
    _style_axes(ax, "peak RSS (MB)")
    ax.set_title("Peak memory vs. prefetch depth", color=INK_PRIMARY, loc="left")
    fig.tight_layout()
    fig.savefig("docs/plots/memory_vs_depth.png", dpi=150)
    plt.close(fig)


def plot_pytorch_comparison():
    workers = [1, 2, 4, 8]
    ours = [25.35, 53.94, 96.22, 161.12]
    pytorch_steady = [5.70, 10.88, 19.03, 30.74]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, ours, color=BLUE, linewidth=2, marker="o", markersize=8, label="this project")
    ax.plot(workers, pytorch_steady, color=ORANGE, linewidth=2, marker="o", markersize=8, label="PyTorch DataLoader (steady-state)")
    ax.set_xticks(workers)
    ax.set_yscale("log")
    ax.set_xlabel("worker count", color=INK_SECONDARY)
    _style_axes(ax, "samples/sec (thousands, log scale)")
    ax.set_title("Throughput vs. PyTorch DataLoader", color=INK_PRIMARY, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/plots/pytorch_comparison.png", dpi=150)
    plt.close(fig)


def main():
    plot_throughput_vs_workers()
    plot_starvation_vs_workers()
    plot_starvation_vs_depth()
    plot_cost_sweep()
    plot_memory_vs_depth()
    plot_pytorch_comparison()
    print("wrote docs/plots/*.png")


if __name__ == "__main__":
    main()
