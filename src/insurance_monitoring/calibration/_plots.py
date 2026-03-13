"""Visualisation functions for calibration diagnostics.

Matplotlib is imported lazily within each function so that the library is
importable without matplotlib installed (it is a required dependency but this
pattern keeps the import traceback clean if the user has an older environment).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes
    from ._types import AutoCalibResult, BalanceResult, MurphyResult, CalibrationReport


def plot_auto_calibration(
    result: "AutoCalibResult",
    title: str = "Auto-Calibration Diagram",
    log_scale: bool = True,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    """Reliability diagram: predicted vs actual per prediction bin.

    Each point is one prediction quantile bin. Points are sized by exposure.
    The isotonic step function is overlaid in red. The diagonal (y=x) in dashed
    grey represents perfect calibration.

    Parameters
    ----------
    result
        Output of :func:`check_auto_calibration`.
    title
        Plot title.
    log_scale
        If True, use log scale on both axes (recommended for Poisson/Gamma).
    ax
        Existing matplotlib Axes to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    df = result.per_bin
    pred = df["pred_mean"].to_numpy()
    obs = df["obs_mean"].to_numpy()
    exposure = df["exposure"].to_numpy()

    # Size markers by exposure (normalised)
    sizes = 200 * exposure / exposure.max()

    ax.scatter(pred, obs, s=sizes, color="steelblue", alpha=0.8,
               zorder=3, label="Bin mean (size ∝ exposure)")

    # 45-degree diagonal
    all_vals = np.concatenate([pred, obs])
    vmin, vmax = all_vals.min() * 0.9, all_vals.max() * 1.1
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="grey", alpha=0.6,
            linewidth=1.5, label="Perfect calibration (y=x)", zorder=1)

    # Isotonic step function (step-post style)
    sort_idx = np.argsort(pred)
    ax.step(pred[sort_idx], obs[sort_idx], where="post",
            color="firebrick", linewidth=2, label="Isotonic fit", zorder=2)

    if log_scale and all_vals.min() > 0:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Mean predicted rate (per bin)")
    ax.set_ylabel("Mean observed rate (per bin)")
    ax.set_title(title)
    ax.legend(fontsize=9)

    cal_status = "calibrated" if result.is_calibrated else "NOT calibrated"
    ax.annotate(
        f"p={result.p_value:.3f} ({cal_status})\n"
        f"MCB={result.mcb_score:.5f}\n"
        f"Worst bin ratio: {result.worst_bin_ratio:.3f}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    fig.tight_layout()
    return fig


def plot_murphy(
    result: "MurphyResult",
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    """Horizontal stacked bar chart of the Murphy decomposition.

    Shows UNC partitioned into DSC (skill from ranking) and the remainder,
    with MCB (GMCB + LMCB) indicated separately. This makes it easy to see
    at a glance whether miscalibration is dominated by global or local errors.

    Parameters
    ----------
    result
        Output of :func:`murphy_decomposition`.
    ax
        Existing matplotlib Axes. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.get_figure()

    unc = result.uncertainty
    dsc = result.discrimination
    mcb = result.miscalibration
    gmcb = result.global_mcb
    lmcb = result.local_mcb
    total = result.total_deviance

    # Decomposition: D = UNC - DSC + MCB
    # Layout on bar: [DSC | residual_unc | MCB(=GMCB+LMCB)]
    # residual_unc = UNC - DSC - MCB = total? No: D = UNC - DSC + MCB
    # So: UNC = DSC + (D - MCB) + ... let's think differently.
    # The bar represents UNC (total baseline difficulty).
    # Within UNC: DSC = improvement from ranking, (UNC - DSC) = irreducible.
    # MCB = extra deviance the model incurs above what isotonic would give.
    # We show three segments: DSC, (UNC - DSC - MCB) residual, MCB
    # Actually simpler: show D as a bar, subdivided as (D - MCB) | MCB
    # and within MCB: GMCB | LMCB

    y_pos = 0.5

    # Segment 1: D - MCB = UNC - DSC (the irreducible/residual part)
    residual = total - mcb
    residual = max(residual, 0.0)

    ax.barh(y_pos, residual, height=0.4, color="#4e79a7", label="D - MCB (irreducible)")
    ax.barh(y_pos, gmcb, height=0.4, left=residual, color="#f28e2b",
            label=f"GMCB (global, fixable)")
    ax.barh(y_pos, lmcb, height=0.4, left=residual + gmcb, color="#e15759",
            label=f"LMCB (local, needs refit)")

    # Annotations
    def _label(x_start, width, text, fontsize=8):
        if width > 0.005 * total:
            ax.text(x_start + width / 2, y_pos, text,
                    ha="center", va="center", fontsize=fontsize, color="white",
                    fontweight="bold")

    _label(0, residual, f"D−MCB\n{residual:.4f}")
    _label(residual, gmcb, f"GMCB\n{gmcb:.4f}")
    _label(residual + gmcb, lmcb, f"LMCB\n{lmcb:.4f}")

    # Reference lines for UNC and DSC
    ax.axvline(unc, color="black", linestyle="--", linewidth=1.5, label=f"UNC={unc:.4f}")
    ax.axvline(unc - dsc, color="#76b7b2", linestyle=":",
               linewidth=1.5, label=f"UNC−DSC={unc-dsc:.4f}")

    ax.set_xlim(0, max(unc, total) * 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Deviance")
    ax.set_title(f"Murphy Decomposition — Verdict: {result.verdict}")
    ax.legend(fontsize=8, loc="lower right")

    ax.annotate(
        f"UNC={unc:.4f}  DSC={dsc:.4f} ({result.discrimination_pct:.1f}%)\n"
        f"MCB={mcb:.4f} ({result.miscalibration_pct:.1f}%)  "
        f"GMCB={gmcb:.4f}  LMCB={lmcb:.4f}",
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout()
    return fig


def plot_balance_over_time(
    periods: list[str],
    balance_ratios: list[float],
    ci_lowers: list[float],
    ci_uppers: list[float],
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    """Line chart of balance ratio by period with confidence bands.

    Points are coloured red where the CI excludes 1.0 (the CI does not
    straddle the balance line), indicating statistically significant imbalance.

    Parameters
    ----------
    periods
        Period labels (e.g. accident year or quarter).
    balance_ratios
        Point estimates of balance ratio per period.
    ci_lowers
        Lower bounds of CI per period.
    ci_uppers
        Upper bounds of CI per period.
    ax
        Existing matplotlib Axes. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    x = np.arange(len(periods))
    ratios = np.array(balance_ratios)
    lowers = np.array(ci_lowers)
    uppers = np.array(ci_uppers)

    is_imbalanced = (uppers < 1.0) | (lowers > 1.0)

    # CI band
    ax.fill_between(x, lowers, uppers, alpha=0.15, color="steelblue",
                    label="95% CI")

    # Main line
    ax.plot(x, ratios, "-o", color="steelblue", linewidth=2, zorder=3)

    # Highlight imbalanced periods
    if np.any(is_imbalanced):
        ax.plot(x[is_imbalanced], ratios[is_imbalanced], "o",
                color="firebrick", markersize=10, zorder=4,
                label="Imbalanced (CI excludes 1.0)")

    # Reference line at 1.0
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Perfect balance (α=1.0)")

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha="right")
    ax.set_ylabel("Balance ratio α")
    ax.set_title("Balance Ratio Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_calibration_report(
    report: "CalibrationReport",
    figsize: tuple[float, float] = (14, 10),
) -> "matplotlib.figure.Figure":
    """Three-panel combined diagnostic figure for model sign-off documentation.

    Layout:
    - Top left:  Auto-calibration reliability diagram
    - Top right: Murphy decomposition bar chart
    - Bottom:    Per-bin obs/exp table rendered as a heatmap

    Suitable for regulatory reporting and model documentation packs.

    Parameters
    ----------
    report
        Output of :meth:`CalibrationChecker.check`.
    figsize
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.4, wspace=0.35)

    ax_ac = fig.add_subplot(gs[0, 0])
    ax_murphy = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    # Auto-calibration diagram
    plot_auto_calibration(report.auto_calibration, ax=ax_ac)

    # Murphy bar chart
    plot_murphy(report.murphy, ax=ax_murphy)

    # Per-bin heatmap table
    df = report.auto_calibration.per_bin
    ratios = df["ratio"].to_numpy()
    pred = df["pred_mean"].to_numpy()
    obs = df["obs_mean"].to_numpy()
    exposure = df["exposure"].to_numpy()
    bins = df["bin"].to_numpy()

    # Heatmap: colour by ratio deviation from 1.0
    norm = mcolors.TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.3)
    cmap = plt.cm.RdYlGn

    bar_width = 0.7
    for i, (b, r, e) in enumerate(zip(bins, ratios, exposure)):
        colour = cmap(norm(r))
        ax_table.bar(i, 1.0, width=bar_width, color=colour, alpha=0.85)
        ax_table.text(i, 0.5, f"{r:.3f}\n(exp={e:.0f})",
                      ha="center", va="center", fontsize=7.5)

    ax_table.set_xticks(range(len(bins)))
    ax_table.set_xticklabels([f"Bin {b}\n{p:.4f}" for b, p in zip(bins, pred)],
                              fontsize=7)
    ax_table.set_yticks([])
    ax_table.set_title("Per-Bin Obs/Exp Ratio (green=well-calibrated, red=miscalibrated)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_table, orientation="vertical", fraction=0.02,
                 label="obs/exp ratio")

    verdict = report.verdict()
    fig.suptitle(
        f"Calibration Report — {report.distribution.capitalize()} "
        f"— n={report.n_policies:,} — Verdict: {verdict}",
        fontsize=13,
        fontweight="bold",
    )

    return fig
