from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib as mpl
import numpy as np

_SCALE_TO_XY = {
    "linear":   ("linear", "linear"),
    "semilogx": ("log",    "linear"),
    "semilogy": ("linear", "log"),
    "loglog":   ("log",    "log"),
}


def build_xy_axes(
    nrows=1, ncols=1, index=0, figsize=(6, 4), height_ratios=None,
    dpi=600, sharex=False, sharey=False, show_top=True, show_right=True,
    tight=True, equal_aspect=False, scale="linear", hspace=0.0, wspace=0.0,
):
    if height_ratios is None:
        height_ratios = [1] * nrows
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, height_ratios=height_ratios,
        dpi=dpi, sharex=sharex, sharey=sharey,
    )
    plt.rcParams.update({"font.size": 12})

    key = scale.lower()
    if key not in _SCALE_TO_XY:
        raise ValueError(f"Unknown scale '{scale}'. Choose from {list(_SCALE_TO_XY)}.")
    xscale, yscale = _SCALE_TO_XY[key]
    axs_arr = np.ravel(axs) if isinstance(axs, np.ndarray) else np.array([axs])

    def _style(ax):
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_linewidth(2.0)
        ax.spines["top"].set_visible(show_top)
        ax.spines["right"].set_visible(show_right)
        ax.tick_params(direction="in", length=4, width=1, which="major", top=show_top, right=show_right)
        ax.tick_params(direction="in", length=3, width=0.8, which="minor", top=show_top, right=show_right)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    if nrows > 1:
        fig.subplots_adjust(hspace=hspace)
    if ncols > 1:
        fig.subplots_adjust(wspace=wspace)

    if index is None:
        for a in axs_arr:
            _style(a)
        if tight:
            fig.tight_layout()
        return fig, (axs if isinstance(axs, np.ndarray) else np.array([axs]))
    else:
        ax = axs_arr[index]
        _style(ax)
        if tight:
            fig.tight_layout()
        return fig, ax


def _to_seq(obj, n=None):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj] if n is None else [obj] * n


def _filter_for_log_axes(ax, x, y, strict=False, return_mask=False):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if ax.get_xscale() == "log":
        mask &= x > 0
    if ax.get_yscale() == "log":
        mask &= y > 0
    if strict and not np.all(mask):
        raise ValueError("Nonpositive or nonfinite values for log axes.")
    if return_mask:
        return x[mask], y[mask], mask
    return x[mask], y[mask]


def plot_scatter(
    ax, x, y, label=None, s=24, marker="o", alpha=1.0,
    edgecolor=None, facecolor=None, c=None,
    xlim=None, ylim=None, xlabel=None, ylabel=None, title=None,
    legend=True, strict_log=False, **kwargs,
):
    is_multi = isinstance(x, (list, tuple)) and isinstance(y, (list, tuple))
    if not is_multi:
        x, y = _filter_for_log_axes(ax, x, y, strict=strict_log)
        ax.scatter(x, y, s=s, marker=marker, alpha=alpha,
                   edgecolors=edgecolor, facecolors=facecolor, c=c, label=label, **kwargs)
    else:
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of series.")
        for i, (xi, yi) in enumerate(zip(x, y)):
            xi, yi = _filter_for_log_axes(ax, xi, yi, strict=strict_log)
            series_label = label[i] if isinstance(label, (list, tuple)) else None
            ax.scatter(xi, yi, s=s, marker=marker, alpha=alpha,
                       edgecolors=edgecolor, facecolors=facecolor,
                       c=None if isinstance(c, (list, tuple)) else c,
                       label=series_label, **kwargs)

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if legend and label: ax.legend(frameon=False)
    return ax


def scatter_xy(x, y, scale="linear", **kwargs):
    fig, ax = build_xy_axes(scale=scale)
    plot_scatter(ax, x, y, **kwargs)
    return fig, ax


def color_hist_patches(patches, *, cmap=None, edges=None, manual_limits=None,
                       norm=None, vmin=None, vmax=None):
    if cmap is None:
        cmap = mpl.colormaps[mpl.rcParams.get("image.cmap", "viridis")]
    elif isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if manual_limits is not None:
        values = np.linspace(*manual_limits, len(patches))
    elif edges is not None:
        edges = np.asarray(edges, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if len(centers) != len(patches):
            raise ValueError(f"bins ({len(centers)}) must match patches ({len(patches)}).")
        values = centers
    else:
        raise ValueError("Provide either `edges` or `manual_limits`.")

    if norm is None:
        vmin = np.min(values) if vmin is None else vmin
        vmax = np.max(values) if vmax is None else vmax
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for val, patch in zip(values, patches):
        patch.set_facecolor(cmap(norm(val)))
    return norm, cmap


def plot_histogram(
    ax, data, *, bins="fd", binwidth=None, range=None, density=True,
    cmap=None, manual_limits=None, xlabel=None, ylabel=None, title=None,
    xlim=None, ylim=None, log_tag=False, show_stats=True,
    stats_unit="s", stats_fontsize=9, **kwargs,
):
    d = np.asarray(data)
    d = d[np.isfinite(d)]
    if ax.get_xscale() == "log":
        d = d[d > 0]
    if d.size == 0:
        raise ValueError("No finite data to plot.")

    if binwidth is not None:
        lo, hi = range if range is not None else (d.min(), d.max())
        bins = np.arange(lo, hi + binwidth, binwidth)

    n, edges, patches = ax.hist(d, bins=bins, range=range, density=density, **kwargs)

    if cmap is not None:
        color_hist_patches(patches, cmap=cmap, edges=edges, manual_limits=manual_limits)

    if log_tag:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: rf"$10^{{{int(round(x))}}}$"
        ))

    if xlabel: ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else ("Density" if density else "Count"))
    if title: ax.set_title(title)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    if show_stats:
        lin_data = 10**d if log_tag else d
        median = np.median(lin_data)
        q25, q75 = np.percentile(lin_data, [25, 75])
        center = 10**np.mean(d) if log_tag else np.mean(d)
        spread = np.std(d)
        labels_s = ["median", "μg" if log_tag else "μ", "q25", "q75", "σlog10" if log_tag else "σ"]
        values_s = [f"{v:.1f} {stats_unit}" for v in [median, center, q25, q75]] + [f"{spread:.1f}"]
        w = max(len(l) for l in labels_s)
        ax.text(0.95, 0.95,
                "\n".join(f"{l:<{w}} = {v:>7}" for l, v in zip(labels_s, values_s)),
                transform=ax.transAxes, fontsize=stats_fontsize, family="monospace",
                va="top", ha="right", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    return n, edges, patches


def histogram(data, *, bins="fd", binwidth=None, range=None, density=True, cmap=None,
              xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, log_tag=False, **kwargs):
    fig, ax = build_xy_axes(scale="linear")
    n, edges, patches = plot_histogram(
        ax, data, bins=bins, binwidth=binwidth, range=range, density=density,
        cmap=cmap, xlabel=xlabel, ylabel=ylabel, title=title,
        xlim=xlim, ylim=ylim, log_tag=log_tag, **kwargs,
    )
    return fig, ax, n, edges, patches


def plot_lines(
    ax, x, y, *, label=None, color=None, linestyle="-", linewidth=1.8,
    marker=None, markersize=4, alpha=1.0, scale="linear",
    draw_vlines=None, draw_vlines_color="gray", display_properties=None,
    yerr=None, y_min=None, y_max=None, capsize=3,
    y2=None, y2label=None, y2color=None, y2linestyle=":", y2linewidth=1.8,
    y2marker=None, y2markersize=4, y2alpha=0.9, y2scale="linear",
    y2err=None, y2_min=None, y2_max=None, capsize2=3,
    xlim=None, ylim=None, xlabel=None, ylabel=None, title=None,
    legend=True, strict_log=False, **kwargs,
):
    def _ts(v, n):
        return list(v) + [None] * (n - len(v)) if isinstance(v, (list, tuple)) else [v] * n

    def _asarr(a):
        return None if a is None else np.asarray(a)

    def _build_yerr(yi, yerr_i, ymin_i, ymax_i):
        if yerr_i is not None:
            return np.asarray(yerr_i)
        if ymin_i is None and ymax_i is None:
            return None
        return np.vstack([yi if ymin_i is None else ymin_i,
                          yi if ymax_i is None else ymax_i])

    def _filter_log(ax_like, xi, yi, yerri):
        mask = np.ones_like(yi, dtype=bool)
        if strict_log:
            if ax_like.get_xscale() == "log":
                mask &= xi > 0
            if ax_like.get_yscale() == "log":
                mask &= yi > 0
        xi_f, yi_f = xi[mask], yi[mask]
        if yerri is None:
            return xi_f, yi_f, None
        yerri = np.asarray(yerri)
        if yerri.ndim == 0:
            return xi_f, yi_f, yerri
        if yerri.ndim == 1:
            return xi_f, yi_f, yerri[mask]
        if yerri.ndim == 2 and yerri.shape[0] == 2:
            return xi_f, yi_f, np.vstack([yerri[0, mask], yerri[1, mask]])
        raise ValueError("yerr must be scalar, (N,), or (2,N).")

    multi = isinstance(y, (list, tuple))
    if not multi:
        y = [np.asarray(y)]
        x = [None if x is None else np.asarray(x)]
    else:
        y = [np.asarray(yi) for yi in y]
        x = [None] * len(y) if x is None else [None if xi is None else np.asarray(xi) for xi in x]
    n = len(y)

    labels = _ts(label, n); colors = _ts(color, n); linestyles = _ts(linestyle, n)
    linewidths = _ts(linewidth, n); markers = _ts(marker, n); markersizes = _ts(markersize, n)
    alphas = _ts(alpha, n); yerrs = _ts(yerr, n); ymins = _ts(y_min, n); ymaxs = _ts(y_max, n)

    xscale, yscale = _SCALE_TO_XY[scale]
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    handles_left = []

    for i in range(n):
        xi = x[i] if x[i] is not None else np.arange(len(y[i]), dtype=float)
        yi = y[i]
        yerri = _build_yerr(yi, yerrs[i], _asarr(ymins[i]), _asarr(ymaxs[i]))
        xi_f, yi_f, yerri_f = _filter_log(ax, np.asarray(xi), np.asarray(yi), yerri)
        style = dict(label=labels[i], color=colors[i], linestyle=linestyles[i],
                     linewidth=linewidths[i], marker=markers[i],
                     markersize=markersizes[i], alpha=alphas[i], **kwargs)
        if yerri_f is not None:
            h = ax.errorbar(xi_f, yi_f, yerr=yerri_f, capsize=capsize, **style)
        else:
            (h,) = ax.plot(xi_f, yi_f, **style)
        handles_left.append(h)

    if draw_vlines is not None:
        for i, xline in enumerate(draw_vlines):
            if draw_vlines_color == "same":
                h = handles_left[i]
                line_color = h.get_color() if hasattr(h, "get_color") else h.lines[0].get_color()
            else:
                line_color = draw_vlines_color
            ax.axvline(x=xline, color=line_color, linestyle=":", linewidth=1.2, alpha=0.8, zorder=0)

    if display_properties is not None:
        x0, y0, dx, dy, spacing, labels_stats, values_stats = display_properties.values()
        for i in range(n):
            w = max(len(l) for l in labels_stats)
            stats_text = "\n".join(
                f"{l:<{w}} = {v:>{spacing}}" for l, v in zip(labels_stats, values_stats[i])
            )
            xpos = x0 - i * dx; ypos = y0 - i * dy
            ax.text(xpos, ypos, stats_text, transform=ax.transAxes, fontsize=9,
                    family="monospace", va="top", ha="left",
                    color=colors[i] if colors[i] is not None else None,
                    bbox=dict(facecolor="none", edgecolor="none", pad=2))
            if labels[i] is not None:
                ax.text(xpos, ypos + 0.01, f"{labels[i]}:", transform=ax.transAxes,
                        fontsize=9, family="monospace", va="bottom", ha="left",
                        color=colors[i] if colors[i] is not None else None)

    ax2 = None
    handles_right = []
    if y2 is not None:
        ax2 = ax.twinx()
        y2_list = ([None if v is None else np.asarray(v) for v in y2]
                   if isinstance(y2, (list, tuple)) else [np.asarray(y2)] + [None] * (n - 1))
        y2errs = _ts(y2err, n); y2mins = _ts(y2_min, n); y2maxs = _ts(y2_max, n)
        y2colors = _ts(y2color, n); y2markers = _ts(y2marker, n); y2ms = _ts(y2markersize, n)
        ax2.set_yscale(y2scale)
        for i in range(n):
            yi2 = y2_list[i]
            if yi2 is None:
                handles_right.append(None)
                continue
            xi = x[i] if x[i] is not None else np.arange(len(yi2), dtype=float)
            y2erri = _build_yerr(yi2, y2errs[i], _asarr(y2mins[i]), _asarr(y2maxs[i]))
            xi_f, yi2_f, y2erri_f = _filter_log(ax2, np.asarray(xi), np.asarray(yi2), y2erri)
            c2 = (y2colors[i] if y2colors[i] is not None
                  else (handles_left[i].get_color() if hasattr(handles_left[i], "get_color") else colors[i]))
            style2 = dict(color=c2, linestyle=y2linestyle, linewidth=y2linewidth,
                          marker=y2markers[i], markersize=y2ms[i], alpha=y2alpha)
            if y2erri_f is not None:
                h2 = ax2.errorbar(xi_f, yi2_f, yerr=y2erri_f, capsize=capsize2, **style2)
            else:
                (h2,) = ax2.plot(xi_f, yi2_f, **style2)
            handles_right.append(h2)
        if y2label:
            ax2.set_ylabel(y2label)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if legend and any(l is not None for l in labels):
        ax.legend(frameon=False, fontsize=9)
    return ax, ax2, handles_left, handles_right


def lines_xy(fig, ax, x, y, *, scale="linear", **kwargs):
    plot_lines(ax, x, y, scale=scale, **kwargs)
    return fig, ax


def add_secondary_yaxis(ax, y2label, kind="reciprocal"):
    if kind == "reciprocal":
        secax = ax.secondary_yaxis("right", functions=(lambda y: 1 / y, lambda y: 1 / y))
        if y2label:
            secax.set_ylabel(y2label)
        return secax
    return None
