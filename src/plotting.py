# src/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

from src.statistics_engine import StatisticsEngine, stars, format_p


class PlottingEngine:
    @staticmethod
    def generate_plot(df, plot_type, x_col, y_col, ax, dark_mode=False, palette="muted",
                       stat_mode="Auto (Prism Mode)", hue_col=None, log_scale=False,
                       reference=None, pairing_col=None, show_pvalue=False,
                       y_min=None, y_max=None, group_colors=None, bw_mode=False,
                       hist_bins=20, custom_title=None, custom_xlabel=None,
                       custom_ylabel=None, group_order=None, show_ns=False):
        """Generates plots while respecting active application theme states and overlays stars."""
        ax.clear()

        if dark_mode:
            plt.style.use("dark_background")
            ax.set_facecolor("#1E1E1E")
            ax.figure.set_facecolor("#1E1E1E")
            text_color = "#E0E0E0"
            grid_color = "#333333"
            line_color = "#AAAAAA"
        else:
            plt.style.use("default")
            ax.set_facecolor("white")
            ax.figure.set_facecolor("white")
            text_color = "#2C3E50"
            grid_color = "#E2E8F0"
            line_color = "#444444"

        if df is None or df.empty or not x_col or not y_col:
            ax.text(0.5, 0.5, "Select X and Y variables to plot", ha='center', va='center', transform=ax.transAxes, color=text_color)
            return

        ax.grid(True, linestyle="--", alpha=0.5, color=grid_color)

        # Keep an explicit categorical order for alignment
        unique_groups = list(df[x_col].dropna().unique())
        if len(unique_groups) == 0:
            return

        if group_order:
            # Manual reordering (from right-clicking a group's label/legend
            # entry on the plot). Groups not mentioned in group_order keep
            # their natural (first-seen) relative order, appended at the end
            # - so a newly-appearing group (new file loaded, filter changed)
            # doesn't just vanish from the axis.
            ordered = [g for g in group_order if g in unique_groups]
            remaining = [g for g in unique_groups if g not in ordered]
            unique_groups = ordered + remaining

        hue_col = hue_col if (hue_col and hue_col in df.columns and hue_col != x_col) else None
        legend_needed = hue_col is not None
        is_line = (plot_type == "Line")

        # --- Resolve the actual color for each "color category" (hue levels if a
        # Group/Color By column is set, otherwise the x-axis groups themselves).
        # Any category the user has manually recolored (via clicking the plot)
        # keeps that color; everything else falls back to the chosen palette. ---
        group_colors = group_colors or {}
        if hue_col:
            color_categories = list(df[hue_col].dropna().unique())
        elif plot_type == "Line":
            # A single line (no Group/Color-By column) has no natural set of
            # "categories" to click-recolor - key it off y_col itself so the
            # curve is still individually addressable/recolorable.
            color_categories = [y_col]
        else:
            color_categories = unique_groups

        if bw_mode:
            # Black & white mode: replace hue entirely with grayscale shades
            # (still one shade per category so groups stay distinguishable),
            # ignoring the seaborn palette and any manual color picks.
            n = max(len(color_categories), 1)
            shades = np.linspace(0.75, 0.05, n) if n > 1 else np.array([0.2])
            default_colors = [(s, s, s) for s in shades]
        else:
            default_colors = sns.color_palette(palette, n_colors=max(len(color_categories), 1))
        resolved_palette = [group_colors.get(str(cat), default_colors[i]) if not bw_mode
                             else default_colors[i]
                             for i, cat in enumerate(color_categories)]

        # --- Special plot types: these don't share the categorical group /
        # significance-overlay / xtick-rotation pipeline below, so they draw
        # themselves fully and return early. ---
        if plot_type in ("Heatmap", "Volcano Plot", "Dendrogram", "Network Graph", "Stacked Histogram", "Stacked Bar"):
            if plot_type == "Heatmap":
                PlottingEngine._draw_heatmap(df, x_col, y_col, hue_col, ax, palette, text_color)
            elif plot_type == "Volcano Plot":
                PlottingEngine._draw_volcano(df, x_col, y_col, hue_col, ax, text_color, grid_color)
            elif plot_type == "Dendrogram":
                PlottingEngine._draw_dendrogram(df, x_col, y_col, ax, palette, text_color)
            elif plot_type == "Network Graph":
                PlottingEngine._draw_network(df, x_col, y_col, ax, palette, text_color)
            elif plot_type == "Stacked Histogram":
                PlottingEngine._draw_stacked_histogram(df, x_col, y_col, hue_col, ax, resolved_palette,
                                                         text_color, bw_mode, hist_bins)
            elif plot_type == "Stacked Bar":
                PlottingEngine._draw_stacked_bar(df, x_col, y_col, hue_col, ax, resolved_palette,
                                                   text_color, unique_groups, bw_mode, group_colors)
            return

        hue_order = color_categories

        if plot_type == "Bar":
            PlottingEngine._draw_bar(df, x_col, y_col, ax, unique_groups, hue_col, resolved_palette, dark_mode, text_color, bw_mode)
        elif plot_type == "Line":
            PlottingEngine._draw_line(df, x_col, y_col, ax, hue_col, resolved_palette, text_color, bw_mode)
        elif plot_type == "Boxplot":
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                        palette=resolved_palette, width=0.5, order=unique_groups, hue_order=hue_order,
                        legend=legend_needed, dodge=bool(hue_col))
        elif plot_type == "Violin":
            sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                            palette=resolved_palette, inner="quartile", order=unique_groups, hue_order=hue_order,
                            legend=legend_needed, dodge=bool(hue_col))
        elif plot_type == "Scatter":
            # Color by hue_col when set, otherwise fall back to x_col so each
            # category still gets its own color (matching Boxplot/Violin/Beeswarm)
            # instead of collapsing every point to one flat blue.
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                             alpha=0.8, s=60, palette=resolved_palette, hue_order=hue_order, legend=legend_needed)
        elif plot_type == "Beeswarm":
            sns.swarmplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                          palette=resolved_palette, size=5, order=unique_groups, hue_order=hue_order,
                          legend=legend_needed, dodge=bool(hue_col))
        elif plot_type == "Box + Swarm":
            sns.boxplot(data=df, x=x_col, y=y_col, color=("#2D2D2D" if dark_mode else "white"),
                        ax=ax, width=0.5, fliersize=0, order=unique_groups)
            # Same fix as Scatter: fall back to x_col for hue so dots are
            # colored by group even when no explicit hue_col is chosen.
            sns.swarmplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax,
                          palette=resolved_palette, size=4, alpha=0.6, order=unique_groups, hue_order=hue_order,
                          legend=legend_needed, dodge=bool(hue_col))

        if log_scale:
            ax.set_yscale("log")
            lo_hi = pd.to_numeric(df[y_col], errors="coerce")
            lo_hi = lo_hi[lo_hi > 0]
            if len(lo_hi) and lo_hi.max() / lo_hi.min() < 100:
                # under two decades the default log ticks have no labels
                from matplotlib.ticker import FuncFormatter, LogLocator
                fmt = FuncFormatter(lambda v, _: f"{v:g}")
                ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(fmt)
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))

        # --- STATISTICAL OVERLAY CORE (categorical charts only — a continuous
        # line plot compares curves, not discrete groups, so it's skipped) ---
        if (not is_line and len(unique_groups) >= 2 and plot_type in
                ["Boxplot", "Violin", "Beeswarm", "Box + Swarm", "Bar"] and stat_mode != "None"):
            bar_tops = None
            if plot_type == "Bar":
                # brackets must clear the error bars, not just the points
                keys = [x_col] + ([hue_col] if hue_col else [])
                yv = pd.to_numeric(df[y_col], errors="coerce")
                g = yv.groupby([df[k] for k in keys])
                tops = (g.mean() + g.sem().fillna(0)).dropna()
                if hue_col:
                    tops = tops.groupby(level=0).max()
                bar_tops = {str(k): float(v) for k, v in tops.items()}
            PlottingEngine.annotate_significance(
                df, x_col, y_col, unique_groups, ax, stat_mode, line_color,
                reference=reference, pairing_col=pairing_col,
                log_scale=log_scale, show_pvalue=show_pvalue,
                hue_col=hue_col, hue_order=[str(h) for h in hue_order],
                dodge_width=PlottingEngine.DODGE_WIDTH.get(plot_type, 0.8),
                show_ns=show_ns, bar_tops=bar_tops)

        if not is_line:
            # Rotate labels 45 degrees, shrink font to size 10, and right-align them
            ax.set_xticklabels(
                [str(g) for g in unique_groups],
                rotation=45,
                horizontalalignment='right',
                fontsize=10
            )
            if not hue_col:
                # Make each x-tick label clickable so the UI can offer
                # "click a group's label to recolor it" (only meaningful when
                # colors are assigned per x-group, i.e. no separate hue column).
                for label in ax.get_xticklabels():
                    label.set_picker(True)
                    label.set_url("group_color:" + label.get_text())

        # Style labels to match dark/light theme choices
        sns.despine(ax=ax, left=True, bottom=True)

        # Title / axis labels: each is picker-enabled with a distinct URL so
        # the UI can catch a click on it (same pick_event pathway used for
        # group-label recoloring) and swap in a user-typed override. Falls
        # back to the auto-generated text until the user has customized it.
        default_title = f"{y_col} vs {x_col}" + (f" (by {hue_col})" if hue_col else "")
        title_obj = ax.set_title(custom_title if custom_title else default_title,
                                  fontsize=14, fontweight='bold', pad=15, color=text_color)
        title_obj.set_picker(True)
        title_obj.set_url(f"edit_title:{y_col}")

        xlabel_obj = ax.set_xlabel(custom_xlabel if custom_xlabel else x_col,
                                    fontsize=12, fontweight='bold', color=text_color)
        xlabel_obj.set_picker(True)
        xlabel_obj.set_url("edit_xlabel")

        ylabel_obj = ax.set_ylabel(custom_ylabel if custom_ylabel else y_col,
                                    fontsize=12, fontweight='bold', color=text_color)
        ylabel_obj.set_picker(True)
        ylabel_obj.set_url(f"edit_ylabel:{y_col}")

        ax.tick_params(colors=text_color)
        if legend_needed and not is_line:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(title=hue_col, fontsize=8, title_fontsize=9)
                # Make each legend label clickable too, so hue-colored plots
                # (color assigned by Group/Color-By) can also be recolored by
                # clicking their legend entry instead of an x-tick label.
                for text_obj, lbl in zip(legend.get_texts(), labels):
                    text_obj.set_picker(True)
                    text_obj.set_url("group_color:" + lbl)

        # Manual y-axis range: applied last so it always wins over the
        # significance overlay's auto-computed headroom, whichever the
        # user actually set (only overriding the side(s) they specified).
        if not log_scale and (y_min is not None or y_max is not None):
            cur_bottom, cur_top = ax.get_ylim()
            ax.set_ylim(bottom=y_min if y_min is not None else cur_bottom,
                        top=y_max if y_max is not None else cur_top)

    @staticmethod
    def _draw_bar(df, x_col, y_col, ax, unique_groups, hue_col, palette, dark_mode, text_color, bw_mode=False):
        """Grouped bar chart of mean +/- SEM, matching the classic Prism-style bar plot."""
        df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna(subset=[x_col, y_col])
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[y_col])

        # In B&W mode, layer a distinct hatch pattern per category on top of
        # the grayscale shades so bars stay distinguishable without color.
        hatch_cycle = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]

        if hue_col:
            hue_levels = list(df_clean[hue_col].dropna().unique())
            n_hues = len(hue_levels)
            bar_width = 0.8 / max(n_hues, 1)
            # `palette` here is already the resolved per-category color list built
            # in generate_plot (defaults + any manual click-to-recolor overrides),
            # in the same order as hue_levels — use it directly rather than
            # re-deriving colors, so overrides land on the right bar.
            colors = palette if len(palette) >= n_hues else sns.color_palette("muted", n_colors=max(n_hues, 1))

            for h_idx, h_level in enumerate(hue_levels):
                sub = df_clean[df_clean[hue_col] == h_level]
                means, sems, xs = [], [], []
                for g_idx, g in enumerate(unique_groups):
                    vals = sub[sub[x_col] == g][y_col].values
                    means.append(np.mean(vals) if len(vals) else 0.0)
                    sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)
                    offset = (h_idx - (n_hues - 1) / 2.0) * bar_width
                    xs.append(g_idx + offset)
                bars = ax.bar(xs, means, width=bar_width * 0.95, yerr=sems, capsize=3,
                       color=colors[h_idx], label=str(h_level),
                       edgecolor=text_color, linewidth=0.6, error_kw={"ecolor": text_color, "elinewidth": 1})
                if bw_mode:
                    for b in bars:
                        b.set_hatch(hatch_cycle[h_idx % len(hatch_cycle)])
        else:
            # No grouping column: give each x-axis category its own color,
            # the same way Boxplot/Violin do (rather than one flat color).
            # `palette` is the resolved per-category list from generate_plot.
            colors = palette if len(palette) >= len(unique_groups) else sns.color_palette("muted", n_colors=max(len(unique_groups), 1))
            means, sems = [], []
            for g in unique_groups:
                vals = df_clean[df_clean[x_col] == g][y_col].values
                means.append(np.mean(vals) if len(vals) else 0.0)
                sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)
            bars = ax.bar(range(len(unique_groups)), means, width=0.6, yerr=sems, capsize=3,
                   color=colors, edgecolor=text_color, linewidth=0.6,
                   error_kw={"ecolor": text_color, "elinewidth": 1})
            if bw_mode:
                for g_idx, b in enumerate(bars):
                    b.set_hatch(hatch_cycle[g_idx % len(hatch_cycle)])

        ax.set_xticks(range(len(unique_groups)))

    @staticmethod
    def _draw_line(df, x_col, y_col, ax, hue_col, palette, text_color, bw_mode=False):
        """
        Connected mean +/- SEM line plot for continuous X data (e.g. a time
        course or substrate-concentration curve in an enzyme kinetics assay).
        One line per hue level if a Group/Color-By column is set.
        """
        df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna(subset=[x_col, y_col])
        df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[x_col, y_col])

        if df_clean.empty:
            ax.text(0.5, 0.5, f"'{x_col}' and/or '{y_col}' are not numeric — Line plots need numeric X/Y values",
                     ha='center', va='center', transform=ax.transAxes, color=text_color, fontsize=10, wrap=True)
            return

        hue_levels = list(df_clean[hue_col].dropna().unique()) if hue_col else [y_col]
        # `palette` arrives pre-resolved from generate_plot (defaults, manual
        # click-to-recolor overrides, or B&W shades already applied) - use it
        # directly instead of re-deriving colors from the palette name.
        colors = palette if len(palette) >= len(hue_levels) else sns.color_palette("muted", n_colors=max(len(hue_levels), 1))
        # In B&W mode, also vary line style/marker per curve so overlapping
        # conditions stay distinguishable without relying on color/print.
        bw_styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"), ("-", "v"), ("--", "P")]

        lines_for_legend = []
        for idx, h_level in enumerate(hue_levels):
            sub = df_clean[df_clean[hue_col] == h_level] if hue_col else df_clean
            grouped = sub.groupby(x_col)[y_col]
            means = grouped.mean()
            sems = grouped.apply(lambda v: stats.sem(v) if len(v) > 1 else 0.0)
            order = means.index.argsort()
            xs = means.index.values[order]
            ys = means.values[order]
            errs = sems.values[order]
            linestyle, marker = bw_styles[idx % len(bw_styles)] if bw_mode else ("-", "o")
            line = ax.errorbar(xs, ys, yerr=errs, marker=marker, linestyle=linestyle, capsize=3,
                        color=colors[idx], linewidth=1.8, markersize=5,
                        label=str(h_level))
            lines_for_legend.append((line, str(h_level)))

        # Always show a legend for Line plots (even a single curve) so every
        # curve has a clickable target for manual recoloring.
        legend = ax.legend(title=hue_col if hue_col else None, fontsize=8, title_fontsize=9)
        if legend:
            for text_obj, (line, lbl) in zip(legend.get_texts(), lines_for_legend):
                text_obj.set_picker(True)
                text_obj.set_url("group_color:" + lbl)

    @staticmethod
    def _draw_stacked_histogram(df, x_col, y_col, hue_col, ax, resolved_palette, text_color,
                                 bw_mode=False, bins=20):
        """
        Stacked histogram: bins y_col's numeric values, with one stacked
        layer per category - the Group/Color-By column if set, otherwise the
        x_col groups themselves - e.g. the distribution of a readout, split
        by condition, stacked into a single set of bars per bin.
        """
        stack_col = hue_col if hue_col else x_col
        df_clean = df[[y_col, stack_col]].copy()
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[y_col, stack_col])

        categories = list(df_clean[stack_col].dropna().unique())
        if not categories or df_clean.empty:
            ax.text(0.5, 0.5, f"Not enough numeric data in '{y_col}' to build a histogram",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        data_arrays = [df_clean[df_clean[stack_col] == cat][y_col].values for cat in categories]
        colors = resolved_palette if len(resolved_palette) >= len(categories) else \
            sns.color_palette("muted", n_colors=max(len(categories), 1))

        _, _, patch_layers = ax.hist(data_arrays, bins=bins, stacked=True, color=colors,
                                      edgecolor=text_color, linewidth=0.6,
                                      label=[str(c) for c in categories])

        if bw_mode:
            hatch_cycle = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]
            for idx, layer in enumerate(patch_layers):
                for patch in layer:
                    patch.set_hatch(hatch_cycle[idx % len(hatch_cycle)])

        ax.set_title(f"Stacked distribution of {y_col} by {stack_col}", fontsize=14,
                     fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel(y_col, fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel("Count", fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)

        legend = ax.legend(title=stack_col, fontsize=8, title_fontsize=9)
        if legend:
            for text_obj, cat in zip(legend.get_texts(), categories):
                text_obj.set_picker(True)
                text_obj.set_url("group_color:" + str(cat))
        sns.despine(ax=ax, left=True, bottom=True)

    @staticmethod
    def _draw_stacked_bar(df, x_col, y_col, hue_col, ax, resolved_palette, text_color,
                           unique_groups, bw_mode=False, group_colors=None):
        """
        Stacked bar chart: one bar per x_col category, with the bar's total
        height being the sum of y_col across that category's rows - shown as
        stacked segments. If a Group/Color-By column is set, each segment is
        the sum of y_col for that hue level within the x category (the
        classic "stacked bar of sub-totals" layout). Otherwise, each row
        belonging to that x category becomes its own stacked segment (e.g.
        stacking individual replicate readings into one total-per-sample bar).
        """
        hatch_cycle = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]
        df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna(subset=[x_col, y_col])
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[y_col])

        if df_clean.empty:
            ax.text(0.5, 0.5, f"Not enough numeric data in '{y_col}' to build a stacked bar chart",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        x_positions = list(range(len(unique_groups)))

        if hue_col:
            # One segment per hue level, each segment = sum of y_col for that
            # (x_col, hue level) combination.
            hue_levels = list(df_clean[hue_col].dropna().unique())
            colors = resolved_palette if len(resolved_palette) >= len(hue_levels) else \
                sns.color_palette("muted", n_colors=max(len(hue_levels), 1))

            bottoms = [0.0] * len(unique_groups)
            for h_idx, h_level in enumerate(hue_levels):
                sub = df_clean[df_clean[hue_col] == h_level]
                heights = [sub[sub[x_col] == g][y_col].sum() for g in unique_groups]
                bars = ax.bar(x_positions, heights, bottom=bottoms, width=0.6,
                               color=colors[h_idx], label=str(h_level),
                               edgecolor=text_color, linewidth=0.6)
                if bw_mode:
                    for b in bars:
                        b.set_hatch(hatch_cycle[h_idx % len(hatch_cycle)])
                bottoms = [b + h for b, h in zip(bottoms, heights)]

            legend = ax.legend(title=hue_col, fontsize=8, title_fontsize=9)
            if legend:
                for text_obj, lbl in zip(legend.get_texts(), hue_levels):
                    text_obj.set_picker(True)
                    text_obj.set_url("group_color:" + str(lbl))
        else:
            # No hue column: stack each row within a category as its own
            # segment, so e.g. 2 replicate values for "a" become 2 segments
            # in one bar, totalling their sum. Segments are named "Value 1",
            # "Value 2"... and are individually recolorable via their legend
            # entry (group_colors is keyed by that same name).
            group_colors = group_colors or {}
            max_rows = int(df_clean.groupby(x_col).size().max())
            default_colors = sns.color_palette("muted", n_colors=max(max_rows, 1))
            segment_names = [f"Value {i + 1}" for i in range(max_rows)]
            colors = [group_colors.get(name, default_colors[i]) for i, name in enumerate(segment_names)]

            bottoms = [0.0] * len(unique_groups)
            for row_idx, seg_name in enumerate(segment_names):
                heights = []
                for g in unique_groups:
                    vals = df_clean[df_clean[x_col] == g][y_col].values
                    heights.append(vals[row_idx] if row_idx < len(vals) else 0.0)
                bars = ax.bar(x_positions, heights, bottom=bottoms, width=0.6,
                               color=colors[row_idx], label=seg_name,
                               edgecolor=text_color, linewidth=0.6)
                if bw_mode:
                    for b in bars:
                        b.set_hatch(hatch_cycle[row_idx % len(hatch_cycle)])
                bottoms = [b + h for b, h in zip(bottoms, heights)]

            if max_rows > 1:
                legend = ax.legend(fontsize=8)
                if legend:
                    for text_obj, seg_name in zip(legend.get_texts(), segment_names):
                        text_obj.set_picker(True)
                        text_obj.set_url("group_color:" + seg_name)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(g) for g in unique_groups], rotation=45,
                            horizontalalignment="right", fontsize=10)

        ax.set_title(f"Stacked total {y_col} by {x_col}" + (f" (by {hue_col})" if hue_col else ""),
                     fontsize=14, fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel(x_col, fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel(y_col, fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        sns.despine(ax=ax, left=True, bottom=True)

    @staticmethod
    def _resolve_continuous_cmap(palette, diverging=False):
        """The palette selector holds seaborn *qualitative* palette names
        (muted, deep, colorblind) alongside two that are also valid Matplotlib
        colormaps (rocket, crest). Heatmaps/networks need a continuous
        colormap, so only rocket/crest pass through as-is; anything else
        falls back to a sensible continuous default."""
        if palette in ("rocket", "crest"):
            return palette
        return "vlag" if diverging else "rocket"

    @staticmethod
    def _draw_heatmap(df, x_col, y_col, hue_col, ax, palette, text_color):
        """
        2D heatmap. If a Group/Color-By column is set, pivots x_col vs hue_col
        with the mean of y_col as the cell value. Otherwise, falls back to a
        multi-metric profile heatmap: every other numeric column found in the
        dataset (e.g. GFP/mKate/OD600) becomes its own z-scored column, one
        row per x_col group - handy for comparing strains/conditions across
        several fluorescence channels at once.
        """
        ax.grid(False)
        df_clean = df.copy()
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")

        diverging = False
        if hue_col:
            pivot = df_clean.pivot_table(index=x_col, columns=hue_col, values=y_col, aggfunc="mean")
            title = f"Mean {y_col} — {x_col} x {hue_col}"
            cbar_label = f"Mean {y_col}"
            fmt = ".2g"
        else:
            numeric_cols = [c for c in df_clean.columns
                            if c != x_col and pd.api.types.is_numeric_dtype(df_clean[c])]
            if len(numeric_cols) >= 2:
                grouped = df_clean.groupby(x_col)[numeric_cols].mean()
                stds = grouped.std(ddof=0).replace(0, 1)
                pivot = (grouped - grouped.mean()) / stds
                title = f"Metric Profile Heatmap by {x_col} (z-scored)"
                cbar_label = "z-score"
                fmt = ".2f"
                diverging = True
            else:
                pivot = df_clean.groupby(x_col)[[y_col]].mean()
                title = f"Mean {y_col} by {x_col}"
                cbar_label = f"Mean {y_col}"
                fmt = ".2g"

        if pivot.empty or pivot.isna().all().all():
            ax.text(0.5, 0.5, "Not enough numeric data to build a heatmap",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        cmap = PlottingEngine._resolve_continuous_cmap(palette, diverging=diverging)
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=fmt, linewidths=0.5,
                    linecolor=text_color, cbar_kws={"label": cbar_label},
                    annot_kws={"fontsize": 8})

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel(pivot.columns.name or "", fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel(pivot.index.name or x_col, fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color, rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    @staticmethod
    def _draw_volcano(df, x_col, y_col, hue_col, ax, text_color, grid_color):
        """
        Standard volcano plot: X = log2 fold-change (numeric column chosen
        via the X Axis selector), Y = raw p-value (numeric column chosen via
        the Metric selector), rendered as -log10(p). Points are classified as
        Up/Down/Not-significant using the conventional |log2FC| >= 1 and
        p < 0.05 thresholds. If a Group/Color-By column is set, it doubles as
        a label column (e.g. gene/feature name) and the most significant
        points are annotated with it.
        """
        ax.grid(True, linestyle="--", alpha=0.5, color=grid_color)

        cols = [x_col, y_col] + ([hue_col] if hue_col else [])
        d = df[cols].copy()
        d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        d = d[d[y_col] > 0]

        if d.empty:
            ax.text(0.5, 0.5,
                    f"'{x_col}' and/or '{y_col}' aren't usable as numeric values.\n"
                    "Volcano needs a numeric log2 fold-change column (X Axis)\n"
                    "and a numeric p-value column (Metric/Y Axis), both > 0.",
                    ha="center", va="center", transform=ax.transAxes, color=text_color,
                    fontsize=10, wrap=True)
            return

        fc_thresh, p_thresh = 1.0, 0.05
        d["_neglog10p"] = -np.log10(d[y_col])

        conditions = [
            (d[y_col] < p_thresh) & (d[x_col] >= fc_thresh),
            (d[y_col] < p_thresh) & (d[x_col] <= -fc_thresh),
        ]
        d["_class"] = np.select(conditions, ["Up", "Down"], default="NS")
        color_map = {"Up": "#D62728", "Down": "#1F77B4", "NS": "#9AA0A6"}

        for cls in ["NS", "Down", "Up"]:  # draw NS first so hits sit on top
            sub = d[d["_class"] == cls]
            if sub.empty:
                continue
            ax.scatter(sub[x_col], sub["_neglog10p"], s=45, alpha=0.75,
                      color=color_map[cls], edgecolor="none", label=f"{cls} (n={len(sub)})")

        ax.axhline(-np.log10(p_thresh), color=text_color, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(fc_thresh, color=text_color, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(-fc_thresh, color=text_color, linestyle="--", linewidth=1, alpha=0.6)

        if hue_col:
            top_hits = d[d["_class"] != "NS"].nlargest(8, "_neglog10p")
            for _, row in top_hits.iterrows():
                ax.annotate(str(row[hue_col]), (row[x_col], row["_neglog10p"]),
                           fontsize=7, color=text_color, xytext=(3, 3), textcoords="offset points")

        ax.set_title(f"Volcano Plot — {y_col} vs {x_col}", fontsize=14, fontweight="bold",
                    pad=15, color=text_color)
        ax.set_xlabel(f"{x_col} (log2 fold change)", fontsize=12, fontweight="bold", color=text_color)
        ax.set_ylabel(f"-log10({y_col})", fontsize=12, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        legend = ax.legend(fontsize=8)
        if legend:
            for text in legend.get_texts():
                text.set_color(text_color)
        sns.despine(ax=ax)

    @staticmethod
    def _draw_dendrogram(df, x_col, y_col, ax, palette, text_color):
        """
        Hierarchical-clustering dendrogram of x_col groups, based on their
        mean profile across every numeric metric column available (z-scored
        so different units/scales contribute comparably). This is a
        similarity tree over the *numeric plate-reader data*, not a
        sequence-based phylogenetic tree - there's no alignment or
        nucleotide/protein distance model behind it, which the caption below
        the plot makes explicit.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist

        ax.grid(False)

        numeric_cols = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
        if y_col in df.columns and y_col not in numeric_cols:
            numeric_cols = [y_col] + numeric_cols
        numeric_cols = list(dict.fromkeys(numeric_cols))

        if not numeric_cols:
            ax.text(0.5, 0.5, "No numeric metric columns found to cluster on",
                    ha="center", va="center", transform=ax.transAxes, color=text_color)
            return

        grouped = df.groupby(x_col)[numeric_cols].mean().dropna()
        if grouped.shape[0] < 3:
            ax.text(0.5, 0.5,
                    f"Need at least 3 groups in '{x_col}' to build a dendrogram\n(found {grouped.shape[0]})",
                    ha="center", va="center", transform=ax.transAxes, color=text_color, fontsize=10, wrap=True)
            return

        stds = grouped.std(ddof=0).replace(0, 1)
        z = (grouped - grouped.mean()) / stds
        dist = pdist(z.values, metric="euclidean")
        link = linkage(dist, method="average")

        with plt.rc_context({"lines.linewidth": 1.4}):
            dendrogram(link, labels=grouped.index.astype(str).tolist(), ax=ax,
                      color_threshold=0, above_threshold_color=text_color, orientation="right")

        ax.set_title(f"Hierarchical Clustering Dendrogram ({x_col} groups)", fontsize=13,
                    fontweight="bold", pad=15, color=text_color)
        ax.set_xlabel("Distance (z-scored Euclidean, average linkage)", fontsize=10,
                      fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.99, 0.02,
                "Clustered by numeric profile similarity — not a\nsequence-based phylogenetic tree",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=text_color,
                alpha=0.75, style="italic")

    @staticmethod
    def _draw_network(df, x_col, y_col, ax, palette, text_color):
        """
        Correlation network between the dataset's numeric metric columns
        (e.g. OD600/GFP/mKate). Nodes are placed on a circle; edges are drawn
        between pairs whose |Pearson r| clears a threshold, colored blue for
        positive and red for negative correlation, with width/opacity scaled
        to |r|. Built with plain numpy/matplotlib (no networkx dependency).
        """
        ax.grid(False)

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if y_col in df.columns and y_col not in numeric_cols:
            numeric_cols.append(y_col)
        numeric_cols = list(dict.fromkeys(numeric_cols))

        if len(numeric_cols) < 3:
            ax.text(0.5, 0.5,
                    "Need at least 3 numeric metric columns to build a correlation network\n"
                    f"(found {len(numeric_cols)}: {', '.join(numeric_cols) if numeric_cols else 'none'})",
                    ha="center", va="center", transform=ax.transAxes, color=text_color, fontsize=10, wrap=True)
            return

        corr = df[numeric_cols].corr(method="pearson")
        n = len(numeric_cols)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {col: (np.cos(a), np.sin(a)) for col, a in zip(numeric_cols, angles)}

        edge_thresh = 0.3
        for i in range(n):
            for j in range(i + 1, n):
                r = corr.iloc[i, j]
                if pd.isna(r) or abs(r) < edge_thresh:
                    continue
                x1, y1 = pos[numeric_cols[i]]
                x2, y2 = pos[numeric_cols[j]]
                edge_color = "#D62728" if r < 0 else "#1F77B4"
                ax.plot([x1, x2], [y1, y2], color=edge_color, alpha=min(abs(r), 0.9),
                       linewidth=1 + abs(r) * 4, zorder=1)

        node_colors = sns.color_palette(palette, n_colors=n)
        for idx, col in enumerate(numeric_cols):
            x0, y0 = pos[col]
            ax.scatter([x0], [y0], s=900, color=node_colors[idx], edgecolor=text_color,
                      linewidth=1.2, zorder=2)
            ax.text(x0 * 1.25, y0 * 1.25, col, ha="center", va="center", fontsize=9,
                   fontweight="bold", color=text_color, zorder=3)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Metric Correlation Network (|r| \u2265 {edge_thresh})", fontsize=14,
                    fontweight="bold", pad=15, color=text_color)

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="#1F77B4", lw=3, label="Positive correlation"),
            Line2D([0], [0], color="#D62728", lw=3, label="Negative correlation"),
        ]
        legend = ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)
        if legend:
            for text in legend.get_texts():
                text.set_color(text_color)

    # Dodge width used by each chart type when a Color-By column is set
    # (where each colour's bar/box sits inside its x group)
    DODGE_WIDTH = {"Bar": 0.8, "Boxplot": 0.5, "Violin": 0.8,
                   "Beeswarm": 0.8, "Box + Swarm": 0.8}
    MAX_BRACKETS = 15

    @staticmethod
    def _artist_tops(ax, n_groups) -> dict:
        """{x position: highest y drawn there} over everything on the axes
        (violins reach past the data points; bars carry error bars)."""
        from matplotlib.collections import PathCollection
        xs, ys = [], []
        for coll in ax.collections:
            if coll.get_transform() != ax.transData and \
                    not isinstance(coll, PathCollection):
                continue
            if isinstance(coll, PathCollection):
                off = np.asarray(coll.get_offsets())
                if off.size:
                    xs.append(off[:, 0]); ys.append(off[:, 1])
            else:
                for path in coll.get_paths():
                    v = path.vertices
                    if len(v):
                        i = np.argmax(v[:, 1])
                        xs.append([v[:, 0].mean()]); ys.append([v[i, 1]])
        for line in ax.lines:
            xy = np.asarray(line.get_xydata(), float)
            if xy.size:
                xs.append(xy[:, 0]); ys.append(xy[:, 1])
        for patch in ax.patches:
            try:
                bb = patch.get_bbox()
                xs.append([(bb.x0 + bb.x1) / 2]); ys.append([bb.y1])
            except Exception:
                continue
        if not xs:
            return {}
        x = np.concatenate([np.ravel(a) for a in xs]).astype(float)
        y = np.concatenate([np.ravel(a) for a in ys]).astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        out = {}
        for i in range(n_groups):
            m = np.abs(x - i) < 0.5
            if m.any():
                out[i] = float(y[m].max())
        return out

    @staticmethod
    def annotate_significance(df, x_col, y_col, groups, ax, stat_mode, line_color,
                               reference=None, pairing_col=None, log_scale=False,
                               show_pvalue=False, hue_col=None, hue_order=None,
                               dodge_width=0.8, show_ns=False, bar_tops=None):
        """
        Draws significance brackets from StatisticsEngine.compare() — the
        same results the Statistical Analysis tab reports. Without a
        Color-By column the x groups are compared; with one, the colours are
        compared inside each x group (Holm-corrected across groups).
        Brackets are stacked from the shortest up, each just above the data
        it spans. Returns the comparisons drawn.
        """
        SE = StatisticsEngine
        d = df.copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])
        if d.empty:
            return []
        d["_x"] = d[x_col].astype(str)
        groups = [str(g) for g in groups]
        xpos = {g: i for i, g in enumerate(groups)}

        if hue_col:
            d["_h"] = d[hue_col].astype(str)
            levels = [str(h) for h in (hue_order or d["_h"].unique())]
            n = max(len(levels), 1)
            step = dodge_width / n

            def pos(c):
                base = xpos[c.within]
                return (base + (levels.index(c.a) - (n - 1) / 2) * step,
                        base + (levels.index(c.b) - (n - 1) / 2) * step)
            results = SE.compare_within(d, "_x", "_h", y_col, stat_mode,
                                        reference, pairing_col, levels)
            comps = [c for r in results.values() for c in r.comparisons
                     if c.within in xpos]
        else:
            r = SE.compare(d, "_x", y_col, stat_mode, reference, pairing_col,
                           group_order=groups)
            comps = [c for c in r.comparisons if c.a in xpos and c.b in xpos]

            def pos(c):
                return xpos[c.a], xpos[c.b]

        shown = [c for c in comps if show_ns or c.significant]
        if not shown:
            return []

        # work in display units (log10 on a log axis)
        to_u = (lambda v: np.log10(v)) if log_scale else (lambda v: v)
        from_u = (lambda u: 10 ** u) if log_scale else (lambda u: u)
        vals = d[y_col].to_numpy(float)
        if log_scale:
            vals = vals[vals > 0]
            if vals.size == 0:
                return []
        lo, hi = to_u(vals.min()), to_u(vals.max())
        if bar_tops:
            hi = max(hi, max(to_u(v) for v in bar_tops.values() if v > 0)
                     if any(v > 0 for v in bar_tops.values()) else hi)
        span = (hi - lo) or 1.0
        step_u = span * (0.13 if show_pvalue else 0.10)
        tick = step_u * 0.18

        drawn_tops = PlottingEngine._artist_tops(ax, len(groups))
        if drawn_tops:
            hi = max([hi] + [to_u(v) for v in drawn_tops.values() if v > 0])
            span = (hi - lo) or 1.0
            step_u = span * (0.13 if show_pvalue else 0.10)
            tick = step_u * 0.18

        def top_of(x1, x2):
            """Highest thing drawn (points, bar + error, box, violin)
            between two x positions."""
            a, b = sorted((x1, x2))
            gs = [g for g, i in xpos.items() if a - 0.5 < i < b + 0.5]
            sub = d[d["_x"].isin(gs)][y_col].to_numpy(float)
            if log_scale:
                sub = sub[sub > 0]
            t = to_u(sub.max()) if sub.size else hi
            for g in gs:
                v = drawn_tops.get(xpos[g])
                if v is not None and (v > 0 or not log_scale):
                    t = max(t, to_u(v))
            if bar_tops:
                t = max([t] + [to_u(bar_tops[g]) for g in gs
                               if bar_tops.get(g, 0) > 0])
            return t

        placed = []                         # (x1, x2, level)
        shown.sort(key=lambda c: abs(pos(c)[1] - pos(c)[0]))
        drawn = []
        for c in shown[:PlottingEngine.MAX_BRACKETS]:
            x1, x2 = sorted(pos(c))
            level = top_of(x1, x2) + step_u * 0.6
            for (a, b, lev) in placed:      # stack above overlapping ones
                if a - 0.05 <= x2 and x1 <= b + 0.05:
                    level = max(level, lev + step_u)
            placed.append((x1, x2, level))
            label = stars(c.p)
            if show_pvalue:
                label = f"{label} ({format_p(c.p)})" if label != "ns" \
                    else format_p(c.p)
            ys = [from_u(level), from_u(level + tick)]
            ax.plot([x1, x1, x2, x2], [ys[0], ys[1], ys[1], ys[0]], lw=1.1,
                    color=line_color, solid_capstyle="butt")
            ax.text((x1 + x2) / 2, from_u(level + tick), label, ha="center",
                    va="bottom", color=line_color,
                    fontsize=8.5 if show_pvalue else 10.5,
                    fontweight="bold" if label != "ns" else "normal")
            drawn.append(c)
        top = max(lev for _, _, lev in placed) + step_u
        extra = len(shown) - len(drawn)
        if extra > 0:
            ax.text(0.99, 0.99, f"+{extra} more significant comparison"
                    f"{'s' if extra != 1 else ''} — see Statistical Analysis",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=line_color)
        bottom = ax.get_ylim()[0]
        ax.set_ylim(bottom=bottom, top=from_u(top))
        return drawn
