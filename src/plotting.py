# Revenue guidance vs actual charting utilities.
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def prepare_revenue_plot_df(
    df: pd.DataFrame,
    fiscal_year_col: str = 'fiscal_year',
    fiscal_quarter_col: str = 'fiscal_quarter',
    target_min_col: str = 'revenue_target_min',
    target_max_col: str = 'revenue_target_max',
    actual_col: str = 'actual_revenue',
    convert_to_billion: bool = True
) -> pd.DataFrame:
    # Build a clean analysis dataframe with status, gap, band, and display columns for revenue interval charts.

    data = df.copy()

    def parse_quarter(x):
        # Parse fiscal quarter from int, float, or string to 1–4.
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            q = int(x)
            if q in [1, 2, 3, 4]:
                return q
        s = str(x).strip().upper()
        m = re.search(r'([1-4])', s)
        if m:
            return int(m.group(1))
        raise ValueError(f"Cannot parse fiscal quarter: {x}")

    data[fiscal_year_col] = pd.to_numeric(data[fiscal_year_col], errors='coerce')
    data['_quarter_num'] = data[fiscal_quarter_col].apply(parse_quarter)

    data[target_min_col] = pd.to_numeric(data[target_min_col], errors='coerce')
    data[target_max_col] = pd.to_numeric(data[target_max_col], errors='coerce')
    data[actual_col] = pd.to_numeric(data[actual_col], errors='coerce')

    data = data.sort_values([fiscal_year_col, '_quarter_num']).reset_index(drop=True)

    data['revenue_target_min'] = np.fmin(data[target_min_col], data[target_max_col])
    data['revenue_target_max'] = np.fmax(data[target_min_col], data[target_max_col])
    data['actual_revenue'] = data[actual_col]

    data['status'] = np.where(
        data['actual_revenue'] < data['revenue_target_min'], 'below',
        np.where(data['actual_revenue'] > data['revenue_target_max'], 'above', 'within')
    )

    data['gap_to_min'] = data['actual_revenue'] - data['revenue_target_min']
    data['gap_to_max'] = data['actual_revenue'] - data['revenue_target_max']
    data['gap_to_interval'] = np.where(
        data['actual_revenue'] < data['revenue_target_min'],
        data['actual_revenue'] - data['revenue_target_min'],
        np.where(
            data['actual_revenue'] > data['revenue_target_max'],
            data['actual_revenue'] - data['revenue_target_max'],
            0.0
        )
    )

    data['band_width'] = data['revenue_target_max'] - data['revenue_target_min']
    data['band_mid'] = (data['revenue_target_min'] + data['revenue_target_max']) / 2.0
    data['actual_vs_mid'] = data['actual_revenue'] - data['band_mid']
    data['actual_vs_mid_pct'] = np.where(
        data['band_mid'] != 0,
        data['actual_vs_mid'] / data['band_mid'],
        np.nan
    )

    scale = 1000.0 if convert_to_billion else 1.0
    data['display_target_min'] = data['revenue_target_min'] / scale
    data['display_target_max'] = data['revenue_target_max'] / scale
    data['display_actual'] = data['actual_revenue'] / scale
    data['display_gap_to_interval'] = data['gap_to_interval'] / scale

    data['quarter_num'] = data['_quarter_num']
    data['quarter_label'] = 'Q' + data['quarter_num'].astype('Int64').astype(str)

    result = data[[
        fiscal_year_col,
        fiscal_quarter_col,
        'quarter_num',
        'quarter_label',
        'revenue_target_min',
        'revenue_target_max',
        'actual_revenue',
        'status',
        'gap_to_min',
        'gap_to_max',
        'gap_to_interval',
        'band_width',
        'band_mid',
        'actual_vs_mid',
        'actual_vs_mid_pct',
        'display_target_min',
        'display_target_max',
        'display_actual',
        'display_gap_to_interval'
    ]].rename(columns={
        fiscal_year_col: 'fiscal_year',
        fiscal_quarter_col: 'fiscal_quarter'
    })

    return result

def add_initial_interval_to_plot_df(
    revenue_band_df: pd.DataFrame,
    guidance_m_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    convert_to_billion: bool = True
) -> pd.DataFrame:
    # Join the earliest guidance interval for each target year into the plot dataframe.

    g = guidance_m_df.copy()

    def parse_quarter(x):
        # Parse fiscal quarter from int, float, or string to 1–4.
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip().upper()
        m = re.search(r'([1-4])', s)
        if m:
            return int(m.group(1))
        raise ValueError(f"Cannot parse quarter: {x}")

    if 'quarter_num' not in g.columns:
        g['quarter_num'] = g['fiscal_quarter'].apply(parse_quarter)

    if 'release_date' in g.columns:
        g['_sort_date'] = pd.to_datetime(g['release_date'].astype(str), format='%Y%m%d', errors='coerce')
        g = g.sort_values(['target_year', '_sort_date', 'fiscal_year', 'quarter_num'])
    else:
        g = g.sort_values(['target_year', 'fiscal_year', 'quarter_num'])

    initial_guidance = (
        g.groupby('target_year', as_index=False)
         .first()[['target_year', 'revenue_min_M', 'revenue_max_M']]
         .rename(columns={
             'revenue_min_M': 'initial_revenue_min_M',
             'revenue_max_M': 'initial_revenue_max_M'
         })
    )

    share_df = revenue_band_df[['fiscal_year', 'fiscal_quarter', 'revenue_share_used']].copy()

    share_df = share_df.merge(
        initial_guidance,
        left_on='fiscal_year',
        right_on='target_year',
        how='left'
    )

    share_df['initial_revenue_target_min'] = (
        share_df['initial_revenue_min_M'] * share_df['revenue_share_used']
    )
    share_df['initial_revenue_target_max'] = (
        share_df['initial_revenue_max_M'] * share_df['revenue_share_used']
    )

    scale = 1000.0 if convert_to_billion else 1.0
    share_df['display_initial_target_min'] = share_df['initial_revenue_target_min'] / scale
    share_df['display_initial_target_max'] = share_df['initial_revenue_target_max'] / scale

    out = plot_df.merge(
        share_df[[
            'fiscal_year', 'fiscal_quarter',
            'initial_revenue_target_min', 'initial_revenue_target_max',
            'display_initial_target_min', 'display_initial_target_max'
        ]],
        on=['fiscal_year', 'fiscal_quarter'],
        how='left'
    )

    return out

if __name__ == "__main__":
    import pandas as pd
    from src.analysis import normalize_guidance_to_million, build_revenue_quarterly_target_band

    df_filling = pd.read_csv('../data/revenue_filling.csv')
    df_extract_guidance = pd.read_csv('../data/revenue_extract_guidance.csv')

    result_df = normalize_guidance_to_million(df_filling, df_extract_guidance)
    revenue_band_df = build_revenue_quarterly_target_band(df_filling, result_df)

    print("Prepared revenue band dataframe:")
    print(revenue_band_df.head())


def plot_revenue_interval_actual_by_quarter(
    plot_df: pd.DataFrame,
    figsize=(18, 5.4),
    sharey=False,
    annotate_gap=True,
    suptitle='Revenue vs Initial and Latest Target Intervals by Quarter',
    ylabel='Revenue ($ in Billions)',
    initial_band_color='deepskyblue',
    latest_band_color='violet',
    actual_line_color='tab:gray',
    initial_band_alpha=0.5,
    latest_band_alpha=0.5,
    band_half_width=0.16,
    point_size=60,
    annotation_fontsize=11,
    y_pad_ratio=0.18,
    extra_top_pad_ratio=0.10,
):
    # Plot revenue actual vs initial and latest guidance bands across four quarter subplots.

    required = {
        'fiscal_year', 'fiscal_quarter', 'quarter_num', 'quarter_label',
        'status',
        'display_target_min', 'display_target_max', 'display_actual',
        'display_gap_to_interval',
        'display_initial_target_min', 'display_initial_target_max'
    }
    missing = required - set(plot_df.columns)
    if missing:
        raise ValueError(f"plot_df missing columns: {sorted(missing)}")

    data = plot_df.copy().sort_values(['quarter_num', 'fiscal_year']).reset_index(drop=True)

    quarter_order = [1, 2, 3, 4]
    quarter_name = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
    status_color = {'within': 'tab:green', 'below': 'tab:red', 'above': 'tab:orange'}

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=sharey)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    if sharey:
        all_y = np.r_[
            data['display_initial_target_min'].dropna().values,
            data['display_initial_target_max'].dropna().values,
            data['display_target_min'].dropna().values,
            data['display_target_max'].dropna().values,
            data['display_actual'].dropna().values
        ]
        y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
        span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0) * 0.1
        y_pad = span * y_pad_ratio
        extra_top = span * extra_top_pad_ratio if annotate_gap else 0.0
        global_ylim = (y_min - y_pad, y_max + y_pad + extra_top)
    else:
        global_ylim = None

    for ax, q in zip(axes, quarter_order):
        d = data[data['quarter_num'] == q].copy().sort_values('fiscal_year').reset_index(drop=True)

        if d.empty:
            ax.set_title(quarter_name[q])
            ax.set_xticks([])
            ax.grid(True, alpha=0.3)
            continue

        d['_x'] = np.arange(len(d))

        for _, row in d.iterrows():
            x0 = row['_x'] - band_half_width
            x1 = row['_x'] + band_half_width

            ax.fill_between(
                [x0, x1],
                [row['display_initial_target_min'], row['display_initial_target_min']],
                [row['display_initial_target_max'], row['display_initial_target_max']],
                color=initial_band_color,
                alpha=initial_band_alpha,
                zorder=0.8
            )

            ax.fill_between(
                [x0, x1],
                [row['display_target_min'], row['display_target_min']],
                [row['display_target_max'], row['display_target_max']],
                color=latest_band_color,
                alpha=latest_band_alpha,
                zorder=1.1
            )

        ax.plot(
            d['_x'],
            d['display_actual'],
            color=actual_line_color,
            linewidth=1.8,
            alpha=0.9,
            marker='o',
            markersize=4.5,
            zorder=2
        )

        for _, row in d.iterrows():
            ax.scatter(
                row['_x'],
                row['display_actual'],
                color=status_color.get(row['status'], 'tab:gray'),
                s=point_size,
                zorder=3
            )

            if annotate_gap and row['status'] != 'within':
                ax.annotate(
                    f"{row['display_gap_to_interval']:+.2f}",
                    (row['_x'], row['display_actual']),
                    textcoords='offset points',
                    xytext=(0, 9),
                    ha='center',
                    va='bottom',
                    fontsize=annotation_fontsize,
                    clip_on=False,
                    color='#444444'
                )

        ax.set_title(quarter_name[q], fontsize=14)
        ax.set_xticks(d['_x'])
        ax.set_xticklabels(d['fiscal_year'].astype('Int64').astype(str), rotation=45, ha='right')
        ax.set_xlabel('Fiscal Year')
        ax.grid(True, alpha=0.3)

        if global_ylim is not None:
            ax.set_ylim(global_ylim)
        else:
            local_y = np.r_[
                d['display_initial_target_min'].dropna().values,
                d['display_initial_target_max'].dropna().values,
                d['display_target_min'].dropna().values,
                d['display_target_max'].dropna().values,
                d['display_actual'].dropna().values
            ]
            y_min, y_max = np.nanmin(local_y), np.nanmax(local_y)
            span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0) * 0.1
            y_pad = span * y_pad_ratio
            extra_top = span * extra_top_pad_ratio if annotate_gap else 0.0
            ax.set_ylim(y_min - y_pad, y_max + y_pad + extra_top)

    axes[0].set_ylabel(ylabel)

    main_legend = [
        Patch(facecolor=initial_band_color, edgecolor='gray', alpha=initial_band_alpha, label='Initial Guidance Interval'),
        Patch(facecolor=latest_band_color, edgecolor=latest_band_color, alpha=latest_band_alpha, label='Latest Guidance Interval'),
        Line2D([0], [0], color=actual_line_color, marker='o', lw=1.8, label='Actual Revenue')
    ]

    status_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markersize=8, label='Within Latest'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markersize=8, label='Below Latest'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', markersize=8, label='Above Latest')
    ]

    leg1 = fig.legend(
        handles=main_legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=True
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=status_legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        title='Actual status'
    )

    fig.suptitle(suptitle, y=1.09, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    return fig, axes


def plot_revenue_gap_to_interval(
    plot_df: pd.DataFrame,
    by='quarter',   # 'quarter' or 'all'
    figsize=None,
    suptitle='Revenue Gap vs Target Interval',
    ylabel='Gap to Interval ($ in Billions)',
    annotation_fontsize=10,
    y_pad_ratio=0.25,
    extra_top_pad_ratio=0.15,
):
    # Plot the revenue gap to the latest guidance interval as a bar chart, split by quarter or combined.
    required = {
        'fiscal_year', 'fiscal_quarter', 'quarter_num', 'quarter_label',
        'status', 'display_gap_to_interval'
    }
    missing = required - set(plot_df.columns)
    if missing:
        raise ValueError(f"plot_df missing columns: {sorted(missing)}")

    data = plot_df.copy().sort_values(['quarter_num', 'fiscal_year']).reset_index(drop=True)

    def bar_color(v):
        # Return red for miss-low, orange for beat, green for on-target.
        if v < 0:
            return 'tab:red'
        if v > 0:
            return 'tab:orange'
        return 'tab:green'

    if by == 'all':
        if figsize is None:
            figsize = (14, 5)

        data = data.sort_values(['fiscal_year', 'quarter_num']).reset_index(drop=True)
        data['_x'] = np.arange(len(data))
        data['_label'] = data['fiscal_year'].astype('Int64').astype(str) + ' ' + data['quarter_label']
        colors = [bar_color(v) for v in data['display_gap_to_interval']]

        fig, ax = plt.subplots(figsize=figsize)

        fiscal_years = data['fiscal_year'].astype('Int64')
        unique_years = fiscal_years.unique()
        bg_colors = ["#ffe2e2", "#d4ecff", "#ffeed5", "#e0f8da", "#f7dae7"]
        for idx, year in enumerate(unique_years):
            year_mask = fiscal_years == year
            if not year_mask.any():
                continue
            x_start = data.loc[year_mask, '_x'].min() - 0.5
            x_end = data.loc[year_mask, '_x'].max() + 0.5
            ax.axvspan(
                x_start,
                x_end,
                color=bg_colors[idx % len(bg_colors)],
                alpha=0.5,
                zorder=0
            )

        ax.bar(data['_x'], data['display_gap_to_interval'], color=colors, alpha=0.85, zorder=2)
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', zorder=3)

        for _, row in data.iterrows():
            gap = row['display_gap_to_interval']
            if gap != 0:
                ax.annotate(
                    f"{gap:+.2f}",
                    (row['_x'], gap),
                    textcoords='offset points',
                    xytext=(0, 6 if gap >= 0 else -12),
                    ha='center',
                    fontsize=annotation_fontsize
                )

        ax.set_xticks(data['_x'])
        ax.set_xticklabels(data['_label'], rotation=45, ha='right')
        ax.set_xlabel('Fiscal Year & Quarter')
        ax.set_ylabel(ylabel)
        ax.set_title(suptitle)
        ax.grid(True, axis='y', alpha=0.3)

        all_y = data['display_gap_to_interval'].dropna().values
        y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
        span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0) * 0.1
        y_pad = span * y_pad_ratio
        extra_top = span * extra_top_pad_ratio
        ax.set_ylim(y_min - y_pad, y_max + y_pad + extra_top)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color='tab:red', label='Below'),
            plt.Rectangle((0, 0), 1, 1, color='tab:green', label='Within'),
            plt.Rectangle((0, 0), 1, 1, color='tab:orange', label='Above')
        ]
        ax.legend(handles=handles, loc='upper right')

        plt.tight_layout()
        return fig, ax

    elif by == 'quarter':
        if figsize is None:
            figsize = (18, 4.8)

        quarter_order = [1, 2, 3, 4]
        quarter_name = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}

        fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, q in zip(axes, quarter_order):
            d = data[data['quarter_num'] == q].copy().sort_values('fiscal_year').reset_index(drop=True)

            if d.empty:
                ax.set_title(quarter_name[q])
                ax.set_xticks([])
                ax.grid(True, axis='y', alpha=0.3)
                continue

            d['_x'] = np.arange(len(d))
            colors = [bar_color(v) for v in d['display_gap_to_interval']]

            ax.bar(d['_x'], d['display_gap_to_interval'], color=colors, alpha=0.85)
            ax.axhline(0, color='black', linewidth=1.2, linestyle='--')

            for _, row in d.iterrows():
                gap = row['display_gap_to_interval']
                if gap != 0:
                    ax.annotate(
                        f"{gap:+.2f}",
                        (row['_x'], gap),
                        textcoords='offset points',
                        xytext=(0, 6 if gap >= 0 else -12),
                        ha='center',
                        fontsize=annotation_fontsize
                    )

            ax.set_title(quarter_name[q], fontsize=14)
            ax.set_xticks(d['_x'])
            ax.set_xticklabels(d['fiscal_year'].astype('Int64').astype(str), rotation=45, ha='right')
            ax.set_xlabel('Fiscal Year')
            ax.grid(True, axis='y', alpha=0.3)

            local_y = d['display_gap_to_interval'].dropna().values
            y_min, y_max = np.nanmin(local_y), np.nanmax(local_y)
            span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0) * 0.1
            y_pad = span * y_pad_ratio
            extra_top = span * extra_top_pad_ratio
            ax.set_ylim(y_min - y_pad, y_max + y_pad + extra_top)

        axes[0].set_ylabel(ylabel)

        handles = [
            plt.Rectangle((0,0),1,1, color='tab:red', label='Below'),
            plt.Rectangle((0,0),1,1, color='tab:green', label='Within'),
            plt.Rectangle((0,0),1,1, color='tab:orange', label='Above')
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3)

        fig.suptitle(suptitle, y=1.03, fontsize=16)
        plt.tight_layout()
        return fig, axes

    else:
        raise ValueError("by must be 'quarter' or 'all'")

if __name__ == "__main__":
    import pandas as pd
    from src.analysis import normalize_guidance_to_million, build_revenue_quarterly_target_band
    from src.plotting import prepare_revenue_plot_df, add_initial_interval_to_plot_df

    df_filling = pd.read_csv('../data/revenue_filling.csv')
    df_extract_guidance = pd.read_csv('../data/revenue_extract_guidance.csv')

    result_df = normalize_guidance_to_million(df_filling, df_extract_guidance)
    revenue_band_df = build_revenue_quarterly_target_band(df_filling, result_df)

    plot_df = prepare_revenue_plot_df(revenue_band_df, convert_to_billion=True)
    plot_df = add_initial_interval_to_plot_df(revenue_band_df, result_df, plot_df, convert_to_billion=True)

    fig1, axes1 = plot_revenue_interval_actual_by_quarter(plot_df)
    fig2, axes2 = plot_revenue_gap_to_interval(plot_df, by='quarter')
