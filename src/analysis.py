# Guidance normalization, quarterly target band construction, and revenue revision tracking.
import pandas as pd
import numpy as np
import re

def normalize_guidance_to_million(actual_df: pd.DataFrame,
                                  guidance_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize raw guidance values to absolute million-dollar amounts using prior-year actuals as base.
    actual_required = {'fiscal_year', 'fiscal_quarter', 'revenue_ytd_M', 'fcf_ytd_M'}
    guidance_required = {
        'fiscal_year', 'fiscal_quarter',
        'revenue_min', 'revenue_max', 'revenue_unit',
        'fcf_min', 'fcf_max', 'fcf_unit'
    }

    missing_actual = actual_required - set(actual_df.columns)
    missing_guidance = guidance_required - set(guidance_df.columns)

    if missing_actual:
        raise ValueError(f"actual_df missing columns: {sorted(missing_actual)}")
    if missing_guidance:
        raise ValueError(f"guidance_df missing columns: {sorted(missing_guidance)}")

    actual = actual_df.copy()
    guidance = guidance_df.copy()

    def parse_quarter(x):
        # Parse fiscal quarter from int, float, or string to 1–4.
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            q = int(x)
            if q in [1, 2, 3, 4]:
                return q
        if isinstance(x, float) and not np.isnan(x):
            q = int(x)
            if q in [1, 2, 3, 4]:
                return q
        s = str(x).strip().upper()
        m = re.search(r'([1-4])', s)
        if m:
            return int(m.group(1))
        raise ValueError(f"Cannot parse fiscal_quarter value: {x}")

    PCT_UNITS = {'%', 'PERCENT', 'PERCENTAGE', 'PCT'}
    M_UNITS   = {'$M', 'M', 'MM', 'MILLION', 'MILLIONS', 'USDM'}
    B_UNITS   = {'$B', 'B', 'BN', 'BILLION', 'BILLIONS', 'USDB'}
    ONE_UNITS = {'1', '$', 'USD', 'DOLLAR', 'DOLLARS'}

    def normalize_unit(unit):
        # Strip and uppercase a unit string.
        if pd.isna(unit):
            return None
        return str(unit).strip().upper()

    def convert_to_million(value, unit, base_year_end_M):
        # Convert one guidance value to absolute million-dollar value.
        if pd.isna(value):
            return np.nan

        u = normalize_unit(unit)
        v = float(value)

        if u in PCT_UNITS:
            if pd.isna(base_year_end_M):
                return np.nan
            return base_year_end_M * (1.0 + v / 100.0)

        if u in M_UNITS:
            return v

        if u in B_UNITS:
            return v * 1000.0

        if u in ONE_UNITS:
            return v / 1_000_000.0

        raise ValueError(f"Unsupported unit: {unit}")

    actual['fiscal_year'] = actual['fiscal_year'].astype(int)
    actual['quarter_num'] = actual['fiscal_quarter'].apply(parse_quarter).astype(int)

    actual['revenue_ytd_M'] = pd.to_numeric(actual['revenue_ytd_M'], errors='coerce')
    actual['fcf_ytd_M'] = pd.to_numeric(actual['fcf_ytd_M'], errors='coerce')

    actual = actual.sort_values(['fiscal_year', 'quarter_num']).reset_index(drop=True)

    # Prefer Q4 as year-end; fall back to latest available quarter if Q4 is missing.
    year_end_actual = (
        actual.sort_values(['fiscal_year', 'quarter_num'])
              .groupby('fiscal_year', as_index=False)
              .tail(1)[['fiscal_year', 'revenue_ytd_M', 'fcf_ytd_M']]
              .rename(columns={
                  'fiscal_year': 'base_year',
                  'revenue_ytd_M': 'base_revenue_year_end_M',
                  'fcf_ytd_M': 'base_fcf_year_end_M'
              })
    )

    guidance['fiscal_year'] = guidance['fiscal_year'].astype(int)
    guidance['quarter_num'] = guidance['fiscal_quarter'].apply(parse_quarter).astype(int)

    guidance['target_year'] = np.where(
        guidance['quarter_num'] == 4,
        guidance['fiscal_year'] + 1,
        guidance['fiscal_year']
    ).astype(int)

    guidance['base_year'] = guidance['target_year'] - 1

    out = guidance.merge(year_end_actual, on='base_year', how='left')

    out['revenue_min_M'] = out.apply(
        lambda r: convert_to_million(
            r['revenue_min'],
            r['revenue_unit'],
            r['base_revenue_year_end_M']
        ),
        axis=1
    )
    out['revenue_max_M'] = out.apply(
        lambda r: convert_to_million(
            r['revenue_max'],
            r['revenue_unit'],
            r['base_revenue_year_end_M']
        ),
        axis=1
    )

    out['fcf_min_M'] = out.apply(
        lambda r: convert_to_million(
            r['fcf_min'],
            r['fcf_unit'],
            r['base_fcf_year_end_M']
        ),
        axis=1
    )
    out['fcf_max_M'] = out.apply(
        lambda r: convert_to_million(
            r['fcf_max'],
            r['fcf_unit'],
            r['base_fcf_year_end_M']
        ),
        axis=1
    )

    out['revenue_min_M'], out['revenue_max_M'] = (
        np.fmin(out['revenue_min_M'], out['revenue_max_M']),
        np.fmax(out['revenue_min_M'], out['revenue_max_M'])
    )

    out['fcf_min_M'], out['fcf_max_M'] = (
        np.fmin(out['fcf_min_M'], out['fcf_max_M']),
        np.fmax(out['fcf_min_M'], out['fcf_max_M'])
    )

    return out

def build_revenue_quarterly_target_band(actual_df: pd.DataFrame,
                                        guidance_m_df: pd.DataFrame,
                                        dedupe_same_effective_quarter: bool = True) -> pd.DataFrame:
    # Expand normalized year-end guidance into a quarter-level comparable revenue YTD target band.

    actual_required = {'fiscal_year', 'fiscal_quarter', 'revenue_ytd_M'}
    guidance_required = {
        'fiscal_year', 'fiscal_quarter', 'quarter_num',
        'target_year', 'revenue_min_M', 'revenue_max_M'
    }

    missing_actual = actual_required - set(actual_df.columns)
    missing_guidance = guidance_required - set(guidance_m_df.columns)

    if missing_actual:
        raise ValueError(f"actual_df missing columns: {sorted(missing_actual)}")
    if missing_guidance:
        raise ValueError(f"guidance_m_df missing columns: {sorted(missing_guidance)}")

    actual = actual_df.copy()
    guidance = guidance_m_df.copy()

    def parse_quarter(x):
        # Parse fiscal quarter from int, float, or string to 1–4.
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            q = int(x)
            if q in [1, 2, 3, 4]:
                return q
        if isinstance(x, float) and not np.isnan(x):
            q = int(x)
            if q in [1, 2, 3, 4]:
                return q
        s = str(x).strip().upper()
        m = re.search(r'([1-4])', s)
        if m:
            return int(m.group(1))
        raise ValueError(f"Cannot parse fiscal_quarter value: {x}")

    actual['fiscal_year'] = actual['fiscal_year'].astype(int)
    actual['quarter_num'] = actual['fiscal_quarter'].apply(parse_quarter).astype(int)
    actual['revenue_ytd_M'] = pd.to_numeric(actual['revenue_ytd_M'], errors='coerce')

    actual = actual.sort_values(['fiscal_year', 'quarter_num']).reset_index(drop=True)

    year_end_actual = (
        actual.sort_values(['fiscal_year', 'quarter_num'])
              .groupby('fiscal_year', as_index=False)
              .tail(1)[['fiscal_year', 'revenue_ytd_M']]
              .rename(columns={'revenue_ytd_M': 'revenue_year_end_M'})
    )

    actual = actual.merge(year_end_actual, on='fiscal_year', how='left')

    actual['prev_revenue_ytd_share'] = np.where(
        actual['revenue_year_end_M'] != 0,
        actual['revenue_ytd_M'] / actual['revenue_year_end_M'],
        np.nan
    )

    prev_share = actual[['fiscal_year', 'quarter_num', 'prev_revenue_ytd_share']].copy()
    prev_share['target_year'] = prev_share['fiscal_year'] + 1
    prev_share = prev_share.drop(columns=['fiscal_year'])

    guidance['fiscal_year'] = guidance['fiscal_year'].astype(int)
    guidance['quarter_num'] = guidance['quarter_num'].apply(parse_quarter).astype(int)
    guidance['target_year'] = guidance['target_year'].astype(int)

    guidance['effective_from_quarter'] = np.where(
        guidance['quarter_num'] == 4,
        1,
        guidance['quarter_num']
    ).astype(int)

    if dedupe_same_effective_quarter:
        if 'release_date' in guidance.columns:
            guidance['_release_sort'] = pd.to_datetime(
                guidance['release_date'].astype(str),
                format='%Y%m%d',
                errors='coerce'
            )
            guidance = (
                guidance.sort_values(
                    ['target_year', 'effective_from_quarter', '_release_sort', 'fiscal_year', 'quarter_num']
                )
                .drop_duplicates(['target_year', 'effective_from_quarter'], keep='last')
                .drop(columns=['_release_sort'])
            )
        else:
            guidance = (
                guidance.sort_values(['target_year', 'effective_from_quarter', 'fiscal_year', 'quarter_num'])
                        .drop_duplicates(['target_year', 'effective_from_quarter'], keep='last')
            )

    guidance = guidance.rename(columns={
        'fiscal_year': 'guidance_source_year',
        'fiscal_quarter': 'guidance_source_quarter'
    })

    guidance = guidance.sort_values(
        ['target_year', 'effective_from_quarter', 'guidance_source_year', 'quarter_num']
    ).reset_index(drop=True)

    ranges = []
    for target_year, grp in guidance.groupby('target_year', sort=True):
        grp = grp.sort_values('effective_from_quarter').reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i].copy()
            start_q = int(row['effective_from_quarter'])

            if i < len(grp) - 1:
                end_q = int(grp.iloc[i + 1]['effective_from_quarter']) - 1
            else:
                end_q = 4

            if end_q < start_q:
                continue

            row = row.to_dict()
            row['effective_to_quarter'] = end_q
            ranges.append(row)

    ranges_df = pd.DataFrame(ranges)

    if ranges_df.empty:
        return pd.DataFrame(columns=[
            'fiscal_year', 'fiscal_quarter',
            'revenue_target_min', 'revenue_target_max', 'actual_revenue'
        ])

    expanded_rows = []
    for _, row in ranges_df.iterrows():
        for q in range(int(row['effective_from_quarter']), int(row['effective_to_quarter']) + 1):
            one = row.to_dict()
            one['compare_quarter_num'] = q
            one['fiscal_quarter'] = f'Q{q}'
            expanded_rows.append(one)

    expanded = pd.DataFrame(expanded_rows)

    expanded = expanded.merge(
        prev_share.rename(columns={'quarter_num': 'compare_quarter_num'}),
        on=['target_year', 'compare_quarter_num'],
        how='left'
    )

    expanded['revenue_share_used'] = expanded['prev_revenue_ytd_share']
    expanded['revenue_share_basis'] = 'prev_year_ytd_share'

    fallback_mask = expanded['revenue_share_used'].isna()
    expanded.loc[fallback_mask, 'revenue_share_used'] = (
        expanded.loc[fallback_mask, 'compare_quarter_num'] / 4.0
    )
    expanded.loc[fallback_mask, 'revenue_share_basis'] = 'linear_q_over_4'

    expanded['revenue_target_min'] = expanded['revenue_min_M'] * expanded['revenue_share_used']
    expanded['revenue_target_max'] = expanded['revenue_max_M'] * expanded['revenue_share_used']

    actual_compare = actual[['fiscal_year', 'quarter_num', 'revenue_ytd_M']].copy()

    expanded = expanded.merge(
        actual_compare.rename(columns={
            'fiscal_year': 'target_year',
            'quarter_num': 'compare_quarter_num',
            'revenue_ytd_M': 'actual_revenue'
        }),
        on=['target_year', 'compare_quarter_num'],
        how='left'
    )

    result = expanded[[
        'target_year',
        'fiscal_quarter',
        'revenue_target_min',
        'revenue_target_max',
        'actual_revenue',
        'guidance_source_year',
        'guidance_source_quarter',
        'effective_from_quarter',
        'effective_to_quarter',
        'revenue_share_used',
        'revenue_share_basis'
    ]].rename(columns={
        'target_year': 'fiscal_year'
    })

    result = result.sort_values(
        ['fiscal_year', 'fiscal_quarter']
    ).reset_index(drop=True)

    return result

def build_revenue_revision_table(guidance_m_df: pd.DataFrame) -> pd.DataFrame:
    # Compare consecutive revenue guidance versions within each target year and classify the direction of change.

    required = {'fiscal_year', 'fiscal_quarter', 'target_year', 'revenue_min_M', 'revenue_max_M'}
    missing = required - set(guidance_m_df.columns)
    if missing:
        raise ValueError(f"guidance_m_df missing columns: {sorted(missing)}")

    df = guidance_m_df.copy()

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

    if 'quarter_num' not in df.columns:
        df['quarter_num'] = df['fiscal_quarter'].apply(parse_quarter)

    df['fiscal_year'] = pd.to_numeric(df['fiscal_year'], errors='coerce').astype('Int64')
    df['target_year'] = pd.to_numeric(df['target_year'], errors='coerce').astype('Int64')
    df['quarter_num'] = pd.to_numeric(df['quarter_num'], errors='coerce').astype('Int64')
    df['revenue_min_M'] = pd.to_numeric(df['revenue_min_M'], errors='coerce')
    df['revenue_max_M'] = pd.to_numeric(df['revenue_max_M'], errors='coerce')

    if 'release_date' in df.columns:
        df['_release_sort'] = pd.to_datetime(df['release_date'].astype(str), format='%Y%m%d', errors='coerce')
        sort_cols = ['target_year', '_release_sort', 'fiscal_year', 'quarter_num']
    else:
        sort_cols = ['target_year', 'fiscal_year', 'quarter_num']

    df = df.sort_values(sort_cols).reset_index(drop=True)

    dedupe_cols = ['target_year', 'fiscal_year', 'quarter_num', 'revenue_min_M', 'revenue_max_M']
    df = df.drop_duplicates(subset=dedupe_cols, keep='last').reset_index(drop=True)

    df['revenue_mid_M'] = (df['revenue_min_M'] + df['revenue_max_M']) / 2.0
    df['revenue_width_M'] = df['revenue_max_M'] - df['revenue_min_M']

    out_rows = []

    for target_year, grp in df.groupby('target_year', sort=True):
        grp = grp.sort_values(sort_cols[1:] if sort_cols[0] == 'target_year' else sort_cols).reset_index(drop=True)

        if len(grp) <= 1:
            continue

        for i in range(1, len(grp)):
            prev = grp.iloc[i - 1]
            curr = grp.iloc[i]

            delta_min = curr['revenue_min_M'] - prev['revenue_min_M']
            delta_max = curr['revenue_max_M'] - prev['revenue_max_M']
            delta_mid = curr['revenue_mid_M'] - prev['revenue_mid_M']
            delta_width = curr['revenue_width_M'] - prev['revenue_width_M']

            if pd.isna(delta_min) or pd.isna(delta_max):
                direction = 'missing'
            elif delta_min > 0 and delta_max > 0:
                direction = 'up'
            elif delta_min < 0 and delta_max < 0:
                direction = 'down'
            elif delta_min == 0 and delta_max == 0:
                direction = 'unchanged'
            else:
                direction = 'mixed'

            row = {
                'target_year': int(target_year),

                'prev_guidance_year': int(prev['fiscal_year']) if pd.notna(prev['fiscal_year']) else np.nan,
                'prev_guidance_quarter': prev['fiscal_quarter'],
                'curr_guidance_year': int(curr['fiscal_year']) if pd.notna(curr['fiscal_year']) else np.nan,
                'curr_guidance_quarter': curr['fiscal_quarter'],

                'prev_revenue_min_M': prev['revenue_min_M'],
                'prev_revenue_max_M': prev['revenue_max_M'],
                'curr_revenue_min_M': curr['revenue_min_M'],
                'curr_revenue_max_M': curr['revenue_max_M'],

                'delta_revenue_min_M': delta_min,
                'delta_revenue_max_M': delta_max,
                'delta_revenue_mid_M': delta_mid,
                'delta_revenue_width_M': delta_width,

                'revision_direction': direction
            }

            if 'release_date' in df.columns:
                row['prev_release_date'] = prev.get('release_date', np.nan)
                row['curr_release_date'] = curr.get('release_date', np.nan)

            out_rows.append(row)

    return pd.DataFrame(out_rows)

def summarize_revenue_revisions(revision_df: pd.DataFrame) -> pd.DataFrame:
    # Summarize up/down/mixed/unchanged revision counts and average mid-point change by target year.
    if revision_df.empty:
        return pd.DataFrame(columns=[
            'target_year', 'n_revisions', 'up_cnt', 'down_cnt', 'mixed_cnt', 'unchanged_cnt'
        ])

    summary = (
        revision_df.groupby('target_year')
        .agg(
            n_revisions=('revision_direction', 'size'),
            up_cnt=('revision_direction', lambda s: (s == 'up').sum()),
            down_cnt=('revision_direction', lambda s: (s == 'down').sum()),
            mixed_cnt=('revision_direction', lambda s: (s == 'mixed').sum()),
            unchanged_cnt=('revision_direction', lambda s: (s == 'unchanged').sum()),
            avg_mid_change_M=('delta_revenue_mid_M', 'mean'),
            total_mid_change_M=('delta_revenue_mid_M', 'sum')
        )
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    actual_df = pd.DataFrame({
        'fiscal_year': [2020, 2020, 2020, 2020, 2021, 2021],
        'fiscal_quarter': ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2'],
        'revenue_ytd_M': [100, 200, 300, 400, 150, 250],
        'fcf_ytd_M': [10, 20, 30, 40, 15, 25]
    })

    guidance_df = pd.DataFrame({
        'fiscal_year': [2020, 2020, 2021],
        'fiscal_quarter': ['Q4', 'Q4', 'Q4'],
        'revenue_min': ['10%', '500M', '600M'],
        'revenue_max': ['20%', '600M', '700M'],
        'revenue_unit': ['%', '$M', '$M'],
        'fcf_min': ['5%', '50M', '60M'],
        'fcf_max': ['15%', '70M', '80M'],
        'fcf_unit': ['%', '$M', '$M']
    })

    normalized_guidance = normalize_guidance_to_million(actual_df, guidance_df)
    print(normalized_guidance[['fiscal_year', 'fiscal_quarter', 'target_year',
                               'revenue_min_M', 'revenue_max_M',
                               'fcf_min_M', 'fcf_max_M']])
