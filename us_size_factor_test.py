#!/usr/bin/env python
"""
US stock size factor test on CRSP monthly data.

Features:
- Basic data cleaning and missing-value handling
- Delisting return adjustment
- More realistic liquidity filter using price, rolling turnover, and rolling dollar volume
- Quantile portfolio returns (equal-weight and value-weight)
- IC / Rank IC (Pearson and Spearman)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRSP US size factor test")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../CRSP数据"),
        help="Directory that contains CRSP zip files",
    )
    parser.add_argument(
        "--monthly-zip",
        type=str,
        default="us_month_stock.zip",
        help="Monthly stock file zip name",
    )
    parser.add_argument(
        "--delisting-zip",
        type=str,
        default="delisting.zip",
        help="Delisting file zip name",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=20,
        help="Number of quantile groups",
    )
    parser.add_argument(
        "--min-turnover",
        type=float,
        default=0.01,
        help="Minimum rolling monthly turnover over liquidity window",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=5.0,
        help="Minimum formation-month price estimated as MthCap / ShrOut",
    )
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=0.0,
        help="Optional minimum rolling average dollar volume; 0 disables absolute cutoff",
    )
    parser.add_argument(
        "--liquidity-window",
        type=int,
        default=3,
        help="Rolling window length in months for liquidity measures",
    )
    parser.add_argument(
        "--volume-quantile-cut",
        type=float,
        default=0.2,
        help="Drop stocks below this monthly rolling dollar-volume quantile",
    )
    parser.add_argument(
        "--min-stocks-per-month",
        type=int,
        default=50,
        help="Minimum names in a month to run grouping/IC",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug cap for total loaded rows from monthly file",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=24,
        help="Rolling window length in months for IC plot",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for formation month filter, e.g. 1965-01-01",
    )
    return parser.parse_args()


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def winsorize_by_month(df: pd.DataFrame, col: str, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    q = df.groupby("month")[col].quantile([lower, upper]).unstack()
    q.columns = ["q_low", "q_high"]
    out = df[["month", col]].join(q, on="month")
    return out[col].clip(out["q_low"], out["q_high"])


def month_tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    std = x.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return x.mean() / (std / np.sqrt(len(x)))


def max_drawdown(ret_series: pd.Series) -> float:
    s = ret_series.dropna()
    if s.empty:
        return np.nan
    nav = (1.0 + s).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def load_delisting(data_dir: Path, delisting_zip: str) -> pd.DataFrame:
    path = data_dir / delisting_zip
    usecols = ["PERMNO", "DelistingDt", "DelRet"]
    dtypes = {"PERMNO": "int32"}
    dld = pd.read_csv(path, compression="zip", usecols=usecols, dtype=dtypes, low_memory=False)
    dld["DelRet"] = safe_to_numeric(dld["DelRet"])
    dld["DelistingDt"] = pd.to_datetime(dld["DelistingDt"], errors="coerce")
    dld = dld.dropna(subset=["PERMNO", "DelistingDt"])
    dld["month"] = dld["DelistingDt"].dt.to_period("M").dt.to_timestamp("M")
    dld = dld.sort_values(["PERMNO", "month", "DelistingDt"])
    dld = dld.groupby(["PERMNO", "month"], as_index=False).last()
    dld = dld[["PERMNO", "month", "DelRet"]]
    return dld


def load_monthly(data_dir: Path, monthly_zip: str, max_rows: int | None) -> pd.DataFrame:
    path = data_dir / monthly_zip
    usecols = [
        "PERMNO",
        "MthCalDt",
        "MthRet",
        "MthCap",
        "MthVol",
        "ShrOut",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "USIncFlg",
        "SecurityActiveFlg",
    ]
    chunks = []
    loaded = 0
    chunk_size = 1_000_000
    reader = pd.read_csv(path, compression="zip", usecols=usecols, chunksize=chunk_size, low_memory=False)
    for ch in reader:
        ch["PERMNO"] = pd.to_numeric(ch["PERMNO"], errors="coerce").astype("Int64")
        ch["MthCalDt"] = pd.to_datetime(ch["MthCalDt"], errors="coerce")
        ch["MthRet"] = safe_to_numeric(ch["MthRet"]).astype("float32")
        ch["MthCap"] = safe_to_numeric(ch["MthCap"]).astype("float64")
        ch["MthVol"] = safe_to_numeric(ch["MthVol"]).astype("float64")
        ch["ShrOut"] = safe_to_numeric(ch["ShrOut"]).astype("float64")

        # Keep US common equities only.
        cond = (
            (ch["SecurityType"] == "EQTY")
            & (ch["SecuritySubType"] == "COM")
            & (ch["USIncFlg"] == "Y")
            & (ch["SecurityActiveFlg"] == "Y")
        )
        # ShareType can be missing; if present, keep NS (common) and empty.
        cond = cond & (ch["ShareType"].isna() | (ch["ShareType"] == "NS"))
        ch = ch.loc[cond, ["PERMNO", "MthCalDt", "MthRet", "MthCap", "MthVol", "ShrOut"]].copy()
        ch = ch.dropna(subset=["PERMNO", "MthCalDt"])
        ch["PERMNO"] = ch["PERMNO"].astype("int32")
        ch["month"] = ch["MthCalDt"].dt.to_period("M").dt.to_timestamp("M")
        ch = ch.drop(columns=["MthCalDt"])
        chunks.append(ch)

        loaded += len(ch)
        if max_rows is not None and loaded >= max_rows:
            break

    if not chunks:
        raise ValueError("No valid monthly rows loaded. Check input files/filters.")
    df = pd.concat(chunks, ignore_index=True)
    return df


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    monthly = load_monthly(args.data_dir, args.monthly_zip, args.max_rows)
    dld = load_delisting(args.data_dir, args.delisting_zip)

    df = monthly.merge(dld, on=["PERMNO", "month"], how="left")

    # Adjust monthly return with delisting return.
    r = df["MthRet"].astype("float64")
    dr = df["DelRet"].astype("float64")
    ret_adj = np.where(
        r.notna() & dr.notna(),
        (1.0 + r) * (1.0 + dr) - 1.0,
        np.where(r.notna(), r, dr),
    )
    df["ret_adj"] = pd.Series(ret_adj, index=df.index).astype("float64")

    # Basic cleaning and missing handling.
    df.loc[df["MthCap"] <= 0, "MthCap"] = np.nan
    df.loc[df["ShrOut"] <= 0, "ShrOut"] = np.nan
    df.loc[df["MthVol"] < 0, "MthVol"] = np.nan
    df["turnover"] = df["MthVol"] / df["ShrOut"]
    df.loc[~np.isfinite(df["turnover"]), "turnover"] = np.nan
    df["price"] = df["MthCap"] / df["ShrOut"]
    df.loc[~np.isfinite(df["price"]) | (df["price"] <= 0), "price"] = np.nan
    df["dollar_vol"] = df["price"] * df["MthVol"]
    df.loc[~np.isfinite(df["dollar_vol"]) | (df["dollar_vol"] < 0), "dollar_vol"] = np.nan
    df["log_mcap"] = np.log(df["MthCap"])

    # Winsorize factor cross-section each month.
    df["log_mcap_w"] = winsorize_by_month(df, "log_mcap", 0.01, 0.99)

    df = df.sort_values(["PERMNO", "month"]).reset_index(drop=True)
    window = max(int(args.liquidity_window), 1)
    df["turnover_avg"] = df.groupby("PERMNO")["turnover"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    df["dollar_vol_avg"] = df.groupby("PERMNO")["dollar_vol"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )

    # More realistic monthly liquidity filter:
    # 1) tradable price
    # 2) positive current-month volume
    # 3) rolling turnover above threshold
    # 4) rolling dollar volume above month cross-sectional cutoff
    dollar_vol_cut = df.groupby("month")["dollar_vol_avg"].quantile(args.volume_quantile_cut).rename("dollar_vol_cut")
    df = df.join(dollar_vol_cut, on="month")
    df["is_liquid"] = (
        (df["price"] >= args.min_price)
        & (df["MthVol"] > 0)
        & (df["turnover_avg"] >= args.min_turnover)
        & (df["dollar_vol_avg"] >= df["dollar_vol_cut"])
    )
    if args.min_dollar_volume > 0:
        df["is_liquid"] = df["is_liquid"] & (df["dollar_vol_avg"] >= args.min_dollar_volume)

    if args.start_date is not None:
        start_ts = pd.to_datetime(args.start_date)
        df = df[df["month"] >= start_ts].copy()

    return df


def assign_group(x: pd.Series, n_groups: int) -> pd.Series:
    n = x.notna().sum()
    if n == 0:
        return pd.Series(np.nan, index=x.index)
    r = x.rank(method="first", pct=True)
    g = np.ceil(r * n_groups).clip(1, n_groups)
    return g.astype("Int64")


def run_factor_test(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Formation month t uses size at month t; holding return is month t+1 adjusted return.
    panel = df[["PERMNO", "month", "log_mcap_w", "MthCap", "ret_adj", "is_liquid"]].copy()
    panel = panel.sort_values(["PERMNO", "month"])

    panel["next_ret"] = panel.groupby("PERMNO")["ret_adj"].shift(-1)
    panel["next_month"] = panel.groupby("PERMNO")["month"].shift(-1)
    panel["expected_next_month"] = panel["month"] + pd.offsets.MonthEnd(1)
    panel = panel[panel["next_month"] == panel["expected_next_month"]].copy()
    panel = panel.drop(columns=["next_month", "expected_next_month", "ret_adj"])

    panel = panel[
        panel["is_liquid"]
        & panel["log_mcap_w"].notna()
        & panel["MthCap"].notna()
        & panel["next_ret"].notna()
    ].copy()

    month_count = panel.groupby("month")["PERMNO"].transform("count")
    panel = panel[month_count >= args.min_stocks_per_month].copy()

    panel["group"] = panel.groupby("month")["log_mcap_w"].transform(assign_group, n_groups=args.n_groups)
    panel = panel.dropna(subset=["group"])
    panel["group"] = panel["group"].astype(int)

    # Group returns.
    ew = panel.groupby(["month", "group"], as_index=False)["next_ret"].mean().rename(
        columns={"next_ret": "ew_ret"}
    )

    vw = (
        panel.assign(weight=panel["MthCap"])
        .groupby(["month", "group"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "vw_ret": np.average(x["next_ret"], weights=x["weight"])
                    if x["weight"].sum() > 0
                    else np.nan
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    grp = ew.merge(vw, on=["month", "group"], how="outer").sort_values(["month", "group"])

    # Add long-short: Small minus Big (G1 - Gn).
    def add_long_short(ret_col: str) -> pd.DataFrame:
        piv = grp.pivot(index="month", columns="group", values=ret_col)
        ls = (piv[1] - piv[args.n_groups]).rename(ret_col)
        out = ls.reset_index()
        out["group"] = 0
        return out[["month", "group", ret_col]]

    grp = pd.concat([grp, add_long_short("ew_ret"), add_long_short("vw_ret")], ignore_index=True)
    grp = grp.sort_values(["month", "group"]).reset_index(drop=True)

    # IC and Rank IC by month.
    ic = (
        panel.groupby("month")
        .apply(
            lambda x: pd.Series(
                {
                    "ic_pearson": x["log_mcap_w"].corr(x["next_ret"], method="pearson"),
                    "ic_spearman": x["log_mcap_w"].corr(x["next_ret"], method="spearman"),
                    "n_stocks": x["PERMNO"].nunique(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # Summary.
    summary_rows = []
    for col in ["ic_pearson", "ic_spearman"]:
        s = ic[col].dropna()
        summary_rows.append(
            {
                "metric": col,
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "t_stat": month_tstat(s),
                "ir": s.mean() / s.std(ddof=1) if s.std(ddof=1) not in [0, np.nan] else np.nan,
                "n_months": s.shape[0],
            }
        )

    ls = grp[grp["group"] == 0].copy()
    for col in ["ew_ret", "vw_ret"]:
        s = ls[col].dropna()
        summary_rows.append(
            {
                "metric": f"long_short_{col}",
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "t_stat": month_tstat(s),
                "ann_mean": (1.0 + s.mean()) ** 12 - 1.0 if not s.empty else np.nan,
                "n_months": s.shape[0],
            }
        )

    summary = pd.DataFrame(summary_rows)
    return grp, ic, summary


def evaluate_groups(grp: pd.DataFrame, n_groups: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    test_records = []

    for ret_col in ["ew_ret", "vw_ret"]:
        piv = grp[grp["group"] > 0].pivot(index="month", columns="group", values=ret_col).sort_index()
        for g in range(1, n_groups + 1):
            s = piv[g].dropna() if g in piv.columns else pd.Series(dtype=float)
            mean_r = s.mean()
            std_r = s.std(ddof=1)
            sharpe = mean_r / std_r if std_r and not np.isnan(std_r) else np.nan
            records.append(
                {
                    "scheme": ret_col.replace("_ret", "").upper(),
                    "group": g,
                    "mean_return": mean_r,
                    "std": std_r,
                    "sharpe": sharpe,
                    "t_stat": month_tstat(s),
                    "max_drawdown": max_drawdown(s),
                    "n_months": int(s.shape[0]),
                }
            )

        # Q1 vs QN t-test, and ANOVA across groups.
        q1 = piv[1].dropna() if 1 in piv.columns else pd.Series(dtype=float)
        qn = piv[n_groups].dropna() if n_groups in piv.columns else pd.Series(dtype=float)
        common = q1.index.intersection(qn.index)
        if len(common) >= 3:
            t_stat, p_ttest = stats.ttest_rel(q1.loc[common], qn.loc[common], nan_policy="omit")
        else:
            t_stat, p_ttest = (np.nan, np.nan)

        samples = [piv[g].dropna().values for g in range(1, n_groups + 1) if g in piv.columns and piv[g].notna().sum() >= 3]
        if len(samples) >= 2:
            f_stat, p_anova = stats.f_oneway(*samples)
        else:
            f_stat, p_anova = (np.nan, np.nan)

        test_records.append(
            {
                "scheme": ret_col.replace("_ret", "").upper(),
                "long_short_mean_q1_minus_qn": (q1.loc[common] - qn.loc[common]).mean() if len(common) else np.nan,
                "t_stat_q1_vs_qn": t_stat,
                "p_ttest_q1_vs_qn": p_ttest,
                "f_stat_anova": f_stat,
                "p_anova": p_anova,
                "n_common_months_q1_qn": int(len(common)),
            }
        )

    stats_df = pd.DataFrame(records).sort_values(["scheme", "group"]).reset_index(drop=True)
    tests_df = pd.DataFrame(test_records).sort_values(["scheme"]).reset_index(drop=True)
    return stats_df, tests_df


def print_report(ic: pd.DataFrame, stats_df: pd.DataFrame, tests_df: pd.DataFrame, n_groups: int) -> None:
    print("=" * 70)
    print("Factor Analysis Report — US Size Factor (CRSP)")
    print("=" * 70)

    for col in ["ic_pearson", "ic_spearman"]:
        s = ic[col].dropna()
        print(
            f"{col:<12} mean={s.mean(): .4f} std={s.std(ddof=1): .4f} "
            f"ir={((s.mean()/s.std(ddof=1)) if s.std(ddof=1) else np.nan): .4f} "
            f"win_rate={(s > 0).mean(): .2%}"
        )

    for scheme in ["EW", "VW"]:
        sub = stats_df[stats_df["scheme"] == scheme].copy().sort_values("group")
        if sub.empty:
            continue
        print(f"\n[{scheme} Group Stats]")
        print("group  mean_return     std    sharpe   t_stat   max_dd")
        for _, r in sub.iterrows():
            print(
                f"G{int(r['group']):<3} {r['mean_return']:>10.4%} {r['std']:>8.4%} "
                f"{r['sharpe']:>8.2f} {r['t_stat']:>8.2f} {r['max_drawdown']:>8.2%}"
            )
        tr = tests_df[tests_df["scheme"] == scheme]
        if not tr.empty:
            tr = tr.iloc[0]
            print(
                f"Long-Short G1-G{n_groups}: {tr['long_short_mean_q1_minus_qn']:+.4%} | "
                f"t={tr['t_stat_q1_vs_qn']:+.3f}, p={tr['p_ttest_q1_vs_qn']:.4f} | "
                f"ANOVA p={tr['p_anova']:.4f}"
            )


def plot_dashboard(
    grp: pd.DataFrame,
    ic: pd.DataFrame,
    stats_df: pd.DataFrame,
    n_groups: int,
    output_dir: Path,
    scheme: str = "EW",
) -> None:
    ret_col = "ew_ret" if scheme.upper() == "EW" else "vw_ret"
    piv = grp[grp["group"] > 0].pivot(index="month", columns="group", values=ret_col).sort_index()
    ls = piv[1] - piv[n_groups]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Single Factor Analysis Dashboard — Size Factor ({scheme.upper()})", fontsize=14, fontweight="bold")
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.1, n_groups))

    # 1) Cumulative group returns
    ax = axes[0, 0]
    for g in range(1, n_groups + 1):
        if g not in piv.columns:
            continue
        nav = (1.0 + piv[g].fillna(0.0)).cumprod()
        ax.plot(nav.index, nav.values, label=f"G{g}", color=colors[g - 1], linewidth=1.1)
    ax.set_title("Cumulative Returns by Group")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=4)

    # 2) Mean return bar
    ax = axes[0, 1]
    sub = stats_df[stats_df["scheme"] == scheme.upper()].sort_values("group")
    x = [f"G{int(v)}" for v in sub["group"].values]
    y = sub["mean_return"].values * 100.0
    ax.bar(x, y, color=colors, alpha=0.9, edgecolor="white")
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_title("Mean Return per Month (%)")
    ax.grid(axis="y", alpha=0.3)

    # 3) IC time series
    ax = axes[0, 2]
    ic_series = ic["ic_spearman"].copy()
    ax.plot(ic["month"], ic_series, alpha=0.5, color="steelblue", linewidth=1.0, label="Spearman IC")
    roll = ic_series.rolling(12, min_periods=6).mean()
    ax.plot(ic["month"], roll, color="orange", linewidth=1.8, label="12M MA")
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.axhline(ic_series.mean(), color="red", linewidth=1.0, linestyle="--", label=f"Mean={ic_series.mean():.3f}")
    ax.set_title("IC Time Series")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 4) IC histogram
    ax = axes[1, 0]
    ic_clean = ic_series.dropna()
    ax.hist(ic_clean, bins=35, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.axvline(ic_clean.mean(), color="red", linewidth=1.2, linestyle="--")
    ax.set_title("IC Distribution")
    ax.grid(alpha=0.3)

    # 5) Long-short NAV
    ax = axes[1, 1]
    ls_nav = (1.0 + ls.fillna(0.0)).cumprod()
    ax.plot(ls_nav.index, ls_nav.values, color="purple", linewidth=1.8)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_title(f"Long-Short NAV (G1 - G{n_groups})")
    ax.grid(alpha=0.3)

    # 6) Monotonicity line
    ax = axes[1, 2]
    ax.plot(x, y, "o-", linewidth=2.0, markersize=5, color="darkblue")
    ax.set_title("Monotonicity Check")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"size_factor_dashboard_{scheme.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(grp: pd.DataFrame, ic: pd.DataFrame, output_dir: Path, rolling_window: int) -> None:
    # Group cumulative NAV: plot groups 1..N (exclude long-short group=0).
    nav_src = grp[grp["group"] > 0][["month", "group", "ew_ret", "vw_ret"]].sort_values(["group", "month"]).copy()
    nav_src["ew_nav"] = nav_src.groupby("group")["ew_ret"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    nav_src["vw_nav"] = nav_src.groupby("group")["vw_ret"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    groups = sorted(nav_src["group"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(12, 6))
    for g in groups:
        d = nav_src[nav_src["group"] == g]
        ax.plot(d["month"], d["ew_nav"], label=f"G{g}")
    ax.set_title("Size Factor Group NAV (Equal-Weight)")
    ax.set_xlabel("Month")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "size_group_nav_ew.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for g in groups:
        d = nav_src[nav_src["group"] == g]
        ax.plot(d["month"], d["vw_nav"], label=f"G{g}")
    ax.set_title("Size Factor Group NAV (Value-Weight)")
    ax.set_xlabel("Month")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "size_group_nav_vw.png", dpi=150)
    plt.close(fig)

    # Rolling IC mean
    ic_plot = ic[["month", "ic_pearson", "ic_spearman"]].sort_values("month").copy()
    ic_plot["ic_pearson_roll"] = ic_plot["ic_pearson"].rolling(rolling_window, min_periods=max(6, rolling_window // 2)).mean()
    ic_plot["ic_spearman_roll"] = ic_plot["ic_spearman"].rolling(rolling_window, min_periods=max(6, rolling_window // 2)).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ic_plot["month"], ic_plot["ic_pearson_roll"], label=f"Pearson IC {rolling_window}M rolling mean")
    ax.plot(ic_plot["month"], ic_plot["ic_spearman_roll"], label=f"Spearman IC {rolling_window}M rolling mean")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Rolling Information Coefficient")
    ax.set_xlabel("Month")
    ax.set_ylabel("IC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "size_rolling_ic.png", dpi=150)
    plt.close(fig)

    # Save plot data for quick downstream usage.
    nav_src.to_csv(output_dir / "size_nav_series.csv", index=False)
    ic_plot.to_csv(output_dir / "size_rolling_ic_series.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args)
    grp, ic, summary = run_factor_test(df, args)
    stats_df, tests_df = evaluate_groups(grp, args.n_groups)
    make_plots(grp, ic, args.output_dir, args.rolling_window)
    plot_dashboard(grp, ic, stats_df, args.n_groups, args.output_dir, scheme="EW")
    plot_dashboard(grp, ic, stats_df, args.n_groups, args.output_dir, scheme="VW")

    grp.to_csv(args.output_dir / "size_group_returns.csv", index=False)
    ic.to_csv(args.output_dir / "size_ic_series.csv", index=False)
    summary.to_csv(args.output_dir / "size_factor_summary.csv", index=False)
    stats_df.to_csv(args.output_dir / "size_group_stats.csv", index=False)
    tests_df.to_csv(args.output_dir / "size_hypothesis_tests.csv", index=False)

    print(f"Loaded rows after cleaning: {len(df):,}")
    print(f"Months tested: {ic['month'].nunique():,}")
    print(f"Average stocks per month: {ic['n_stocks'].mean():.1f}")
    print_report(ic, stats_df, tests_df, args.n_groups)
    print("Saved:")
    print(args.output_dir / "size_group_returns.csv")
    print(args.output_dir / "size_ic_series.csv")
    print(args.output_dir / "size_factor_summary.csv")
    print(args.output_dir / "size_group_stats.csv")
    print(args.output_dir / "size_hypothesis_tests.csv")
    print(args.output_dir / "size_group_nav_ew.png")
    print(args.output_dir / "size_group_nav_vw.png")
    print(args.output_dir / "size_rolling_ic.png")
    print(args.output_dir / "size_factor_dashboard_ew.png")
    print(args.output_dir / "size_factor_dashboard_vw.png")


if __name__ == "__main__":
    main()
