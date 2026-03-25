#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CRSP US size factor test (daily, hold N trading days)")
    p.add_argument("--data-dir", type=Path, default=Path("../CRSP数据"))
    p.add_argument("--daily-zip", type=str, default="us_daily_stock.zip")
    p.add_argument("--delisting-zip", type=str, default="delisting.zip")
    p.add_argument("--output-dir", type=Path, default=Path("./output"))
    p.add_argument("--n-groups", type=int, default=50)
    p.add_argument("--holding-days", type=int, default=7)
    p.add_argument("--start-date", type=str, default="1965-01-01")
    p.add_argument("--min-turnover", type=float, default=0.001, help="Minimum rolling daily turnover over liquidity window")
    p.add_argument("--min-price", type=float, default=5.0, help="Minimum formation-day price estimated as cap / ShrOut")
    p.add_argument(
        "--min-dollar-volume",
        type=float,
        default=0.0,
        help="Optional minimum rolling average dollar volume; 0 disables absolute cutoff",
    )
    p.add_argument("--liquidity-window", type=int, default=20, help="Rolling window length in trading days for liquidity measures")
    p.add_argument("--volume-quantile-cut", type=float, default=0.2, help="Drop stocks below this rolling ADV quantile")
    p.add_argument("--min-stocks-per-date", type=int, default=100)
    p.add_argument("--rolling-window", type=int, default=126)
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def month_tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    std = x.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return x.mean() / (std / np.sqrt(len(x)))


def max_drawdown(ret: pd.Series) -> float:
    s = ret.dropna()
    if s.empty:
        return np.nan
    nav = (1.0 + s).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def winsorize_by_date(df: pd.DataFrame, col: str, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    q = df.groupby("date")[col].quantile([lo, hi]).unstack()
    q.columns = ["q_low", "q_high"]
    out = df[["date", col]].join(q, on="date")
    return out[col].clip(out["q_low"], out["q_high"])


def load_delisting_daily(data_dir: Path, delisting_zip: str) -> pd.DataFrame:
    d = pd.read_csv(data_dir / delisting_zip, compression="zip", usecols=["PERMNO", "DelistingDt", "DelRet"], low_memory=False)
    d["PERMNO"] = to_num(d["PERMNO"]).astype("Int64")
    d["DelistingDt"] = pd.to_datetime(d["DelistingDt"], errors="coerce")
    d["DelRet"] = to_num(d["DelRet"])
    d = d.dropna(subset=["PERMNO", "DelistingDt"])
    d["PERMNO"] = d["PERMNO"].astype("int32")
    d = d.rename(columns={"DelistingDt": "date", "DelRet": "dlret"})[["PERMNO", "date", "dlret"]]
    d = d.sort_values(["PERMNO", "date"]).groupby(["PERMNO", "date"], as_index=False).last()
    return d


def load_daily(data_dir: Path, daily_zip: str, max_rows: int | None) -> pd.DataFrame:
    usecols = [
        "PERMNO",
        "DlyCalDt",
        "DlyRet",
        "DlyCap",
        "DlyVol",
        "ShrOut",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "USIncFlg",
        "SecurityActiveFlg",
    ]
    chunks = []
    loaded = 0
    reader = pd.read_csv(data_dir / daily_zip, compression="zip", usecols=usecols, chunksize=1_000_000, low_memory=False)
    for ch in reader:
        ch["PERMNO"] = to_num(ch["PERMNO"]).astype("Int64")
        ch["DlyCalDt"] = pd.to_datetime(ch["DlyCalDt"], errors="coerce")
        ch["DlyRet"] = to_num(ch["DlyRet"]).astype("float64")
        ch["DlyCap"] = to_num(ch["DlyCap"]).astype("float64")
        ch["DlyVol"] = to_num(ch["DlyVol"]).astype("float64")
        ch["ShrOut"] = to_num(ch["ShrOut"]).astype("float64")

        cond = (
            (ch["SecurityType"] == "EQTY")
            & (ch["SecuritySubType"] == "COM")
            & (ch["USIncFlg"] == "Y")
            & (ch["SecurityActiveFlg"] == "Y")
        )
        cond = cond & (ch["ShareType"].isna() | (ch["ShareType"] == "NS"))
        ch = ch.loc[cond, ["PERMNO", "DlyCalDt", "DlyRet", "DlyCap", "DlyVol", "ShrOut"]].copy()
        ch = ch.dropna(subset=["PERMNO", "DlyCalDt"])
        ch["PERMNO"] = ch["PERMNO"].astype("int32")
        ch = ch.rename(columns={"DlyCalDt": "date", "DlyRet": "ret", "DlyCap": "cap", "DlyVol": "vol"})
        chunks.append(ch)
        loaded += len(ch)
        if max_rows is not None and loaded >= max_rows:
            break

    if not chunks:
        raise ValueError("No valid daily rows loaded.")
    return pd.concat(chunks, ignore_index=True)


def assign_group(x: pd.Series, n_groups: int) -> pd.Series:
    n = x.notna().sum()
    if n == 0:
        return pd.Series(np.nan, index=x.index)
    r = x.rank(method="first", pct=True)
    g = np.ceil(r * n_groups).clip(1, n_groups)
    return g.astype("Int64")


def calc_fwd_ret_from_series(r: pd.Series, holding_days: int) -> pd.Series:
    # Forward compounded return over next N trading days: prod(1+r_{t+1..t+N})-1
    x = (1.0 + r).shift(-1).rolling(holding_days, min_periods=holding_days).apply(np.prod, raw=True) - 1.0
    return x.shift(-(holding_days - 1))


def build_panel(args: argparse.Namespace) -> pd.DataFrame:
    df = load_daily(args.data_dir, args.daily_zip, args.max_rows)
    dld = load_delisting_daily(args.data_dir, args.delisting_zip)
    df = df.merge(dld, on=["PERMNO", "date"], how="left")

    r = df["ret"].astype("float64")
    dr = df["dlret"].astype("float64")
    df["ret_adj"] = np.where(r.notna() & dr.notna(), (1.0 + r) * (1.0 + dr) - 1.0, np.where(r.notna(), r, dr))

    df.loc[df["cap"] <= 0, "cap"] = np.nan
    df.loc[df["vol"] < 0, "vol"] = np.nan
    df.loc[df["ShrOut"] <= 0, "ShrOut"] = np.nan

    df["turnover"] = df["vol"] / df["ShrOut"]
    df.loc[~np.isfinite(df["turnover"]), "turnover"] = np.nan
    df["price"] = df["cap"] / df["ShrOut"]
    df.loc[~np.isfinite(df["price"]) | (df["price"] <= 0), "price"] = np.nan
    df["dollar_vol"] = df["price"] * df["vol"]
    df.loc[~np.isfinite(df["dollar_vol"]) | (df["dollar_vol"] < 0), "dollar_vol"] = np.nan
    df["log_cap"] = np.log(df["cap"])
    df["log_cap_w"] = winsorize_by_date(df, "log_cap", 0.01, 0.99)

    start_ts = pd.to_datetime(args.start_date)
    df = df[df["date"] >= start_ts].copy()
    df = df.sort_values(["PERMNO", "date"])
    window = max(int(args.liquidity_window), 1)
    df["turnover_avg"] = df.groupby("PERMNO")["turnover"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    df["dollar_vol_avg"] = df.groupby("PERMNO")["dollar_vol"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    dollar_vol_cut = df.groupby("date")["dollar_vol_avg"].quantile(args.volume_quantile_cut).rename("dollar_vol_cut")
    df = df.join(dollar_vol_cut, on="date")
    df["is_liquid"] = (
        (df["price"] >= args.min_price)
        & (df["vol"] > 0)
        & (df["turnover_avg"] >= args.min_turnover)
        & (df["dollar_vol_avg"] >= df["dollar_vol_cut"])
    )
    if args.min_dollar_volume > 0:
        df["is_liquid"] = df["is_liquid"] & (df["dollar_vol_avg"] >= args.min_dollar_volume)

    df["fwd_ret"] = df.groupby("PERMNO")["ret_adj"].transform(
        lambda s: calc_fwd_ret_from_series(s, holding_days=args.holding_days)
    )

    # Rebalance every N trading days on global trading calendar.
    all_days = np.array(sorted(df["date"].dropna().unique()))
    rebalance_days = set(all_days[:: args.holding_days])
    df["is_rebalance"] = df["date"].isin(rebalance_days)

    panel = df[
        df["is_rebalance"]
        & df["is_liquid"]
        & df["log_cap_w"].notna()
        & df["cap"].notna()
        & df["fwd_ret"].notna()
    ][["PERMNO", "date", "log_cap_w", "cap", "fwd_ret"]].copy()

    cnt = panel.groupby("date")["PERMNO"].transform("count")
    panel = panel[cnt >= args.min_stocks_per_date].copy()
    panel["group"] = panel.groupby("date")["log_cap_w"].transform(assign_group, n_groups=args.n_groups).astype("Int64")
    panel = panel.dropna(subset=["group"])
    panel["group"] = panel["group"].astype(int)
    return panel


def run_test(panel: pd.DataFrame, n_groups: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ew = panel.groupby(["date", "group"], as_index=False)["fwd_ret"].mean().rename(columns={"fwd_ret": "ew_ret"})
    vw = (
        panel.assign(weight=panel["cap"])
        .groupby(["date", "group"], as_index=False)
        .apply(
            lambda x: pd.Series({"vw_ret": np.average(x["fwd_ret"], weights=x["weight"]) if x["weight"].sum() > 0 else np.nan}),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    grp = ew.merge(vw, on=["date", "group"], how="outer").sort_values(["date", "group"]).reset_index(drop=True)

    def add_ls(col: str) -> pd.DataFrame:
        piv = grp.pivot(index="date", columns="group", values=col)
        out = (piv[1] - piv[n_groups]).rename(col).reset_index()
        out["group"] = 0
        return out[["date", "group", col]]

    grp = pd.concat([grp, add_ls("ew_ret"), add_ls("vw_ret")], ignore_index=True).sort_values(["date", "group"])

    ic = (
        panel.groupby("date")
        .apply(
            lambda x: pd.Series(
                {
                    "ic_pearson": x["log_cap_w"].corr(x["fwd_ret"], method="pearson"),
                    "ic_spearman": x["log_cap_w"].corr(x["fwd_ret"], method="spearman"),
                    "n_stocks": x["PERMNO"].nunique(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    summary_rows = []
    for c in ["ic_pearson", "ic_spearman"]:
        s = ic[c].dropna()
        summary_rows.append(
            {
                "metric": c,
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "t_stat": month_tstat(s),
                "ir": s.mean() / s.std(ddof=1) if s.std(ddof=1) else np.nan,
                "n_periods": len(s),
            }
        )
    for c in ["ew_ret", "vw_ret"]:
        s = grp.loc[grp["group"] == 0, c].dropna()
        summary_rows.append(
            {
                "metric": f"long_short_{c}",
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "t_stat": month_tstat(s),
                "n_periods": len(s),
            }
        )
    summary = pd.DataFrame(summary_rows)

    # Group stats + tests
    rows = []
    tests = []
    for ret_col, scheme in [("ew_ret", "EW"), ("vw_ret", "VW")]:
        piv = grp[grp["group"] > 0].pivot(index="date", columns="group", values=ret_col).sort_index()
        for g in range(1, n_groups + 1):
            s = piv[g].dropna()
            rows.append(
                {
                    "scheme": scheme,
                    "group": g,
                    "mean_return": s.mean(),
                    "std": s.std(ddof=1),
                    "sharpe": s.mean() / s.std(ddof=1) if s.std(ddof=1) else np.nan,
                    "t_stat": month_tstat(s),
                    "max_drawdown": max_drawdown(s),
                    "n_periods": len(s),
                }
            )
        common = piv[1].dropna().index.intersection(piv[n_groups].dropna().index)
        t_stat, p_t = stats.ttest_rel(piv.loc[common, 1], piv.loc[common, n_groups], nan_policy="omit") if len(common) >= 3 else (np.nan, np.nan)
        samples = [piv[g].dropna().values for g in range(1, n_groups + 1) if piv[g].notna().sum() >= 3]
        f_stat, p_a = stats.f_oneway(*samples) if len(samples) >= 2 else (np.nan, np.nan)
        tests.append(
            {
                "scheme": scheme,
                "long_short_mean_q1_minus_qn": (piv.loc[common, 1] - piv.loc[common, n_groups]).mean() if len(common) else np.nan,
                "t_stat_q1_vs_qn": t_stat,
                "p_ttest_q1_vs_qn": p_t,
                "f_stat_anova": f_stat,
                "p_anova": p_a,
            }
        )
    stats_df = pd.DataFrame(rows)
    tests_df = pd.DataFrame(tests)
    return grp, ic, summary, stats_df, tests_df


def make_plots(grp: pd.DataFrame, ic: pd.DataFrame, stats_df: pd.DataFrame, out: Path, n_groups: int, rolling_window: int) -> None:
    nav = grp[grp["group"] > 0][["date", "group", "ew_ret", "vw_ret"]].sort_values(["group", "date"]).copy()
    nav["ew_nav"] = nav.groupby("group")["ew_ret"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    nav["vw_nav"] = nav.groupby("group")["vw_ret"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())

    fig, ax = plt.subplots(figsize=(12, 6))
    for g in range(1, n_groups + 1):
        d = nav[nav["group"] == g]
        ax.plot(d["date"], d["ew_nav"], label=f"G{g}", linewidth=0.9)
    ax.set_title("Size Factor Group NAV (Daily base, EW)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "size_group_nav_ew.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for g in range(1, n_groups + 1):
        d = nav[nav["group"] == g]
        ax.plot(d["date"], d["vw_nav"], label=f"G{g}", linewidth=0.9)
    ax.set_title("Size Factor Group NAV (Daily base, VW)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "size_group_nav_vw.png", dpi=150)
    plt.close(fig)

    icp = ic[["date", "ic_pearson", "ic_spearman"]].copy().sort_values("date")
    icp["ic_pearson_roll"] = icp["ic_pearson"].rolling(rolling_window, min_periods=max(20, rolling_window // 2)).mean()
    icp["ic_spearman_roll"] = icp["ic_spearman"].rolling(rolling_window, min_periods=max(20, rolling_window // 2)).mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(icp["date"], icp["ic_pearson_roll"], label=f"Pearson {rolling_window}P mean")
    ax.plot(icp["date"], icp["ic_spearman_roll"], label=f"Spearman {rolling_window}P mean")
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax.set_title("Rolling IC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "size_rolling_ic.png", dpi=150)
    plt.close(fig)

    # dashboard (EW and VW)
    for scheme, ret_col in [("ew", "ew_ret"), ("vw", "vw_ret")]:
        piv = grp[grp["group"] > 0].pivot(index="date", columns="group", values=ret_col).sort_index()
        ls = piv[1] - piv[n_groups]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Size Factor Dashboard ({scheme.upper()})", fontsize=14, fontweight="bold")
        colors = plt.cm.RdYlGn(np.linspace(0.9, 0.1, n_groups))

        ax = axes[0, 0]
        for g in range(1, n_groups + 1):
            nav_g = (1.0 + piv[g].fillna(0.0)).cumprod()
            ax.plot(nav_g.index, nav_g.values, color=colors[g - 1], linewidth=0.9, label=f"G{g}")
        ax.set_title("Cumulative Returns by Group")
        ax.legend(ncol=5, fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        s = stats_df[stats_df["scheme"] == scheme.upper()].sort_values("group")
        ax.bar([f"G{i}" for i in s["group"]], s["mean_return"].values * 100, color=colors, edgecolor="white")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Mean Return per Rebalance (%)")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[0, 2]
        ic_s = ic["ic_spearman"]
        ax.plot(ic["date"], ic_s, color="steelblue", alpha=0.55, linewidth=0.9)
        ax.plot(ic["date"], ic_s.rolling(63, min_periods=20).mean(), color="orange", linewidth=1.5, label="63P MA")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("IC Time Series")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.hist(ic_s.dropna(), bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("IC Distribution")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ls_nav = (1.0 + ls.fillna(0.0)).cumprod()
        ax.plot(ls_nav.index, ls_nav.values, color="purple", linewidth=1.4)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Long-Short NAV (G1-G{n_groups})")
        ax.grid(alpha=0.3)

        ax = axes[1, 2]
        y = s["mean_return"].values * 100
        x = [f"G{i}" for i in s["group"]]
        ax.plot(x, y, "o-", color="darkblue", linewidth=1.8, markersize=4)
        ax.set_title("Monotonicity")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out / f"size_factor_dashboard_{scheme}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    nav.to_csv(out / "size_nav_series.csv", index=False)
    icp.to_csv(out / "size_rolling_ic_series.csv", index=False)


def print_report(ic: pd.DataFrame, stats_df: pd.DataFrame, tests_df: pd.DataFrame, n_groups: int) -> None:
    print("=" * 70)
    print("Factor Analysis Report - US Size Factor (Daily, hold 7 trading days)")
    print("=" * 70)
    for c in ["ic_pearson", "ic_spearman"]:
        s = ic[c].dropna()
        print(f"{c:<12} mean={s.mean(): .4f} std={s.std(ddof=1): .4f} ir={((s.mean()/s.std(ddof=1)) if s.std(ddof=1) else np.nan): .4f} win_rate={(s>0).mean(): .2%}")
    for scheme in ["EW", "VW"]:
        sub = stats_df[stats_df["scheme"] == scheme].sort_values("group")
        print(f"\n[{scheme} Group Stats]")
        print("group  mean_return     std    sharpe   t_stat   max_dd")
        for _, r in sub.iterrows():
            print(f"G{int(r['group']):<3} {r['mean_return']:>10.4%} {r['std']:>8.4%} {r['sharpe']:>8.2f} {r['t_stat']:>8.2f} {r['max_drawdown']:>8.2%}")
        tr = tests_df[tests_df["scheme"] == scheme].iloc[0]
        print(f"Long-Short G1-G{n_groups}: {tr['long_short_mean_q1_minus_qn']:+.4%} | t={tr['t_stat_q1_vs_qn']:+.3f}, p={tr['p_ttest_q1_vs_qn']:.4f} | ANOVA p={tr['p_anova']:.4f}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(args)
    grp, ic, summary, stats_df, tests_df = run_test(panel, args.n_groups)
    make_plots(grp, ic, stats_df, args.output_dir, args.n_groups, args.rolling_window)

    grp.to_csv(args.output_dir / "size_group_returns.csv", index=False)
    ic.to_csv(args.output_dir / "size_ic_series.csv", index=False)
    summary.to_csv(args.output_dir / "size_factor_summary.csv", index=False)
    stats_df.to_csv(args.output_dir / "size_group_stats.csv", index=False)
    tests_df.to_csv(args.output_dir / "size_hypothesis_tests.csv", index=False)

    print(f"Rows in panel: {len(panel):,}")
    print(f"Rebalance dates: {ic['date'].nunique():,}")
    print(f"Average stocks per rebalance: {ic['n_stocks'].mean():.1f}")
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
