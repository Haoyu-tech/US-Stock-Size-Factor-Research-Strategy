#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="US size factor grouped VW strategy using daily data with weekly/monthly calendar rebalancing"
    )
    p.add_argument("--data-dir", type=Path, default=Path("../CRSP鏁版嵁"))
    p.add_argument("--daily-zip", type=str, default="us_daily_stock.zip")
    p.add_argument("--delisting-zip", type=str, default="delisting.zip")
    p.add_argument(
        "--fundamentals-zip",
        type=str,
        default="CRSP Compustat Merged Database - Fundamentals Quarterly.zip",
    )
    p.add_argument("--output-dir", type=Path, default=Path("./output_calendar_rebalance"))
    p.add_argument("--frequencies", nargs="+", default=["weekly", "monthly"], choices=["weekly", "monthly"])
    p.add_argument("--n-groups", type=int, default=10)
    p.add_argument("--start-date", type=str, default="1965-01-01")
    p.add_argument("--min-turnover", type=float, default=0.001, help="Minimum rolling daily turnover over liquidity window")
    p.add_argument("--min-price", type=float, default=5.0, help="Minimum formation-day price estimated as cap / ShrOut")
    p.add_argument(
        "--min-dollar-volume",
        type=float,
        default=0.0,
        help="Optional minimum rolling average dollar volume; 0 disables absolute cutoff",
    )
    p.add_argument("--liquidity-window", type=int, default=20, help="Rolling trading-day window for liquidity measures")
    p.add_argument("--volume-quantile-cut", type=float, default=0.2, help="Drop stocks below this rolling ADV quantile")
    p.add_argument("--min-stocks-per-rebalance", type=int, default=100)
    p.add_argument("--min-listing-days", type=int, default=180, help="Minimum days since IPO/listing")
    p.add_argument(
        "--fundamental-lag-days",
        type=int,
        default=90,
        help="Lag in days applied to Compustat fundamentals before they become investable",
    )
    p.add_argument("--cost-bps", type=float, default=10.0, help="One-way transaction cost in bps applied to portfolio turnover")
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def period_tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    std = x.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(x.mean() / (std / np.sqrt(len(x))))


def max_drawdown(ret: pd.Series) -> float:
    s = ret.dropna()
    if s.empty:
        return np.nan
    nav = (1.0 + s).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def annualization_factor(freq: str) -> float:
    return 52.0 if freq == "weekly" else 12.0


def rolling_window_for_freq(freq: str) -> int:
    return 52 if freq == "weekly" else 24


def winsorize_by_date(df: pd.DataFrame, col: str, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    q = df.groupby("date")[col].quantile([lo, hi]).unstack()
    q.columns = ["q_low", "q_high"]
    out = df[["date", col]].join(q, on="date")
    return out[col].clip(out["q_low"], out["q_high"])


def load_delisting_daily(data_dir: Path, delisting_zip: str) -> pd.DataFrame:
    d = pd.read_csv(
        data_dir / delisting_zip,
        compression="zip",
        usecols=["PERMNO", "DelistingDt", "DelRet"],
        low_memory=False,
    )
    d["PERMNO"] = to_num(d["PERMNO"]).astype("Int64")
    d["DelistingDt"] = pd.to_datetime(d["DelistingDt"], errors="coerce")
    d["DelRet"] = to_num(d["DelRet"])
    d = d.dropna(subset=["PERMNO", "DelistingDt"])
    d["PERMNO"] = d["PERMNO"].astype("int32")
    d = d.rename(columns={"DelistingDt": "date", "DelRet": "dlret"})[["PERMNO", "date", "dlret"]]
    d = d.sort_values(["PERMNO", "date"]).groupby(["PERMNO", "date"], as_index=False).last()
    return d


def load_daily(
    data_dir: Path,
    daily_zip: str,
    max_rows: int | None,
    start_date: str | None,
    min_listing_days: int,
) -> pd.DataFrame:
    usecols = [
        "PERMNO",
        "SecurityBegDt",
        "DlyCalDt",
        "DlyRet",
        "DlyCap",
        "DlyVol",
        "ShrOut",
        "PrimaryExch",
        "IssuerType",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "ShrAdrFlg",
        "USIncFlg",
        "SecurityActiveFlg",
        "SICCD",
    ]
    chunks: list[pd.DataFrame] = []
    loaded = 0
    start_ts = pd.to_datetime(start_date) if start_date is not None else None
    reader = pd.read_csv(
        data_dir / daily_zip,
        compression="zip",
        usecols=usecols,
        chunksize=1_000_000,
        low_memory=False,
    )
    for ch in reader:
        ch["PERMNO"] = to_num(ch["PERMNO"]).astype("Int64")
        ch["SecurityBegDt"] = pd.to_datetime(ch["SecurityBegDt"], errors="coerce")
        ch["DlyCalDt"] = pd.to_datetime(ch["DlyCalDt"], errors="coerce")
        ch["DlyRet"] = to_num(ch["DlyRet"]).astype("float32")
        ch["DlyCap"] = to_num(ch["DlyCap"]).astype("float32")
        ch["DlyVol"] = to_num(ch["DlyVol"]).astype("float32")
        ch["ShrOut"] = to_num(ch["ShrOut"]).astype("float32")
        ch["SICCD"] = to_num(ch["SICCD"]).astype("float32")

        cond = (
            (ch["SecurityType"] == "EQTY")
            & (ch["SecuritySubType"] == "COM")
            & (ch["ShareType"] == "NS")
            & (ch["USIncFlg"] == "Y")
            & (ch["SecurityActiveFlg"] == "Y")
            & (ch["PrimaryExch"].isin(["N", "A", "Q"]))
            & (ch["ShrAdrFlg"].fillna("N") != "Y")
            & (ch["IssuerType"] != "REIT")
        )
        cond = cond & (ch["SICCD"].isna() | ~ch["SICCD"].between(6000, 6999))
        if start_ts is not None:
            cond = cond & (ch["DlyCalDt"] >= start_ts)
        if min_listing_days > 0:
            listing_age = (ch["DlyCalDt"] - ch["SecurityBegDt"]).dt.days
            cond = cond & (listing_age >= int(min_listing_days))

        ch = ch.loc[
            cond,
            [
                "PERMNO",
                "SecurityBegDt",
                "DlyCalDt",
                "DlyRet",
                "DlyCap",
                "DlyVol",
                "ShrOut",
                "PrimaryExch",
                "IssuerType",
                "SecurityType",
                "SecuritySubType",
                "ShareType",
                "ShrAdrFlg",
                "USIncFlg",
                "SecurityActiveFlg",
                "SICCD",
            ],
        ].copy()
        ch = ch.dropna(subset=["PERMNO", "DlyCalDt"])
        ch["PERMNO"] = ch["PERMNO"].astype("int32")
        ch = ch.rename(
            columns={
                "DlyCalDt": "date",
                "DlyRet": "ret",
                "DlyCap": "cap",
                "DlyVol": "vol",
                "SecurityBegDt": "listing_date",
            }
        )
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


def load_fundamentals(data_dir: Path, fundamentals_zip: str, lag_days: int) -> pd.DataFrame:
    usecols = [
        "LPERMNO",
        "LINKDT",
        "LINKENDDT",
        "datadate",
        "ipodate",
        "seqq",
        "ceqq",
        "atq",
        "ltq",
        "pstkq",
        "pstkrq",
        "pstknq",
        "txditcq",
    ]
    f = pd.read_csv(data_dir / fundamentals_zip, compression="zip", usecols=usecols, low_memory=False)
    f["LPERMNO"] = to_num(f["LPERMNO"]).astype("Int64")
    f["LINKDT"] = pd.to_datetime(f["LINKDT"], errors="coerce")
    f["LINKENDDT"] = pd.to_datetime(f["LINKENDDT"], errors="coerce")
    f["datadate"] = pd.to_datetime(f["datadate"], errors="coerce")
    f["ipodate"] = pd.to_datetime(f["ipodate"], errors="coerce")
    for col in ["seqq", "ceqq", "atq", "ltq", "pstkq", "pstkrq", "pstknq", "txditcq"]:
        f[col] = to_num(f[col]).astype("float64")

    f = f.dropna(subset=["LPERMNO", "datadate"]).copy()
    f["PERMNO"] = f["LPERMNO"].astype("int32")
    pref = f["pstkrq"].combine_first(f["pstknq"]).combine_first(f["pstkq"]).fillna(0.0)
    seq = f["seqq"]
    seq = seq.where(seq.notna(), f["ceqq"] + pref)
    seq = seq.where(seq.notna(), f["atq"] - f["ltq"])
    f["book_equity"] = seq + f["txditcq"].fillna(0.0) - pref
    f["effective_date"] = f["datadate"] + pd.to_timedelta(int(lag_days), unit="D")
    f["link_end_filled"] = f["LINKENDDT"].fillna(pd.Timestamp("2099-12-31"))
    f = f.sort_values(["PERMNO", "effective_date", "datadate"]).drop_duplicates(
        subset=["PERMNO", "effective_date"], keep="last"
    )
    return f[["PERMNO", "effective_date", "LINKDT", "link_end_filled", "book_equity", "ipodate"]]


def build_base_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = load_daily(args.data_dir, args.daily_zip, args.max_rows, args.start_date, args.min_listing_days)
    dld = load_delisting_daily(args.data_dir, args.delisting_zip)
    fundamentals = load_fundamentals(args.data_dir, args.fundamentals_zip, args.fundamental_lag_days)
    df = df.merge(dld, on=["PERMNO", "date"], how="left")

    r = df["ret"].astype("float64")
    dr = df["dlret"].astype("float64")
    df["ret_adj"] = np.where(
        r.notna() & dr.notna(),
        (1.0 + r) * (1.0 + dr) - 1.0,
        np.where(r.notna(), r, dr),
    )

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

    df = df.sort_values(["date", "PERMNO"]).reset_index(drop=True)

    fundamentals = fundamentals.sort_values(["effective_date", "PERMNO"]).reset_index(drop=True)
    df = pd.merge_asof(
        df,
        fundamentals,
        by="PERMNO",
        left_on="date",
        right_on="effective_date",
        direction="backward",
    )
    link_ok = df["LINKDT"].isna() | ((df["date"] >= df["LINKDT"]) & (df["date"] <= df["link_end_filled"]))
    df.loc[~link_ok, ["book_equity", "ipodate"]] = np.nan
    df = df.sort_values(["PERMNO", "date"]).reset_index(drop=True)

    window = max(int(args.liquidity_window), 1)
    df["turnover_avg"] = df.groupby("PERMNO")["turnover"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    df["dollar_vol_avg"] = df.groupby("PERMNO")["dollar_vol"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    dollar_vol_cut = (
        df.groupby("date")["dollar_vol_avg"].quantile(args.volume_quantile_cut).rename("dollar_vol_cut")
    )
    df = df.join(dollar_vol_cut, on="date")
    first_list_date = df["ipodate"].combine_first(df["listing_date"])
    listing_age = (df["date"] - first_list_date).dt.days
    seasoned = listing_age >= int(args.min_listing_days)
    positive_book_equity = df["book_equity"] > 0

    df["is_liquid"] = (
        seasoned
        & positive_book_equity
        & (df["price"] >= args.min_price)
        & (df["vol"] > 0)
        & (df["turnover_avg"] >= args.min_turnover)
        & (df["dollar_vol_avg"] >= df["dollar_vol_cut"])
    )
    if args.min_dollar_volume > 0:
        df["is_liquid"] = df["is_liquid"] & (df["dollar_vol_avg"] >= args.min_dollar_volume)
    return df[["PERMNO", "date", "ret_adj", "cap", "log_cap_w", "is_liquid"]].copy()


def select_rebalance_dates(all_dates: pd.Series, freq: str) -> pd.Series:
    dates = pd.Series(pd.to_datetime(pd.Index(sorted(pd.unique(all_dates.dropna())))), name="date")
    if freq == "weekly":
        key = dates.dt.to_period("W-FRI")
    elif freq == "monthly":
        key = dates.dt.to_period("M")
    else:
        raise ValueError(f"Unsupported frequency: {freq}")
    return dates.groupby(key).max().sort_values().reset_index(drop=True)


def calc_turnover(prev_weights: dict[int, float] | None, target_weights: dict[int, float]) -> float:
    if prev_weights is None:
        return 1.0 if target_weights else 0.0
    keys = set(prev_weights) | set(target_weights)
    buys = 0.0
    sells = 0.0
    for k in keys:
        delta = target_weights.get(k, 0.0) - prev_weights.get(k, 0.0)
        if delta > 0:
            buys += delta
        else:
            sells -= delta
    return float(max(buys, sells))


def summarize_frequency(
    freq: str,
    interval_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    tests: list[dict[str, float | int | str]] = []
    ann = annualization_factor(freq)

    for col in ["gross_return", "net_return"]:
        piv = interval_df.pivot(index="rebalance_date", columns="group", values=col).sort_index()
        for g in range(1, args.n_groups + 1):
            s = piv[g].dropna()
            rows.append(
                {
                    "frequency": freq,
                    "metric_type": col,
                    "group": g,
                    "mean_return": s.mean(),
                    "ann_return": (1.0 + s.mean()) ** ann - 1.0 if not s.empty else np.nan,
                    "std": s.std(ddof=1),
                    "ann_vol": s.std(ddof=1) * np.sqrt(ann) if len(s) > 1 else np.nan,
                    "sharpe": (s.mean() / s.std(ddof=1) * np.sqrt(ann)) if s.std(ddof=1) else np.nan,
                    "t_stat": period_tstat(s),
                    "max_drawdown": max_drawdown(s),
                    "n_periods": len(s),
                }
            )

        if 0 in piv.columns:
            s = piv[0].dropna()
            rows.append(
                {
                    "frequency": freq,
                    "metric_type": col,
                    "group": 0,
                    "mean_return": s.mean(),
                    "ann_return": (1.0 + s.mean()) ** ann - 1.0 if not s.empty else np.nan,
                    "std": s.std(ddof=1),
                    "ann_vol": s.std(ddof=1) * np.sqrt(ann) if len(s) > 1 else np.nan,
                    "sharpe": (s.mean() / s.std(ddof=1) * np.sqrt(ann)) if s.std(ddof=1) else np.nan,
                    "t_stat": period_tstat(s),
                    "max_drawdown": max_drawdown(s),
                    "n_periods": len(s),
                }
            )

        common = piv[1].dropna().index.intersection(piv[args.n_groups].dropna().index)
        t_stat, p_t = (
            stats.ttest_rel(piv.loc[common, 1], piv.loc[common, args.n_groups], nan_policy="omit")
            if len(common) >= 3
            else (np.nan, np.nan)
        )
        samples = [piv[g].dropna().values for g in range(1, args.n_groups + 1) if piv[g].notna().sum() >= 3]
        f_stat, p_a = stats.f_oneway(*samples) if len(samples) >= 2 else (np.nan, np.nan)
        tests.append(
            {
                "frequency": freq,
                "metric_type": col,
                "long_short_mean_q1_minus_qn": (piv.loc[common, 1] - piv.loc[common, args.n_groups]).mean()
                if len(common)
                else np.nan,
                "t_stat_q1_vs_qn": t_stat,
                "p_ttest_q1_vs_qn": p_t,
                "f_stat_anova": f_stat,
                "p_anova": p_a,
            }
        )

    for col in ["ic_pearson", "ic_spearman"]:
        s = ic_df[col].dropna()
        rows.append(
            {
                "frequency": freq,
                "metric_type": col,
                "group": 0,
                "mean_return": s.mean(),
                "ann_return": np.nan,
                "std": s.std(ddof=1),
                "ann_vol": np.nan,
                "sharpe": s.mean() / s.std(ddof=1) if s.std(ddof=1) else np.nan,
                "t_stat": period_tstat(s),
                "max_drawdown": np.nan,
                "n_periods": len(s),
            }
        )

    ls_daily = daily_df[daily_df["group"] == 0].sort_values("date").copy()
    if not ls_daily.empty:
        rows.append(
            {
                "frequency": freq,
                "metric_type": "long_short_daily_net",
                "group": 0,
                "mean_return": ls_daily["net_return"].mean(),
                "ann_return": (1.0 + ls_daily["net_return"].mean()) ** 252.0 - 1.0,
                "std": ls_daily["net_return"].std(ddof=1),
                "ann_vol": ls_daily["net_return"].std(ddof=1) * np.sqrt(252.0),
                "sharpe": (
                    ls_daily["net_return"].mean() / ls_daily["net_return"].std(ddof=1) * np.sqrt(252.0)
                    if ls_daily["net_return"].std(ddof=1)
                    else np.nan
                ),
                "t_stat": period_tstat(ls_daily["net_return"]),
                "max_drawdown": max_drawdown(ls_daily["net_return"]),
                "n_periods": len(ls_daily),
            }
        )

    turnover_summary = (
        interval_df.groupby("group", as_index=False)[["turnover", "trading_cost"]]
        .mean()
        .rename(columns={"turnover": "avg_turnover", "trading_cost": "avg_trading_cost"})
    )
    stats_df = pd.DataFrame(rows).merge(turnover_summary, on="group", how="left")
    tests_df = pd.DataFrame(tests)
    return stats_df, tests_df


def run_frequency_backtest(df: pd.DataFrame, freq: str, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    all_rebalance_dates = select_rebalance_dates(df["date"], freq)
    schedule = pd.DataFrame({"rebalance_date": all_rebalance_dates})
    schedule["next_rebalance"] = schedule["rebalance_date"].shift(-1)
    schedule = schedule.dropna(subset=["next_rebalance"]).copy()

    formation = df[
        df["date"].isin(schedule["rebalance_date"])
        & df["is_liquid"]
        & df["log_cap_w"].notna()
        & df["cap"].notna()
    ][["date", "PERMNO", "log_cap_w", "cap"]].copy()
    formation = formation.rename(columns={"date": "rebalance_date"})
    formation = formation.sort_values(["rebalance_date", "PERMNO", "cap"]).drop_duplicates(
        subset=["rebalance_date", "PERMNO"], keep="last"
    )
    cnt = formation.groupby("rebalance_date")["PERMNO"].transform("count")
    formation = formation[cnt >= args.min_stocks_per_rebalance].copy()
    formation["group"] = (
        formation.groupby("rebalance_date")["log_cap_w"].transform(assign_group, n_groups=args.n_groups).astype("Int64")
    )
    formation = formation.dropna(subset=["group"]).copy()
    formation["group"] = formation["group"].astype(int)
    formation["target_weight"] = formation.groupby(["rebalance_date", "group"])["cap"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else np.nan
    )
    formation = formation.dropna(subset=["target_weight"]).copy()

    schedule = schedule[schedule["rebalance_date"].isin(formation["rebalance_date"].unique())].copy()
    formation = formation.merge(schedule, on="rebalance_date", how="inner")

    returns_indexed = df[["date", "PERMNO", "ret_adj"]].sort_values(["date", "PERMNO"]).set_index("date")
    interval_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    daily_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    ic_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    holding_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    previous_weights: dict[int, dict[int, float] | None] = {g: None for g in range(1, args.n_groups + 1)}

    for interval in schedule.itertuples(index=False):
        rebalance_date = pd.Timestamp(interval.rebalance_date)
        next_rebalance = pd.Timestamp(interval.next_rebalance)
        formation_slice = formation[formation["rebalance_date"] == rebalance_date].copy()
        if formation_slice.empty:
            continue

        interval_window = returns_indexed.loc[
            (returns_indexed.index > rebalance_date) & (returns_indexed.index <= next_rebalance),
            ["PERMNO", "ret_adj"],
        ].reset_index()
        interval_dates = (
            pd.Series(pd.Index(sorted(pd.unique(interval_window["date"]))))
            if not interval_window.empty
            else pd.Series(dtype="datetime64[ns]")
        )
        if interval_dates.empty:
            continue

        universe_permnos = list(dict.fromkeys(formation_slice["PERMNO"].astype(int).tolist()))
        ret_matrix = (
            interval_window.pivot_table(index="date", columns="PERMNO", values="ret_adj", aggfunc="last")
            .reindex(index=interval_dates)
            .reindex(columns=universe_permnos)
            .fillna(0.0)
        )
        stock_forward = np.prod(1.0 + ret_matrix.to_numpy(dtype=float), axis=0) - 1.0
        signal = (
            formation_slice.drop_duplicates("PERMNO")
            .set_index("PERMNO")
            .loc[universe_permnos, "log_cap_w"]
            .astype(float)
        )
        fwd = pd.Series(stock_forward, index=universe_permnos)
        ic_rows.append(
            {
                "frequency": freq,
                "rebalance_date": rebalance_date,
                "next_rebalance": next_rebalance,
                "ic_pearson": signal.corr(fwd, method="pearson"),
                "ic_spearman": signal.corr(fwd, method="spearman"),
                "n_stocks": int(len(universe_permnos)),
            }
        )

        period_group_daily: dict[int, pd.DataFrame] = {}

        for group in range(1, args.n_groups + 1):
            grp_slice = formation_slice[formation_slice["group"] == group].copy()
            if grp_slice.empty:
                continue
            grp_slice = grp_slice.sort_values(["PERMNO", "cap"]).drop_duplicates(subset=["PERMNO"], keep="last")
            members = list(dict.fromkeys(grp_slice["PERMNO"].astype(int).tolist()))
            target_weights = grp_slice.set_index("PERMNO")["target_weight"].astype(float).to_dict()
            turnover = calc_turnover(previous_weights[group], target_weights)
            trading_cost = min(turnover * (args.cost_bps / 10000.0), 0.99)

            grp_matrix = ret_matrix.reindex(columns=members).fillna(0.0)
            w = np.array([target_weights[m] for m in members], dtype=float)
            gross_daily = np.zeros(len(grp_matrix), dtype=float)
            for i, day_ret in enumerate(grp_matrix.to_numpy(dtype=float)):
                port_ret = float(np.dot(w, day_ret))
                gross_daily[i] = port_ret
                w = w * (1.0 + day_ret)
                total = w.sum()
                if total > 0:
                    w = w / total
            end_weights = {members[i]: float(w[i]) for i in range(len(members)) if w[i] != 0}
            previous_weights[group] = end_weights

            net_daily = gross_daily.copy()
            if len(net_daily):
                net_daily[0] = (1.0 + net_daily[0]) * (1.0 - trading_cost) - 1.0

            interval_rows.append(
                {
                    "frequency": freq,
                    "rebalance_date": rebalance_date,
                    "next_rebalance": next_rebalance,
                    "group": group,
                    "n_stocks": int(len(members)),
                    "turnover": turnover,
                    "trading_cost": trading_cost,
                    "gross_return": float(np.prod(1.0 + gross_daily) - 1.0),
                    "net_return": float(np.prod(1.0 + net_daily) - 1.0),
                }
            )

            for i, dt in enumerate(grp_matrix.index.to_list()):
                daily_rows.append(
                    {
                        "frequency": freq,
                        "rebalance_date": rebalance_date,
                        "next_rebalance": next_rebalance,
                        "date": pd.Timestamp(dt),
                        "group": group,
                        "gross_return": float(gross_daily[i]),
                        "net_return": float(net_daily[i]),
                    }
                )
            period_group_daily[group] = pd.DataFrame(
                {
                    "date": grp_matrix.index.to_list(),
                    "gross_return": gross_daily,
                    "net_return": net_daily,
                }
            ).set_index("date")

            holding_rows.extend(
                {
                    "frequency": freq,
                    "rebalance_date": rebalance_date,
                    "next_rebalance": next_rebalance,
                    "group": group,
                    "PERMNO": int(row.PERMNO),
                    "log_cap_w": float(row.log_cap_w),
                    "cap": float(row.cap),
                    "target_weight": float(row.target_weight),
                }
                for row in grp_slice.itertuples(index=False)
            )

        if 1 in period_group_daily and args.n_groups in period_group_daily:
            p1 = period_group_daily[1]
            pn = period_group_daily[args.n_groups]
            common = p1.index.intersection(pn.index)
            if len(common):
                long_short_gross = p1.loc[common, "gross_return"] - pn.loc[common, "gross_return"]
                long_short_net = p1.loc[common, "net_return"] - pn.loc[common, "net_return"]
                ls_turnover = np.nan
                ls_cost = np.nan
                grp1 = next((r for r in reversed(interval_rows) if r["group"] == 1 and r["rebalance_date"] == rebalance_date), None)
                grpN = next((r for r in reversed(interval_rows) if r["group"] == args.n_groups and r["rebalance_date"] == rebalance_date), None)
                if grp1 and grpN:
                    ls_turnover = float(grp1["turnover"]) + float(grpN["turnover"])
                    ls_cost = float(grp1["trading_cost"]) + float(grpN["trading_cost"])
                interval_rows.append(
                    {
                        "frequency": freq,
                        "rebalance_date": rebalance_date,
                        "next_rebalance": next_rebalance,
                        "group": 0,
                        "n_stocks": int(formation_slice["PERMNO"].nunique()),
                        "turnover": ls_turnover,
                        "trading_cost": ls_cost,
                        "gross_return": float(np.prod(1.0 + long_short_gross.to_numpy()) - 1.0),
                        "net_return": float(np.prod(1.0 + long_short_net.to_numpy()) - 1.0),
                    }
                )
                for dt in common:
                    daily_rows.append(
                        {
                            "frequency": freq,
                            "rebalance_date": rebalance_date,
                            "next_rebalance": next_rebalance,
                            "date": pd.Timestamp(dt),
                            "group": 0,
                            "gross_return": float(long_short_gross.loc[dt]),
                            "net_return": float(long_short_net.loc[dt]),
                        }
                    )

    if not interval_rows or not ic_rows:
        raise ValueError(
            f"No valid {freq} rebalances after applying the current screens. "
            f"Try using more rows or relaxing the filters."
        )

    interval_df = pd.DataFrame(interval_rows).sort_values(["rebalance_date", "group"]).reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows).sort_values(["date", "group"]).reset_index(drop=True)
    ic_df = pd.DataFrame(ic_rows).sort_values("rebalance_date").reset_index(drop=True)
    holdings_df = pd.DataFrame(holding_rows).sort_values(["rebalance_date", "group", "PERMNO"]).reset_index(drop=True)
    stats_df, tests_df = summarize_frequency(freq, interval_df, daily_df, ic_df, args)
    return {
        "interval_returns": interval_df,
        "daily_returns": daily_df,
        "ic": ic_df,
        "stats": stats_df,
        "tests": tests_df,
        "holdings": holdings_df,
    }


def add_summary_text(ax: plt.Axes, freq: str, stats_df: pd.DataFrame, tests_df: pd.DataFrame) -> None:
    ic_row = stats_df[(stats_df["metric_type"] == "ic_spearman") & (stats_df["frequency"] == freq)].iloc[0]
    ls_row = stats_df[(stats_df["metric_type"] == "net_return") & (stats_df["group"] == 0) & (stats_df["frequency"] == freq)].iloc[0]
    test_row = tests_df[(tests_df["metric_type"] == "net_return") & (tests_df["frequency"] == freq)].iloc[0]
    lines = [
        f"Freq: {freq}",
        f"Rank IC mean: {ic_row['mean_return']:+.3f}",
        f"Rank IC t-stat: {ic_row['t_stat']:+.2f}",
        f"LS mean/period: {ls_row['mean_return']:+.2%}",
        f"LS ann. return: {ls_row['ann_return']:+.2%}" if pd.notna(ls_row["ann_return"]) else "LS ann. return: nan",
        f"LS sharpe: {ls_row['sharpe']:+.2f}",
        f"LS max DD: {ls_row['max_drawdown']:+.2%}",
        f"Avg turnover: {ls_row['avg_turnover']:.2f}" if pd.notna(ls_row["avg_turnover"]) else "Avg turnover: nan",
        f"Avg cost: {ls_row['avg_trading_cost']:.3%}" if pd.notna(ls_row["avg_trading_cost"]) else "Avg cost: nan",
        f"Q1-QN t p: {test_row['p_ttest_q1_vs_qn']:.4f}",
        f"ANOVA p: {test_row['p_anova']:.4f}",
    ]
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f6f6", "edgecolor": "#cfcfcf"},
    )


def make_combined_dashboard(results: dict[str, dict[str, pd.DataFrame]], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(len(results), 4, figsize=(24, 6 * len(results)))
    if len(results) == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"US Size Factor VW Group Strategy Dashboard (Daily Base, Cost={args.cost_bps:.1f} bps)",
        fontsize=18,
        fontweight="bold",
    )
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.1, args.n_groups))

    for row_idx, freq in enumerate(results):
        res = results[freq]
        daily_df = res["daily_returns"]
        stats_df = res["stats"]
        tests_df = res["tests"]
        ic_df = res["ic"]

        nav = daily_df[daily_df["group"] > 0][["date", "group", "net_return"]].sort_values(["group", "date"]).copy()
        nav["net_nav"] = nav.groupby("group")["net_return"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())

        ax = axes[row_idx, 0]
        for group in range(1, args.n_groups + 1):
            grp = nav[nav["group"] == group]
            ax.plot(grp["date"], grp["net_nav"], linewidth=1.0, color=colors[group - 1], label=f"G{group}")
        ax.set_title(f"{freq.title()} Net NAV by Group")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)

        ax = axes[row_idx, 1]
        ls = daily_df[daily_df["group"] == 0].sort_values("date").copy()
        if not ls.empty:
            ls["ls_nav"] = (1.0 + ls["net_return"].fillna(0.0)).cumprod()
            ax.plot(ls["date"], ls["ls_nav"], color="#7a1f5c", linewidth=1.6, label="Q1-QN Net")
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            ax.legend(fontsize=9)
        ax.set_title(f"{freq.title()} Long-Short Net NAV")
        ax.grid(alpha=0.25)

        ax = axes[row_idx, 2]
        stats_sub = stats_df[(stats_df["metric_type"] == "net_return") & (stats_df["group"] > 0)].sort_values("group")
        ax.bar(stats_sub["group"].astype(str), stats_sub["mean_return"] * 100, color=colors, edgecolor="white")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{freq.title()} Mean Net Return per Period (%)")
        ax.grid(axis="y", alpha=0.25)

        ax = axes[row_idx, 3]
        ic_roll_window = rolling_window_for_freq(freq)
        ic_plot = ic_df[["rebalance_date", "ic_spearman"]].copy().sort_values("rebalance_date")
        ic_plot["roll"] = ic_plot["ic_spearman"].rolling(
            ic_roll_window,
            min_periods=max(12, ic_roll_window // 2),
        ).mean()
        ax.plot(ic_plot["rebalance_date"], ic_plot["ic_spearman"], color="#9aa0a6", alpha=0.45, linewidth=0.9)
        ax.plot(ic_plot["rebalance_date"], ic_plot["roll"], color="#1b6ca8", linewidth=1.6)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{freq.title()} Spearman IC")
        ax.grid(alpha=0.25)
        add_summary_text(ax.inset_axes([0.56, 0.04, 0.42, 0.9]), freq, stats_df, tests_df)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output_dir / "size_calendar_rebalance_dashboard_vw.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_report(results: dict[str, dict[str, pd.DataFrame]], args: argparse.Namespace) -> None:
    print("=" * 80)
    print(f"US Size Factor VW Group Strategy Report (daily base, cost={args.cost_bps:.1f} bps)")
    print("=" * 80)
    for freq, res in results.items():
        stats_df = res["stats"]
        tests_df = res["tests"]
        ic_df = res["ic"]
        interval_df = res["interval_returns"]
        ls = stats_df[(stats_df["metric_type"] == "net_return") & (stats_df["group"] == 0)].iloc[0]
        ric = stats_df[stats_df["metric_type"] == "ic_spearman"].iloc[0]
        test = tests_df[tests_df["metric_type"] == "net_return"].iloc[0]
        print(f"\n[{freq.upper()}]")
        print(f"Rebalances: {len(ic_df):,}")
        print(f"Average stocks per rebalance: {ic_df['n_stocks'].mean():.1f}")
        print(f"Rank IC mean: {ric['mean_return']:+.4f} | t-stat: {ric['t_stat']:+.2f}")
        print(
            f"Long-Short Q1-Q{args.n_groups} net mean: {ls['mean_return']:+.4%} | ann: {ls['ann_return']:+.2%} | "
            f"sharpe: {ls['sharpe']:+.2f} | max DD: {ls['max_drawdown']:+.2%}"
        )
        print(
            f"Average turnover: {ls['avg_turnover']:.2f} | average cost: {ls['avg_trading_cost']:.3%} | "
            f"t-test p: {test['p_ttest_q1_vs_qn']:.4f} | ANOVA p: {test['p_anova']:.4f}"
        )
        groups = stats_df[(stats_df["metric_type"] == "net_return") & (stats_df["group"] > 0)].sort_values("group")
        print("Group  mean(period)  ann.return  ann.vol  sharpe")
        for _, row in groups.iterrows():
            print(
                f"G{int(row['group']):<2}   {row['mean_return']:>+10.4%}  {row['ann_return']:>+9.2%}  "
                f"{row['ann_vol']:>7.2%}  {row['sharpe']:>+6.2f}"
            )
        print(f"Saved rows: interval={len(interval_df):,}, daily={len(res['daily_returns']):,}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = build_base_dataset(args)
    ordered_freqs = [f for f in ["weekly", "monthly"] if f in args.frequencies]
    results = {freq: run_frequency_backtest(df, freq, args) for freq in ordered_freqs}

    interval_all = pd.concat([results[f]["interval_returns"] for f in ordered_freqs], ignore_index=True)
    daily_all = pd.concat([results[f]["daily_returns"] for f in ordered_freqs], ignore_index=True)
    ic_all = pd.concat([results[f]["ic"] for f in ordered_freqs], ignore_index=True)
    stats_all = pd.concat([results[f]["stats"] for f in ordered_freqs], ignore_index=True)
    tests_all = pd.concat([results[f]["tests"] for f in ordered_freqs], ignore_index=True)
    holdings_all = pd.concat([results[f]["holdings"] for f in ordered_freqs], ignore_index=True)

    interval_all.to_csv(args.output_dir / "size_calendar_rebalance_interval_returns.csv", index=False)
    daily_all.to_csv(args.output_dir / "size_calendar_rebalance_daily_returns.csv", index=False)
    ic_all.to_csv(args.output_dir / "size_calendar_rebalance_ic.csv", index=False)
    stats_all.to_csv(args.output_dir / "size_calendar_rebalance_group_stats.csv", index=False)
    tests_all.to_csv(args.output_dir / "size_calendar_rebalance_tests.csv", index=False)
    holdings_all.to_csv(args.output_dir / "size_calendar_rebalance_holdings.csv", index=False)

    make_combined_dashboard(results, args)
    print_report(results, args)
    print("\nSaved:")
    print(args.output_dir / "size_calendar_rebalance_interval_returns.csv")
    print(args.output_dir / "size_calendar_rebalance_daily_returns.csv")
    print(args.output_dir / "size_calendar_rebalance_ic.csv")
    print(args.output_dir / "size_calendar_rebalance_group_stats.csv")
    print(args.output_dir / "size_calendar_rebalance_tests.csv")
    print(args.output_dir / "size_calendar_rebalance_holdings.csv")
    print(args.output_dir / "size_calendar_rebalance_dashboard_vw.png")


if __name__ == "__main__":
    main()
