from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Optional

RebalanceFreq = Literal["D", "W", "M"]
ScaleMode = Literal["sign", "zscore"]

@dataclass
class FixedWeightBacktester:
    signal_cols: List[str]
    per_signal_position: float = 0.002
    long_only: bool = False
    max_gross: float = 1.0
    tc_bps: float = 0.0
    initial_capital: float = 1_000_000.0
    price_col: str = "price"
    date_col: str = "date"
    code_col: str = "code"
    rebalance: RebalanceFreq = "D"
    scale: ScaleMode = "sign"
    z_clip: float = 3.0

    _equity: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _daily_ret_net: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _weights_wide: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _turnover: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _summary: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)

    # =========================
    #          PUBLIC
    # =========================
    def run(self, df: pd.DataFrame) -> Dict[str, object]:
        use = self._prep_df(df)
        ret_wide = self._compute_returns(use)
        w_wide   = self._build_weights(use)
        w_wide   = self._cap_gross(w_wide)

        w_prev = w_wide.shift(1).fillna(0.0)
        turnover = (w_wide.sub(w_prev).abs()).sum(axis=1)
        tc_rate = self.tc_bps / 10_000.0
        daily_tc = turnover * tc_rate

        ret_wide = ret_wide.reindex_like(w_wide)
        port_ret_gross = (w_wide * ret_wide).sum(axis=1)
        port_ret_net = port_ret_gross - daily_tc

        equity = (1.0 + port_ret_net.fillna(0.0)).cumprod() * self.initial_capital

        # 기본 summary는 아래 성능 메서드들을 재사용해 계산
        summary = dict(
            start = equity.index.min(),
            end   = equity.index.max(),
            CAGR  = self._cagr_from_equity(equity),
            AnnVol = port_ret_net.std(ddof=1) * np.sqrt(self._infer_periods_per_year(equity.index)),
            Sharpe = self._sharpe_from_returns(port_ret_net),
            MaxDrawdown = self._mdd_from_equity(equity),
            AvgDailyTurnover = turnover.mean()
        )

        self._equity = equity
        self._daily_ret_net = port_ret_net.rename("r_port")
        self._weights_wide = w_wide
        self._turnover = turnover.rename("turnover")
        self._summary = summary

        return {
            "equity_curve": equity.rename("equity").to_frame(),
            "daily_return_net": self._daily_ret_net.to_frame(),
            "weights_wide": self._weights_wide,
            "turnover": self._turnover.to_frame(),
            "summary": summary
        }

    # =========================
    #   PERFORMANCE METHODS
    # =========================
    def sharpe_ratio(
        self,
        rf_rate_annual: float = 0.0,
        periods_per_year: Optional[int] = None,
        use_net: bool = True
    ) -> float:
        """
        연 Sharpe. rf_rate_annual은 연 환산 무위험수익률(소수, 예: 0.03=3%).
        periods_per_year 미지정 시 인덱스 빈도로 추정.
        """
        self._require_run()
        r = self._daily_ret_net if use_net else self._gross_returns()
        ppy = periods_per_year or self._infer_periods_per_year(self._equity.index)
        rf_per_period = rf_rate_annual / ppy
        excess = r - rf_per_period
        mu = excess.mean() * ppy
        sd = r.std(ddof=1) * np.sqrt(ppy)
        return np.nan if sd == 0 or not np.isfinite(sd) else mu / sd

    def max_drawdown(self) -> float:
        """최대낙폭(MDD, 음수 비율)."""
        self._require_run()
        return self._mdd_from_equity(self._equity)

    def cagr(self) -> float:
        """CAGR (연복리수익률)."""
        self._require_run()
        return self._cagr_from_equity(self._equity)

    def rolling_sharpe(
        self,
        window: int = 126,
        rf_rate_annual: float = 0.0,
        periods_per_year: Optional[int] = None,
        use_net: bool = True
    ) -> pd.Series:
        """
        롤링 샤프(윈도우 길이는 기간 수 기준).
        """
        self._require_run()
        r = self._daily_ret_net if use_net else self._gross_returns()
        ppy = periods_per_year or self._infer_periods_per_year(self._equity.index)
        rf_per_period = rf_rate_annual / ppy

        def _roll(x):
            if len(x) < 2:
                return np.nan
            excess = x - rf_per_period
            mu = excess.mean() * ppy
            sd = x.std(ddof=1) * np.sqrt(ppy)
            return np.nan if sd == 0 or not np.isfinite(sd) else mu / sd

        return r.rolling(window).apply(_roll, raw=False)

    def performance_report(self) -> Dict[str, float]:
        """핵심 지표를 한 번에 반환."""
        self._require_run()
        ppy = self._infer_periods_per_year(self._equity.index)
        return {
            "CAGR": self.cagr(),
            "AnnVol": self._daily_ret_net.std(ddof=1) * np.sqrt(ppy),
            "Sharpe": self.sharpe_ratio(),
            "MaxDrawdown": self.max_drawdown(),
            "AvgDailyTurnover": float(self._turnover.mean())
        }

    # =========================
    #        INTERNAL
    # =========================
    def _require_run(self):
        if self._equity is None or self._daily_ret_net is None:
            raise RuntimeError("먼저 run(df)를 실행하세요.")

    def _prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        need = {self.date_col, self.code_col, self.price_col} | set(self.signal_cols)
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"입력 df에 필요한 컬럼이 없습니다: {sorted(miss)}")
        use = df.copy()
        use[self.date_col] = pd.to_datetime(use[self.date_col], errors="coerce")
        use = use.dropna(subset=[self.date_col]).sort_values([self.code_col, self.date_col])
        return use

    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp["ret"] = tmp.groupby(self.code_col, sort=False)[self.price_col].pct_change()
        return tmp.pivot(index=self.date_col, columns=self.code_col, values="ret")

    def _rebal_dates(self, all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if self.rebalance == "D":
            return all_dates
        if self.rebalance == "W":
            return all_dates.to_series().groupby(all_dates.to_period("W")).tail(1).index
        if self.rebalance == "M":
            return all_dates.to_series().groupby(all_dates.to_period("M")).tail(1).index
        raise ValueError("rebalance must be 'D' | 'W' | 'M'.")

    def _build_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        use = df.copy()
        lag_cols = []
        for c in self.signal_cols:
            lc = f"{c}_lag1"
            use[lc] = use.groupby(self.code_col, sort=False)[c].shift(1)
            lag_cols.append(lc)

        all_dates = use[self.date_col].drop_duplicates().sort_values()
        rbd = self._rebal_dates(all_dates)
        snap = use[use[self.date_col].isin(rbd)].copy()

        if self.scale == "sign":
            for lc in lag_cols:
                snap[lc] = np.sign(snap[lc].fillna(0.0))
                if self.long_only:
                    snap[lc] = np.where(snap[lc] > 0, 1.0, 0.0)
        elif self.scale == "zscore":
            def _z(x):
                x = x.astype(float)
                m = x.mean()
                s = x.std(ddof=1)
                if s == 0 or not np.isfinite(s):
                    return pd.Series(np.zeros(len(x)), index=x.index)
                z = (x - m) / s
                z = z.clip(-self.z_clip, self.z_clip) / self.z_clip
                return z.fillna(0.0)
            for lc in lag_cols:
                snap[lc] = snap.groupby(self.date_col, sort=False)[lc].transform(_z)
            if self.long_only:
                for lc in lag_cols:
                    snap[lc] = snap[lc].clip(lower=0.0)
        else:
            raise ValueError("scale must be 'sign' or 'zscore'.")

        def _row_weight(row: pd.Series) -> float:
            vals = row[lag_cols].values.astype(float)
            return self.per_signal_position * np.nansum(vals)

        snap["w_raw"] = snap[lag_cols].apply(_row_weight, axis=1)

        w_rb = snap.pivot(index=self.date_col, columns=self.code_col, values="w_raw").sort_index()
        w_rb = self._cap_gross(w_rb)
        w_all = w_rb.reindex(all_dates).ffill().fillna(0.0)
        return w_all

    def _cap_gross(self, w_wide: pd.DataFrame) -> pd.DataFrame:
        W = w_wide.to_numpy()
        for i in range(W.shape[0]):
            gross = np.nansum(np.abs(W[i, :]))
            if np.isfinite(gross) and gross > 0 and gross > self.max_gross:
                W[i, :] = W[i, :] * (self.max_gross / gross)
        return pd.DataFrame(W, index=w_wide.index, columns=w_wide.columns)

    # ---- helpers for performance ----
    def _infer_periods_per_year(self, idx: pd.DatetimeIndex) -> int:
        """인덱스 빈도로 연환산 계수 추정 (기본 252)."""
        if len(idx) < 2:
            return 252
        try:
            freq = pd.infer_freq(idx)
        except Exception:
            freq = None
        if freq is None:
            # 평균 간격으로 추정
            avg_days = np.mean(np.diff(idx.values).astype("timedelta64[D]").astype(float))
            if avg_days <= 2:   # ~일별
                return 252
            if avg_days <= 8:   # ~주별
                return 52
            return 12           # ~월별
        if freq.startswith("B") or freq == "D":
            return 252
        if freq.startswith("W"):
            return 52
        if freq.startswith("M"):
            return 12
        return 252

    def _sharpe_from_returns(self, r: pd.Series, rf_rate_annual: float = 0.0) -> float:
        ppy = self._infer_periods_per_year(r.index)
        rf_per_period = rf_rate_annual / ppy
        excess = r - rf_per_period
        mu = excess.mean() * ppy
        sd = r.std(ddof=1) * np.sqrt(ppy)
        return np.nan if sd == 0 or not np.isfinite(sd) else mu / sd

    def _mdd_from_equity(self, equity: pd.Series) -> float:
        roll_max = equity.cummax()
        dd = equity / roll_max - 1.0
        return float(dd.min())

    def _cagr_from_equity(self, equity: pd.Series) -> float:
        if len(equity) < 2:
            return np.nan
        start_val = float(equity.iloc[0])
        end_val = float(equity.iloc[-1])
        if start_val <= 0:
            return np.nan
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        if years <= 0:
            return np.nan
        return (end_val / start_val) ** (1.0 / years) - 1.0

    def _gross_returns(self) -> pd.Series:
        """거래비용 차감 전 포트 수익률(참고용)."""
        # gross = net + tc
        # run() 이후에만 의미가 있음
        self._require_run()
        ppy = self._infer_periods_per_year(self._equity.index)
        tc_rate = self.tc_bps / 10_000.0
        tc_series = self._turnover * tc_rate
        return self._daily_ret_net + tc_series
    
    
    
if __name__=='__main__':
    bt = FixedWeightBacktester(
        signal_cols=['signal1','signal2'],
        per_signal_position=0.002,
        long_only=False,
        max_gross=1.0,
        tc_bps=2.0,
        initial_capital=1_000_000,
        rebalance="W",
        scale="sign"
    )
    try:
        df=0
        bt.run(df)

        # 개별 지표
        sr   = bt.sharpe_ratio()         # 연 Sharpe (rf=0%)
        mdd  = bt.max_drawdown()         # 최대낙폭(음수)
        cg   = bt.cagr()                 # CAGR

        # 롤링 샤프(6개월≈126거래일 기준)
        rs = bt.rolling_sharpe(window=126)

        # 한 번에 보고서
        perf = bt.performance_report()
        print(perf)
    except Exception as e:
        print(e)

