import pandas as pd
import numpy as np
import warnings
from scipy.stats import f, norm
import statsmodels.api as sm
from tqdm import tqdm
from statsmodels.stats.sandwich_covariance import cov_cluster
import plotly.express as px


#### 나머지 에셋 프라이싱 테스트는 팩터 포트폴리오에 대해서 테스트, 하지만 파마-멕베스는 원래 기업특성들과 개별 주식 수익률에 대해서 테스트. 따라서 클래스를 분리할 필요가 있음. 

warnings.filterwarnings('ignore')

class asset_pricing_test:

    def __init__(self, df, y):
        """
        df: 요인(factor)과 날짜 정보를 담은 DataFrame
            예) columns = ['date', 'code', 'mkt_rf', 'smb', 'hml', ...]
        y : 종속변수(각 자산/포트폴리오 수익률) DataFrame(혹은 pivot 형태)
            예) columns = 여러 종목(포트폴리오) 이름
            index = 날짜(또는 일련번호)
        """
        self.df = df
        self.y = y  # self.self.y -> self.y 로 수정
        # Newey-West 표준오차를 위한 lag 설정(예: HAC)
        self.lags = int(4 * (len(self.df) / 100) ** (4 / 25))
        

    def GRS_test(self, factors):
        """
        GRS(그랜저-롤-쉐이퍼) 테스트
        - factors: ['mkt_rf', 'smb', 'hml'] 등 요인 컬럼 리스트
        """
        alpha_market = []
        beta_market = []
        eps_market = []
        alpha_market_pval = []
        beta_market_pval = []

        # 요인 데이터(중복 제거 및 상수항 추가)
        factor_df = self.df.drop_duplicates(subset=['date'])[factors].reset_index(drop=True)
        x = sm.add_constant(factor_df[factors])

        # y(포트폴리오) 컬럼별로 OLS 실행
        for col in tqdm(self.y.columns):
            y_in = self.y[col].values  # 결측치 처리는 sm.OLS에서 missing='drop'으로
            model_market = sm.OLS(y_in, x, missing='drop').fit(
                cov_type='HAC',
                cov_kwds={'maxlags': self.lags}
            )
            alpha_market.append(model_market.params[0])
            beta_market.append(model_market.params[1])  # 첫 번째 factor에 대한 베타
            eps_market.append(model_market.resid)
            alpha_market_pval.append(model_market.pvalues[0])
            beta_market_pval.append(model_market.pvalues[1])

        # 시점 개수 T, 포트폴리오(자산) 개수 N
        T = self.df['date'].nunique()   # 날짜 수
        N = len(self.y.columns)        # 포트폴리오(자산) 수

        # 잔차 행렬: eps_market는 리스트(각 자산별 resid) -> shape: N x T
        eps_df = pd.DataFrame(eps_market)  # 행 = 자산, 열 = 시점
        var_cov = eps_df.cov()            # NxN 공분산행렬

        # alpha를 ndarray로 변환
        alpha_arr = np.array(alpha_market).reshape(-1, 1)  # shape: Nx1

        # (MKT-RF)의 평균/표준편차 → factors[0]이 'mkt_rf'라고 가정
        # 만약 첫 factor가 mkt_rf 가 아니라면 적절히 수정
        mkt_mean = factor_df[factors[0]].mean()
        mkt_std = factor_df[factors[0]].std()
        denom = 1 + (mkt_mean / mkt_std) ** 2 if mkt_std != 0 else 1.0

        # GRS 통계량
        # F = ((T - N - 1) / N) * [alpha' Σ^-1 alpha] / denom
        # Σ = var_cov
        F_stat = ((T - N - 1) / N) * (
            (alpha_arr.T @ np.linalg.inv(var_cov) @ alpha_arr) / denom
        )
        F_stat = float(F_stat)  # scalar

        # p-value
        p_value = f.sf(F_stat, N, T - N - 1)

        # 추정계수 요약표
        coeff_summary = pd.DataFrame({
            'portfolio': self.y.columns,
            'alpha': alpha_market,
            'beta': beta_market,
            'alpha_pval': alpha_market_pval,
            'beta_pval': beta_market_pval
        })

        return p_value, F_stat, coeff_summary

    def fama_french_ts_regression(self, factors):
        """
        Fama-French 시계열 회귀(포트폴리오별로 시계열 OLS)
        - factors: ['mkt_rf', 'smb', 'hml'] 등
        """
        # 상수항 + 요인
        factor_df = self.df.drop_duplicates(subset=['date'])[factors].reset_index(drop=True)
        x = sm.add_constant(factor_df[factors])

        k = len(factors) + 1  # alpha 포함한 계수 개수
        coeffs = [[] for _ in range(k)]
        pvals = [[] for _ in range(k)]
        tstats = [[] for _ in range(k)]

        for col in tqdm(self.y.columns):
            y_in = self.y[col].values
            model = sm.OLS(y_in, x, missing='drop').fit(
                cov_type='HAC',
                cov_kwds={'maxlags': self.lags}
            )
            for i in range(k):
                coeffs[i].append(model.params[i])
                pvals[i].append(model.pvalues[i])
                tstats[i].append(model.tvalues[i])

        # 컬럼명(alpha, factor1, factor2, ...)
        named_factors = ['alpha'] + factors
        coeff_dict = {f'{named_factors[i]}_coeff': coeffs[i] for i in range(k)}
        pvals_dict = {f'{named_factors[i]}_pval': pvals[i] for i in range(k)}
        tstats_dict = {f'{named_factors[i]}_tstat': tstats[i] for i in range(k)}

        result = {**coeff_dict, **pvals_dict, **tstats_dict}
        result['portfolio'] = self.y.columns
        result_df = pd.DataFrame(result)
        return result_df





def fama_macbeth_regression(
    data,
    factors,
    nw_lags=None,
    weight_var=None,
    winsor_q=0.005
):
    
    df = data.copy()
    lambdas_list = []
    r2_list = []
    dates = sorted(df['date'].unique())

    for date in tqdm(dates, desc="Cross-section regressions"):
        grp = df[df['date'] == date]
        cols = ['rtn'] + factors
        if weight_var:
            cols.append(weight_var)
        if 'rf' in grp.columns:
            cols.append('rf')

        tmp = grp[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.shape[0] <= len(factors) + 1:
            continue

        # 1) Winsorize each factor at [winsor_q, 1-winsor_q]
        for f in factors:
            lo = tmp[f].quantile(winsor_q)
            hi = tmp[f].quantile(1 - winsor_q)
            tmp[f] = tmp[f].clip(lower=lo, upper=hi)

        # 2) 표준화(Standardize each factor)
        for f in factors:
            m, s = tmp[f].mean(), tmp[f].std(ddof=0)
            tmp[f] = (tmp[f] - m) / s if s != 0 else tmp[f] - m

        # 3) 회귀 접수
        y = tmp['rtn'] - tmp.get('rf', 0)
        X = sm.add_constant(tmp[factors])
        model = (
            sm.WLS(y, X, weights=tmp[weight_var])
            if weight_var else sm.OLS(y, X)
        )
        res = model.fit()
        lambdas_list.append(res.params.values)
        r2_list.append(res.rsquared)

    if not lambdas_list:
        raise ValueError("No valid cross-sectional regressions.")

    lambdas_arr = np.vstack(lambdas_list)
    T, K1 = lambdas_arr.shape
    mean_lambda = lambdas_arr.mean(axis=0)
    mean_R2     = float(np.nanmean(r2_list))

    # Newey–West lag auto
    if nw_lags is None:
        nw_lags = int(4 * (T / 100) ** (4 / 25))
    nw_lags = max(nw_lags, 1)

    # 시간축 회귀로 SE, t, p 계산
    Xc = np.ones((T, 1))
    ses, ts, ps = np.zeros(K1), np.zeros(K1), np.zeros(K1)
    for j in range(K1):
        yj = lambdas_arr[:, j]
        ts_res = sm.OLS(yj, Xc).fit(
            cov_type='HAC', cov_kwds={'maxlags': nw_lags}
        )
        ses[j] = ts_res.bse[0]
        ts[j]  = mean_lambda[j] / ses[j]
        ps[j]  = ts_res.pvalues[0]

    result = pd.DataFrame({
        'factor': ['alpha'] + factors,
        'lambda': mean_lambda,
        'se': ses,
        't': ts,
        'p': ps
    })
    result[['lambda','se','t','p']] = result[['lambda','se','t','p']].round(4)
    return result, round(mean_R2, 4)

