import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

class ex_post_portfolio_optimizer:
    def __init__(self, df, universe, risk_free_rate=None, allow_short=False, annualize_factor=12):
        """
        Parameters
        ----------
        df : pd.DataFrame
            각 자산(또는 요인)의 월간(또는 주간/일간) 수익률이 index(날짜)와 columns(자산명)로 들어있는 DataFrame
        universe : list
            최적화에 사용할 자산(또는 요인) 리스트 (예: ['Market', 'Size', 'Value', 'Momentum', 'CbOP'])
        risk_free_rate : float, pd.Series, or pd.DataFrame, optional
            무위험수익률. 스칼라로 입력하지 않고 시계열(Series/DataFrame)로 입력하면 평균값을 사용합니다.
            기본값은 None인 경우 0으로 사용합니다.
        allow_short : bool
            공매도 허용 여부. False이면 가중치가 [0,1] 범위, True이면 제약 없음.
        annualize_factor : int or float
            연환산 계수 (예: 월간 데이터이면 12, 일간이면 252)
        """
        self.df = df[universe].dropna()
        self.universe = universe
        self.annualize_factor = annualize_factor
        self.allow_short = allow_short
        
        # risk_free_rate 처리: None이면 0, Series나 DataFrame이면 평균값 사용
        if risk_free_rate is None:
            self.risk_free_rate = 0.0
        elif isinstance(risk_free_rate, (pd.Series, pd.DataFrame)):
            self.risk_free_rate = risk_free_rate.mean().item()
        else:
            self.risk_free_rate = risk_free_rate
        
        # 역사적 평균 수익률 및 공분산행렬 (ex post)
        self.mean_returns = self.df.mean()
        self.cov_matrix = self.df.cov()
        
        # 최적화 결과 저장 변수
        self.max_sharpe_weights = None
        self.min_var_weights = None

    def portfolio_performance(self, weights):
        """
        기대(역사적 평균) 수익률, 변동성, Sharpe ratio를 계산 (ex ante)
        """
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (ret - self.risk_free_rate) / vol if vol != 0 else 0
        return ret, vol, sharpe

    def realized_performance(self, weights):
        """
        입력 가중치를 실제(실현) 포트폴리오 수익률 시계열에 적용하여,
        평균 수익률, 표준편차, Sharpe ratio 및 연환산 성과를 계산합니다.
        """
        port_returns = self.df.dot(weights)
        avg_return = port_returns.mean()
        vol = port_returns.std()
        sharpe = (avg_return - self.risk_free_rate) / vol if vol != 0 else 0
        
        # 연환산 (예: 월간 데이터 기준)
        avg_return_annual = avg_return * self.annualize_factor
        vol_annual = vol * np.sqrt(self.annualize_factor)
        sharpe_annual = (avg_return_annual - self.risk_free_rate) / vol_annual if vol_annual != 0 else 0
        
        return avg_return, vol, sharpe, avg_return_annual, vol_annual, sharpe_annual

    def negative_sharpe(self, weights):
        """
        최적화 시 샤프 지수 최대화를 위해 음수의 샤프 지수를 최소화하는 함수
        """
        return -self.portfolio_performance(weights)[2]

    def portfolio_variance(self, weights):
        """
        포트폴리오 분산을 계산하는 함수
        """
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    def optimize_max_sharpe(self):
        """
        ex post 최대 Sharpe ratio 포트폴리오(즉, tangency portfolio)를 최적화합니다.
        제약조건: 가중치 합 = 1
        """
        num_assets = len(self.mean_returns)
        init_guess = np.repeat(1/num_assets, num_assets)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = None if self.allow_short else [(0, 1) for _ in range(num_assets)]
            
        result = minimize(self.negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        self.max_sharpe_weights = result.x
        return result

    def optimize_min_variance(self):
        """
        ex post 최소 분산 포트폴리오를 최적화합니다.
        제약조건: 가중치 합 = 1
        """
        num_assets = len(self.mean_returns)
        init_guess = np.repeat(1/num_assets, num_assets)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = None if self.allow_short else [(0, 1) for _ in range(num_assets)]
            
        result = minimize(self.portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        self.min_var_weights = result.x
        return result

    def run_optimization(self, mode='both'):
        """
        mode에 따라 ex post 최대 샤프 포트폴리오, 최소 분산 포트폴리오 또는 둘 다를 최적화하고,
        예상(역사적 평균) 및 실현(표본) 성과를 출력합니다.
        
        mode : str, 'max_sharpe' / 'min_variance' / 'both'
        """
        results = {}
        if mode in ['max_sharpe', 'both']:
            ms_res = self.optimize_max_sharpe()
            exp_ret, exp_vol, exp_sharpe = self.portfolio_performance(ms_res.x)
            (real_avg, real_vol, real_sharpe,
             ann_avg, ann_vol, ann_sharpe) = self.realized_performance(ms_res.x)
            results['max_sharpe'] = {
                'weights': ms_res.x,
                'expected_return': exp_ret,
                'expected_volatility': exp_vol,
                'expected_sharpe': exp_sharpe,
                'realized_avg_return': real_avg,
                'realized_volatility': real_vol,
                'realized_sharpe': real_sharpe,
                'annualized_avg_return': ann_avg,
                'annualized_volatility': ann_vol,
                'annualized_sharpe': ann_sharpe
            }
        if mode in ['min_variance', 'both']:
            mv_res = self.optimize_min_variance()
            exp_ret, exp_vol, exp_sharpe = self.portfolio_performance(mv_res.x)
            (real_avg, real_vol, real_sharpe,
             ann_avg, ann_vol, ann_sharpe) = self.realized_performance(mv_res.x)
            results['min_variance'] = {
                'weights': mv_res.x,
                'expected_return': exp_ret,
                'expected_volatility': exp_vol,
                'expected_sharpe': exp_sharpe,
                'realized_avg_return': real_avg,
                'realized_volatility': real_vol,
                'realized_sharpe': real_sharpe,
                'annualized_avg_return': ann_avg,
                'annualized_volatility': ann_vol,
                'annualized_sharpe': ann_sharpe
            }
        
        # ex post 결과 출력 (Table 8 참조)
        if 'max_sharpe' in results:
            print("Ex Post Maximum Sharpe Ratio Portfolio:")
            print("Weights:", results['max_sharpe']['weights'])
            print("Expected (Historical) -> Return: {:.4f}, Volatility: {:.4f}, Sharpe: {:.4f}".format(
                results['max_sharpe']['expected_return'],
                results['max_sharpe']['expected_volatility'],
                results['max_sharpe']['expected_sharpe']
            ))
            print("Realized (Sample, Annualized) -> Return: {:.4f}, Volatility: {:.4f}, Sharpe: {:.4f}".format(
                results['max_sharpe']['annualized_avg_return'],
                results['max_sharpe']['annualized_volatility'],
                results['max_sharpe']['annualized_sharpe']
            ))
            print()
        if 'min_variance' in results:
            print("Ex Post Minimum Variance Portfolio:")
            print("Weights:", results['min_variance']['weights'])
            print("Expected (Historical) -> Return: {:.4f}, Volatility: {:.4f}, Sharpe: {:.4f}".format(
                results['min_variance']['expected_return'],
                results['min_variance']['expected_volatility'],
                results['min_variance']['expected_sharpe']
            ))
            print("Realized (Sample, Annualized) -> Return: {:.4f}, Volatility: {:.4f}, Sharpe: {:.4f}".format(
                results['min_variance']['annualized_avg_return'],
                results['min_variance']['annualized_volatility'],
                results['min_variance']['annualized_sharpe']
            ))
        return results

    def plot_efficient_frontier(self, n_points=50):
        """
        ex post 효율적 프론티어를 계산하여 plotly를 이용해 시각화합니다.
        (각 목표 수익률을 만족하는 최소 분산 포트폴리오를 구하는 방식)
        """
        num_assets = len(self.mean_returns)
        base_returns = self.df.dot(np.repeat(1/num_assets, num_assets))
        target_returns = np.linspace(base_returns.min(), base_returns.max(), n_points)
        frontier_vols = []
        frontier_expret = []
        for tr in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - tr}
            )
            bounds = None if self.allow_short else [(0, 1) for _ in range(num_assets)]
            init_guess = np.repeat(1/num_assets, num_assets)
            res = minimize(self.portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            if res.success:
                w = res.x
                frontier_expret.append(np.dot(w, self.mean_returns))
                frontier_vols.append(np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))))
        
        # plotly 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frontier_vols, y=frontier_expret, mode='lines', name='Efficient Frontier'))
        if self.max_sharpe_weights is not None:
            ms_vol = np.sqrt(np.dot(self.max_sharpe_weights.T, np.dot(self.cov_matrix, self.max_sharpe_weights)))
            ms_ret = np.dot(self.max_sharpe_weights, self.mean_returns)
            fig.add_trace(go.Scatter(x=[ms_vol], y=[ms_ret],
                                     mode='markers', marker=dict(size=12, color='red'),
                                     name='Max Sharpe'))
        if self.min_var_weights is not None:
            mv_vol = np.sqrt(np.dot(self.min_var_weights.T, np.dot(self.cov_matrix, self.min_var_weights)))
            mv_ret = np.dot(self.min_var_weights, self.mean_returns)
            fig.add_trace(go.Scatter(x=[mv_vol], y=[mv_ret],
                                     mode='markers', marker=dict(size=12, color='green'),
                                     name='Min Variance'))
        fig.update_layout(title="Ex Post Efficient Frontier",
                          xaxis_title="Volatility",
                          yaxis_title="Expected Return")
        fig.show()
if __name__=='__main__':
    # -------------------------------
    # 예시 데이터 생성

    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=60, freq='M')  # 60개월 데이터
    assets = ['Market', 'Size', 'Value', 'Momentum', 'CbOP']

    # 임의의 월간 수익률 데이터 생성 (평균 0.01, 표준편차 0.05)
    data = np.random.normal(0.01, 0.05, size=(60, len(assets)))
    df_sample = pd.DataFrame(data, index=dates, columns=assets)

    # 예시: risk_free_rate를 Series로 생성 (예: 매월 0.002의 무위험 수익률)
    risk_free_series = pd.Series(0.002, index=dates)

    # -------------------------------
    # ex_post_portfolio_optimizer 실행 예시
    # -------------------------------
    optimizer = ex_post_portfolio_optimizer(df_sample, universe=assets,
                                        risk_free_rate=risk_free_series,
                                        allow_short=False, annualize_factor=12)
    results = optimizer.run_optimization(mode='both')
    optimizer.plot_efficient_frontier(n_points=50)
