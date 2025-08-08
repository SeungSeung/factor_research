import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import plotly.express as px

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import plotly.express as px

warnings.filterwarnings('ignore')

###논문 용 러이브러리 independent sort,dependent sort 구현
class backtest:

    def __init__(self, code, factor_df, quantile_1, quantile_2, factor_1, factor_2, break_points1=False, break_points2=False):
        self.code = code        
        self.factor_df = factor_df.copy()
        self.quantile_1 = quantile_1
        self.quantile_2 = quantile_2
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.test = None  # 정렬 결과를 저장할 공간
        self.break_points1 = break_points1  # 첫 번째 팩터의 브레이크포인트 옵션 
        self.break_points2 = break_points2  # 두 번째 팩터의 브레이크포인트 옵션 

    def winsorizing(self, factor_list, q):
        """
        지정된 팩터 컬럼을 날짜별로 클리핑하여 윈저라이징
        하위 q, 상위 1-q 분위수에서 값을 제한합니다.
        """
        self.factor_df[factor_list] = self.factor_df.groupby('date')[factor_list].transform(
            lambda x: x.clip(lower=x.quantile(q), upper=x.quantile(1-q))
        )

    def assign_scores(self, df, factor, quantile_list, break_points):
        """
        팩터 값에 따라 1에서 len(quantile_list)+1까지 점수를 부여합니다.
        break_points=True인 경우 KOSPI 종목만 사용해 분위수를 계산합니다.-> KOSPI break point
        """
        # 분위수 계산 대상 시리즈 선택
        if not break_points:
            quants = df[factor].quantile(q=quantile_list)
        else:
            quants = df.loc[df['exchange'] == '유가증권시장', factor].quantile(q=quantile_list)

        scores = pd.Series(index=df.index, dtype=float)
        # 첫 번째 구간 (최하위)
        scores[df[factor] <= quants.iloc[0]] = 1
        # 중간 구간
        for i in range(1, len(quantile_list)):
            lower = quants.iloc[i-1]
            upper = quants.iloc[i]
            scores[(df[factor] >= lower) & (df[factor] <= upper)] = i + 1
        # 마지막 구간 (최상위 초과)
        scores[df[factor] >= quants.iloc[-1]] = len(quantile_list) + 1
        return scores

    def sorting(self, independent_sort=True, lagging1=0, lagging2=0):
     
        self.test = self.factor_df.copy()
        # 시차 적용
        if lagging1!=0:
            self.test[self.factor_1] = self.test.groupby(self.code)[self.factor_1].shift(lagging1)
        if lagging2!=0:
            self.test[self.factor_2] = self.test.groupby(self.code)[self.factor_2].shift(lagging2)

        # 1차 점수 (날짜별)
        self.test['score'] = self.test.groupby('date').apply(
            lambda grp: self.assign_scores(grp, self.factor_1, self.quantile_1, self.break_points1)
        ).reset_index(level=0, drop=True)

        # 2차 점수 계산용 그룹 기준 설정
        if independent_sort:
            grp_on = ['date']  # 각 팩터와 상관없이 소트
        else:
            grp_on = ['date', 'score']  # 1차 팩터로 소팅하고 각 소팅 그룹안에서 다시 2차 팩터로 소팅

        # 2차 점수
        self.test['score2'] = self.test.groupby(grp_on).apply(
            lambda grp: self.assign_scores(grp, self.factor_2, self.quantile_2, self.break_points2)
        ).reset_index(level=grp_on, drop=True)

    def run(self, score1, score2, value_weighted=True):
     
        if self.test is None:
            raise ValueError("먼저 sorting() 메서드를 통해 self.test를 생성하세요.")
        # 해당 조합 표시기 생성
        self.test['indicator'] = np.where(
            (self.test['score'] == score1) & (self.test['score2'] == score2), 1, np.nan
        )
       
        # 시가총액 가중 포트폴리오 가중치 계산
        if value_weighted:
            #self.test['size_1'] = self.test.groupby(self.code)['cap'].shift(1)
            v = self.test[self.test['indicator'] == 1].copy()
            if 'me_lag1' not in v.columns:
                v['me_lag1']=v.groupby('code')['me'].shift(1)
            
            v['weight'] = v.groupby('date')['me_lag1'].transform(lambda x: x / x.sum())
            self.port = self.test.merge(
                v[['date', self.code, 'weight']], on=['date', self.code], how='left'
            )
        else:
            # 동일 가중
            self.port = self.test.copy()
            self.port['weight'] = self.port.groupby('date')['indicator'].transform(lambda x: x / x.count())

        # 포트폴리오 일별 수익률 계산
        self.port['port_rtn'] = self.port['rtn'] * self.port['weight']
        self.port_rtn = self.port.groupby('date')['port_rtn'].sum()
        
        return self.port_rtn

    def performance_evaluation(self, freq=12):
        """
        포트폴리오 성과 지표 계산:
        - 연율화 수익률, 연율화 변동성
        - 샤프 비율, t-통계량
        - 최대 낙폭(MDD), 소르티노 비율
        freq: 월간=12, 일간=252, 주간=52
        """
        if not hasattr(self, 'port_rtn'):
            raise ValueError("먼저 run() 메서드를 실행하여 'port_rtn'을 생성하세요.")
        returns = self.port_rtn.dropna()
        mean_ret = returns.mean()
        std_ret = returns.std()
        # 연율화 수익률 및 변동성
        ann_return = (1 + mean_ret) ** freq - 1
        ann_vol = std_ret * np.sqrt(freq)
        # 샤프 비율 및 t-값
        sharpe_ratio = ann_return / ann_vol if ann_vol else np.nan
        t_value = mean_ret / std_ret if std_ret else np.nan
        # 최대 낙폭
        cum = (1 + returns).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        # 소르티노 비율
        down = returns[returns < 0]
        sortino = ann_return / (down.std() * np.sqrt(freq)) if len(down) else np.nan
        # 결과 반환
        return pd.DataFrame({
            '연율화 수익률': ann_return,
            '연율화 변동성': ann_vol,
            '샤프 비율': sharpe_ratio,
            't-통계량': t_value,
            '최대 낙폭(MDD)': mdd,
            '소르티노 비율': sortino
        }, index=['Portfolio'])
