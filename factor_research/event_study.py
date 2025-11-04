import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import plotly.express as px

warnings.filterwarnings('ignore')

class mean_stock_movement:
    def __init__(self, test, signal_num, end_date, factor_name, minus_end_date, 
                 mean_only=True, lagging=1, weighted=False):
        """
        Parameters
        ----------
        test : pd.DataFrame
            입력 데이터. 필수 컬럼: 'date', 'code', 'price', factor_name, 
            (가중평균 옵션 사용시) 'cap', 그리고 선택적으로 'market', 'trading_suspension', 'listed'
        signal_num : numeric
            factor가 이 값보다 큰 경우 신호(1)로 판단.
        end_date : int
            t0부터 미래로 몇 기간(예: 5일) 수익률을 추적할지 지정.
        factor_name : str
            관심 요인 컬럼명.
        minus_end_date : int
            t0 이전 몇 기간(예: -3일) 수익률을 추적할지 지정.
        mean_only : bool, default True
            True이면 전체 평균 수익률만 반환, False이면 종목별 수익률 흐름(피벗 테이블 전체)을 반환.
        lagging : int, default 1
            신호 시점을 lag 만큼 뒤로 미룸.
        weighted : bool, default False
            True이면 'cap' 컬럼을 사용해 시가총액 가중평균 수익률을 계산.
        """
        self.test = test
        self.signal_num = signal_num
        self.end_date = end_date
        self.factor_name = factor_name
        self.minus_end_date = minus_end_date
        self.mean_only = mean_only
        self.lagging = lagging
        self.weighted = weighted

    def go(self):
        """
        특정 factor(factor_name)가 signal_num보다 큰 종목들을 선택하여,
        t0부터 end_date까지, 그리고 minus_end_date부터 t-1까지의 수익률 흐름을 계산합니다.
        
        - 만약 weighted=True이면, 'cap' 컬럼을 이용한 시가총액 가중평균 수익률을 계산합니다.
        - mean_only=True이면 전체 평균 수익률(시계열)만 반환하고, 아니면 각 종목별 수익률 흐름(피벗 테이블 전체)을 반환합니다.
        그리고 plotly를 이용해 누적수익률 그래프를 그립니다.
        """
        if self.test is None:
            raise ValueError("먼저 데이터프레임(test)을 생성하세요.")

        # 데이터 복사 및 기본 필터링
        df = self.test.copy()
        # 예: 'market' 컬럼이 있는 경우, '외감'이나 'KONEX' 제외
        if 'market' in df.columns:
            df = df[(df['market'] != '외감') & (df['market'] != 'KONEX')]
        if 'trading_suspension' in df.columns:
            df = df[df['trading_suspension'] != 1]
        if 'listed' in df.columns:
            df['listed'].fillna(0, inplace=True)
            df = df[df['listed'] != 0]

        # 포지션 계산: factor가 signal_num보다 크면 1, 아니면 0
        df['position'] = np.where(df[self.factor_name] > self.signal_num, 1, 0)

        # 가격 피벗 테이블: index=date, columns=code
        if 'price' not in df.columns:
            raise ValueError("데이터에 'price' 컬럼이 없습니다.")
        close = pd.pivot_table(df[['date', 'code', 'price']],
                               index='date', columns='code', values='price',
                               aggfunc='first', dropna=False)
        close = close.loc[:, ~close.columns.duplicated()].copy()

        # 신호(pivot): index=date, columns=code, 값은 position
        key_df = pd.pivot_table(df[['date', 'code', 'position']],
                                index='date', columns='code', values='position',
                                aggfunc='first', dropna=False)
        # 신호 lag 적용
        key_df = key_df.shift(self.lagging).iloc[self.lagging:]
        
        # 수익률 계산 (일간 혹은 원하는 기간의 pct_change)
        rtn = close.pct_change().iloc[1:]
        
        # 신호가 1인 종목들을 universe로 선정
        universe = list(df.loc[df['position'] == 1, 'code'].unique())
        rtn = rtn[universe]
        
        # rtn의 인덱스 목록
        rtn_index = list(rtn.index)
        
        # 신호가 발생한 (날짜, 종목) 쌍을 keys에 저장
        keys = []
        for dt in key_df.index[1:]:
            for cd in universe:
                if key_df.at[dt, cd] >= 1:
                    dt_idx = rtn_index.index(dt)
                    if dt_idx >= abs(self.minus_end_date) and (len(rtn_index) - dt_idx) > self.end_date:
                        keys.append((dt, cd))
        
        # t-구간 라벨 생성: 예를 들어, t{-3} ~ t{end_date-1}
        t_labels = [f't{i}' for i in range(self.minus_end_date, self.end_date)]
        columns_codes = sorted(set(cd for (_, cd) in keys))
        mean_flow = pd.DataFrame(index=t_labels, columns=columns_codes)
        
        # 각 (날짜, 종목) 쌍에 대해 수익률 흐름 채우기
        for dt, cd in tqdm(keys, desc="Processing keys"):
            base_idx = rtn_index.index(dt)
            # 미래 구간: t0부터 t(end_date-1)
            future_period = rtn.iloc[base_idx : base_idx + self.end_date][cd]
            future_period = future_period.fillna(method='ffill')
            mean_flow.loc['t0':f't{self.end_date-1}', cd] = future_period.values
            
            # 과거 구간: t(minus_end_date)부터 t-1까지
            if (base_idx + self.minus_end_date) >= 0:
                past_period = rtn.iloc[base_idx + self.minus_end_date : base_idx][cd]
                past_period = past_period.fillna(method='ffill')
                mean_flow.loc[f't{self.minus_end_date}':'t-1', cd] = past_period.values
        
        # NaN만 있는 컬럼 제거
        mean_flow.dropna(axis=1, how='all', inplace=True)
        # 단순 평균 컬럼 추가 (종목별 단순 산술평균)
        mean_flow['mean'] = mean_flow.mean(axis=1, skipna=True)
        self.mean_flow = mean_flow

        # 가중평균 옵션: weighted=True이면 'cap' 컬럼으로 가중평균 계산
        if self.weighted:
            if 'cap' not in df.columns:
                raise ValueError("가중평균 옵션 사용 시 'cap' 컬럼이 필요합니다.")
            cap = pd.pivot_table(df[['date', 'code', 'cap']],
                                 index='date', columns='code', values='cap',
                                 aggfunc='first', dropna=False)
            cap = cap.loc[:, ~cap.columns.duplicated()].copy()
            cap = cap[universe]
            # rtn와 날짜 정렬 맞추기
            cap = cap.loc[rtn.index]
            cap_sum = cap.sum(axis=1)
            cap_weights = cap.div(cap_sum, axis=0)
            weighted_rtn = (rtn * cap_weights).sum(axis=1)
            weighted_mean = weighted_rtn
            
            # Plot: 누적수익률 (가중평균)
            cum_return = (1 + weighted_mean.fillna(0)).cumprod()
            fig = px.line(cum_return, title="Weighted Cumulative Return")
            fig.show()
            return weighted_mean
        else:
            # Plot: 누적수익률 (단순 평균 또는 전체 종목별)
            if not self.mean_only:
                cum_flow = (1 + mean_flow.fillna(0)).cumprod()
                base = cum_flow.loc['t0']
                normalized = cum_flow.subtract(base, axis=1)
                fig = px.line(normalized, title="Cumulative Return Flow (All Stocks)")
                fig.show()
            else:
                only_mean = mean_flow['mean']
                cum_mean = (1 + only_mean.fillna(0)).cumprod()
                base = cum_mean.loc['t0']
                normalized = cum_mean - base
                fig = px.line(normalized, title="Cumulative Return Flow (Mean)")
                fig.show()
                return only_mean

# -------------------------------
# 테스트용 데이터 생성
# -------------------------------
# 생성할 날짜: 60일 (예: 2020-01-01부터 2020-03-01)
dates = pd.date_range(start='2020-01-01', periods=60, freq='D')
# 테스트 종목 코드
codes = ['A', 'B', 'C']
# 각 날짜, 종목별 행 생성
rows = []
np.random.seed(42)
for dt in dates:
    for cd in codes:
        row = {
            'date': dt,
            'code': cd,
            # 가격은 100에서 시작해 매일 약간의 변화
            'price': 100 + np.random.randn() * 2,
            # factor 값: 임의의 정규분포 값 (예: 0 ~ 1 사이)
            'factor': np.random.uniform(0, 1),
            # 시가총액: 50~150 사이의 임의 값
            'cap': np.random.uniform(50, 150),
            # market: 모두 'KOSPI'
            'market': 'KOSPI',
            # trading_suspension: 0 (정상 거래)
            'trading_suspension': 0,
            # listed: 1 (상장)
            'listed': 1
        }
        rows.append(row)
test_df = pd.DataFrame(rows)

if __name__=='__main__':
    # -------------------------------
    # 클래스 실행 예시
    # -------------------------------
    # 예: factor 값이 0.7보다 크면 신호로 판단, 미래 5일, 과거 3일, lag=1, 가중평균 옵션 사용
    msm = mean_stock_movement(test=test_df, signal_num=0.7, end_date=5, factor_name='factor', 
                            minus_end_date=-3, mean_only=True, lagging=1, weighted=True)
    weighted_mean_return = msm.go()

    # 만약 가중평균 옵션 없이 단순 평균 수익률 흐름을 보고 싶다면 weighted=False
    msm_simple = mean_stock_movement(test=test_df, signal_num=0.7, end_date=5, factor_name='factor', 
                                minus_end_date=-3, mean_only=True, lagging=1, weighted=False)
    simple_mean_return = msm_simple.go()
