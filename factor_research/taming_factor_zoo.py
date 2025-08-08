import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

class GlobalFactorSelector:
    """
    GlobalFactorSelector 클래스는 "Taming the Global Factor Zoo" 논문의 반복적 두 단계 LASSO
    방법을 기반으로, 글로벌 요인 모형에 포함할 후보 요인들을 순차적으로 선택하는 알고리즘을 구현합니다.
    
    입력 변수:
      r_bar: (n,) array
          각 시점별 테스트 자산(예: 팩터 포트폴리오)의 횡단면 평균 수익률 (demeaned)
      H: (n, m) array
          초기 H 요인 행렬 (예: CAPM의 시장 요인 등)
      G: (n, k) array
          후보 요인 행렬 (예: 152개의 후보 요인)
      candidate_categories: 길이 k의 array
          각 후보 요인에 할당된 범주 (예: 1부터 13까지의 정수). 같은 범주의 후보들은 한 번 선택되면 모두 제거됩니다.
      t_threshold: float, 기본값 1.64
          후보 요인의 t-값 임계치 (t > 1.64이면 유의하다고 판단)
      cv: int, 기본값 5
          LassoCV의 교차검증 fold 수
      random_state: int, 기본값 0
          재현성을 위한 랜덤 시드
      tol: float, 기본값 1e-6
          LASSO 계수가 0으로 간주되는 임계값
    """
    def __init__(self, r_bar, H, G, candidate_categories, t_threshold=1.64, cv=5, random_state=0, tol=1e-6):
        self.r_bar = r_bar
        self.H = H
        self.G = G
        self.candidate_categories = candidate_categories
        self.t_threshold = t_threshold
        self.cv = cv
        self.random_state = random_state
        self.tol = tol

    def first_stage_lasso(self, H_current):
        """
        첫 번째 LASSO 단계: 현재 H_current에 대해 r_bar ~ H_current 회귀를 수행하여,
        계수가 0이 아닌 변수들의 인덱스를 선택합니다.
        """
        lasso_cv = LassoCV(cv=self.cv, random_state=self.random_state).fit(H_current, self.r_bar)
        coef = lasso_cv.coef_
        indices = [i for i, c in enumerate(coef) if np.abs(c) > self.tol]
        return indices, lasso_cv

    def second_stage_lasso(self, g_candidate, H_current):
        """
        두 번째 LASSO 단계: 단일 후보 요인 g_candidate에 대해 r_bar ~ H_current 회귀를 수행하여,
        g_candidate의 위험 프리미엄을 설명하는 H_current의 변수 인덱스를 선택합니다.
        """
        lasso_cv = LassoCV(cv=self.cv, random_state=self.random_state).fit(H_current, g_candidate)
        coef = lasso_cv.coef_
        indices = [i for i, c in enumerate(coef) if np.abs(c) > self.tol]
        return indices, lasso_cv

    def ols_regression_t_value(self, H_selected, g_candidate):
        """
        OLS 회귀: r_bar ~ [H_selected, g_candidate] 회귀를 통해 후보 요인 g_candidate의 계수 t-값을 계산합니다.
        """
        X = np.column_stack((H_selected, g_candidate))
        X = sm.add_constant(X)
        model = sm.OLS(self.r_bar, X).fit()
        t_value = np.abs(model.tvalues[-1])
        return t_value, model

    def post_selection_ols(self, H_final):
        """
        포스트-선택 OLS: 최종적으로 선택된 H_final (시장 요인 + 선택된 후보 요인들)을 사용하여
        전체 모델을 OLS 회귀로 적합합니다.
        
        반환값:
          model: statsmodels의 OLS 결과 모델 (계수, t-값 등 포함)
        """
        X = sm.add_constant(H_final)
        model = sm.OLS(self.r_bar, X).fit()
        return model

    def run(self):
        """
        반복적 두 단계 LASSO 방법을 실행하여, 최종적으로 선택된 후보 요인들을 H에 추가한 최종 H 행렬과
        선택된 후보 요인의 인덱스 및 범주를 반환합니다.
        """
        n, _ = self.H.shape
        k = self.G.shape[1]
        candidate_indices = list(range(k))
        selected_indices = []
        selected_categories = set()
        H_current = self.H.copy()  # 초기 H: 시장 요인 등
        iteration = 0

        while candidate_indices:
            best_t = 0
            best_idx = None

            # 각 후보 요인에 대해 평가
            for idx in candidate_indices:
                g_candidate = self.G[:, idx]
                I1, _ = self.first_stage_lasso(H_current)
                I2, _ = self.second_stage_lasso(g_candidate, H_current)
                I3 = list(set(I1).union(set(I2)))
                if len(I3) == 0:
                    H_selected = np.empty((n, 0))
                else:
                    H_selected = H_current[:, I3]
                t_val, _ = self.ols_regression_t_value(H_selected, g_candidate)
                if t_val > best_t:
                    best_t = t_val
                    best_idx = idx

            # 후보 요인의 t-값이 임계치를 초과하면 해당 후보를 H_current에 추가
            if best_t > self.t_threshold:
                cat = self.candidate_categories[best_idx]
                print(f"Iteration {iteration}: 선택된 후보 인덱스 {best_idx} (범주 {cat}) with t-value {best_t:.2f}")
                H_current = np.column_stack((H_current, self.G[:, best_idx]))
                selected_indices.append(best_idx)
                selected_categories.add(cat)
                # 같은 범주에 속하는 후보들은 모두 제거
                candidate_indices = [idx for idx in candidate_indices if self.candidate_categories[idx] != cat]
                iteration += 1
            else:
                break

        # 최종 H_current를 사용해 포스트-선택 OLS 실행
        final_model = self.post_selection_ols(H_current)
        return H_current, selected_indices, list(selected_categories), final_model

# =============================================================================
# 시나리오 예시: 가상의 데이터를 이용한 글로벌 요인 선택 및 포스트-선택 OLS
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 100      # 테스트 자산(포트폴리오)의 관측치 수
    m = 1        # 초기 H: 예를 들어 CAPM의 시장 요인 (1개의 요인)
    k = 152      # 후보 요인 수 (예: 152개 후보 요인)
    
    # r_bar: (n,) 각 시점의 테스트 자산의 횡단면 평균 수익률 (demeaned된 값)
    r_bar = np.random.randn(n)
    
    # H: 초기 시장 요인 행렬, (n, 1)
    H = np.random.randn(n, m)
    
    # G: 후보 요인 행렬, (n, k)
    G = np.random.randn(n, k)
    
    # candidate_categories: 각 후보 요인에 대해 1부터 13 사이의 임의 범주 할당 (길이 k)
    candidate_categories = np.random.choice(np.arange(1, 14), size=k)
    
    # GlobalFactorSelector 클래스 인스턴스 생성 및 실행
    selector = GlobalFactorSelector(r_bar, H, G, candidate_categories, t_threshold=1.64, cv=5, random_state=42)
    H_final, selected_indices, selected_categories, final_model = selector.run()
    
    print("\n최종 선택된 후보 요인 인덱스:", selected_indices)
    print("선택된 후보 요인 범주:", selected_categories)
    print("최종 H 행렬의 크기:", H_final.shape)
    print("\n포스트-선택 OLS 회귀 결과:")
    print(final_model.summary())
