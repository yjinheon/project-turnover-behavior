from sklearn.pipeline import Pipeline

# 사용 패키지 로드
import pandas as pd
import numpy as np
import seaborn as sns

# modeling
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # 오버샘플링
from sklearn.preprocessing import LabelEncoder # 인코딩용
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder, one_hot


# parameter tuning
from sklearn.model_selection import GridSearchCV

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# 사용근로자 관련 feature 리스트

feature_list = [
        'g181d001', # turnover_exp
        'g181d006', # work_exp
        'g181e001', # 전직장 경험
        'g181a021', # 상용근로자 여부
        'g181sq006', # 사용근로자 여부
        'g181pid', # id 
        'g181majorcat', # 전공계열
        'g181sex', # 성별 
        'g181birthy', # 출생년
        'g181age', # 연력
        'g181graduy', # 졸업년
### 사업체 관련 // 현일자리 시작 연월 사용 불가능    
        'g181a001',  # 현일자리 시작년
        'g181a002', # 현일자리  시작월
        'g181a004_10', # 일자리 산업 대분류
        'g181a010', # 기업체 종사자 수
        'g181a011', # 사업체 종사자 수
        'g181a018', # 출근시간_시간
        'g181a019', # 출근시간_분
        'g181a116', # 주당 정규 근로일
        'g181a117', # 주당정규 근로시간
        'g181a118', # 주당초과 근로시간
        'g181a119', # 월평균 후일근로
        'g181a020', # 사업체형태
        'g181a022',     # 정규직 비정규직 여부
### 혜택관련
        'g181a390',     # 일자리 형태 자발, 비자발 여부
        'g181a035' ,    # 교대제 여부
        'g181a038',     # 퇴직금 제공 여부
        'g181a039',     # 제공여부 2- 유급휴가
        'g181a043',     #  6- 육아휴직
        'g181a045',     # 8- 시간 외 수당
        'g181a046',     # 9- 상여금
        'g181a048',     # 11- 유급주휴
        'g181a392',     # 12- 산전후휴가
        'g181a120',     # 급여 형태 구분
        'g181a122',     # 월 평균 근로소득
        'g181a126',     # 만족도-임금
        'g181a127',     # 만족도-고용안정성
        'g181a128',     # 만족도-직무내용
        'g181a129',     # 만족도-근무환경
        'g181a130',     # 만족도-노동시간
        'g181a131',     # 만족도-발전가능성
        'g181a132',     # 만족도-인간관계
        'g181a133',     # 만족도-복리후생
        'g181a134',     # 만족도-인사체계
        'g181a135',     # 만족도-사회적평판-일
        'g181a136',     # 만족도-자율성 및 권한
        'g181a137',     # 만족도-일자리-사회적 평판
        'g181a138',     # 만족도-적성흥미일치도
        'g181a139',     # 만족도-직무관련 교육
        'g181a140',     # 만족도-일자리_전반적만족도
        'g181a141',     # 만족도-업무_전반적만족도
        'g181a142',     # 교육수준-일수준일치정도
        'g181a143',     # 일기술수준-본인기술수준일치정도
        'g181a144',     # 주전공일치정도
        'g181a146',     # 전공지식업무도움정도
        'g181a158',     # 보험-국민연금
        'g181a159',     # 보험-특수직역연금
        'g181a160',     # 보험-건강보험
        'g181a161',     # 보험-고용보험
        'g181a162',     # 보험-산재보험
        'g181a189',     # 구직활동경험기간-개월
        'g181a283',     # 다른일자리제의여부
        'g181a285',     # 적응시어려움여부
        'g181a297',     # 이직준비 여부: target
        'g181g001',     # 대학원 경험유무
        'g181l001',     # 취업훈련경험유무
        'g181q001',     # 현재 견강상태
        'g181q004',     # 흡연여부
        'g181q006',     # 음주빈도
        'g181q015',     # 삶의만족도-개인적 측면
        'g181q016',     # 삶의만족도-관계적 측면
        'g181q017',     # 삶의만족도-소속집단
        'g181q018',     # 감정빈도-즐거운
        'g181q019',     # 감정빈도-행복한
        'g181q020',     # 감정빈도-편안한
        'g181q021',     # 감정빈도-짜증나는
        'g181q022',     # 감정빈도-부정적인
        'g181q023',     # 감정빈도-무기력한
        'g181p001',     # 혼인여부
        'g181p008',     # 부양자녀 유무
        'g181p036',     # 부모님 자산규모
        'g181p046',     # 거주형태
        'g181p041'      # 경제적 지원여부
]


# dictionary for converting feature id to feature name



# subset data

def subset_df(df):
        df.columns = df.columns.str.lower()
        worker_df = df[feature_list]
        worker_df = worker_df[worker_df['g181sq006'].notnull()] # 근로자(일의 종류)

        # 임금종사자 여부
        """
        1	타인 또는 회사에 고용되어 보수(돈)를 받고 일한다.(직장, 아르바이트 등 포함)
        2	내 사업을 한다.
        3	가족의 일을 돈을 받지 않고 돕는다.
        """
        worker_cond = [
        (worker_df['g181sq006'] == 1), 
        (worker_df['g181sq006'] == 2),
        (worker_df['g181sq006'] == 3)
        ]

        worker_choices = ['employed','self_employed','non_paid']

        worker_df['worker_type'] = np.select(worker_cond, worker_choices,default=None)
        worker_df.drop(['g181sq006'], axis=1, inplace=True)

        # 임금근로자만 추출
        worker_df = worker_df[worker_df['worker_type'] == 'employed'] 

        # 상용 근로자만 추출
        """
        1	상용근로자
        2	상용근로자 외 임시,일용직
        """
        regular_cond = [worker_df['g181a021'] == 1,
                        worker_df['g181a021'] != 1]

        worker_df['regular_worker'] = np.select(regular_cond, ['yes','no'],default=None)
        worker_df = worker_df[worker_df['regular_worker'] == 'yes']

        worker_df.drop(['g181a021'], axis=1, inplace=True)

        return worker_df
