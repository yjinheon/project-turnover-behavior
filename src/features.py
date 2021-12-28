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
from category_encoders import OneHotEncoder


# parameter tuning
from sklearn.model_selection import GridSearchCV

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

"""
get_demo(df):
"""


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

feature_dict =  {
      'g181pid':'id', # id 
       'g181majorcat':'majorcat_cat', # 전공계열
       'g181sex':'sex_cat', # 성별   
       'g181birthy':'birth_date', # 출생년
       'g181age':'age_num', # 연령
       'g181graduy':'graduy_date', #졸업년
### 사업체 관련       
        'g181a001':'year_start_date', # 현일자리 시작년
        'g181a002':'month_start_date', # 현일자리 시작월
        'g181a004_10':'ind_cat', # 일자리 산업 대분류
        'g181a010':'corp_worker_cat', # 기업체 종사자 수 # 결측값문제로 categorical로 변경
        'g181a011':'biz_worker_cat', # 사업체 종사자 수
        'g181a018':'tw_hour', # 출근시간_시간
        'g181a019':'tw_min', # 출근시간_분
        'g181a116':'workday_num', # 주당 정규 근로일
        'g181a117':'worktime_num', # 주당정규 근로시간
        'g181a118':'worktime_ex_num', # 주당초과 근로시간
        'g181a119': 'holy_work_num', # 월평균 휴일근로
        'g181a020': 'biztype_cat', # 사업체형태
       'g181a022':'regular_cat',     # 정규직 비정규직 여부
### 혜택관련
        'g181a390':'voluntary_cat',     # 일자리 형태 자발, 비자발 여부
        'g181a035':'shift_cat' ,    # 교대제 여부
        'g181a038':'pension_cat',     # 퇴직금 제공 여부
        'g181a039':'payed_vc_cat',     # 제공여부 2- 유급휴가
        'g181a043':'maternity_cat',     #  6- 육아휴직
        'g181a045':'overtime_pay_cat',     # 8- 시간 외 수당
        'g181a046':'bonus_cat',     # 9- 상여금
        'g181a048':'weekly_hl_cat',     # 11- 유급주휴
        'g181a392':'baby_vc_cat',     # 12- 산전후휴가
        'g181a120':'wage_type_cat',     # 급여 형태 구분
        'g181a122':'month_wage_num',     # 월 평균 근로소득
        'g181a126':'sat_wage_num',     # 만족도-임금
        'g181a127':'sat_stable_num',     # 만족도-고용안정성
        'g181a128':'sat_work_num',     # 만족도-직무내용
        'g181a129':'sat_env_num',     # 만족도-근무환경
        'g181a130':'sat_wt_num',     # 만족도-노동시간
        'g181a131':'sat_potential_num',     # 만족도-발전가능성
        'g181a132':'sat_relation_num',     # 만족도-인간관계
        'g181a133':'sat_welfare_num',     # 만족도-복리후생
        'g181a134':'sat_hr_num',     # 만족도-인사체계
        'g181a135':'sat_rep1_num',     # 만족도-사회적평판-일
        'g181a136':'sat_auto_num',     # 만족도-자율성 및 권한
        'g181a137':'sat_rep2_num',     # 만족도-일자리-사회적 평판
        'g181a138':'sat_fit_num',     # 만족도-적성흥미일치도
        'g181a139':'sat_edu_num',     # 만족도-직무관련 교육
        'g181a140':'sat_general_num',     # 만족도-일자리_전반적만족도
        'g181a141':'sat_work-general_num',     # 만족도-업무_전반적만족도
        'g181a142':'edu-fit_num',     # 교육수준-일수준일치정도
        'g181a143':'skill-fit_num',     # 일기술수준-본인기술수준일치정도
        'g181a144':'major-fit_num',     # 주전공일치정도
        'g181a146':'major_help_num',     # 전공지식업무도움정도
        'g181a158':'ins_1_num',     # 보험-국민연금
        'g181a159':'ins_2_num',     # 보험-특수직역연금
        'g181a160':'ins_3_num',     # 보험-건강보험
        'g181a161':'ins_4_num',     # 보험-고용보험
        'g181a162':'ins_5_num',     # 보험-산재보험
        'g181a189':'seeking_time_num',     # 구직활동경험기간-개월
        'g181a283':'adjust_difficulty_cat',     # 다른일자리제의여부
        'g181a285':'job_offer_cat',     # 적응시어려움여부
        'g181a297':'turnover_intention',     # 이직준비 여부: target
        'g181g001':'graduate_cat',     # 대학원 경험유무
        'g181l001':'train_cat',     # 취업훈련경험유무
        'g181q001':'health_num',     # 현재 견강상태
        'g181q004':'smoke_cat',     # 흡연여부
        'g181q006':'drink_num,',     # 음주빈도
        'g181q015':'lifesat_personal',     # 삶의만족도-개인적 측면
        'g181q016':'lifesat_relational',     # 삶의만족도-관계적 측면
        'g181q017':'lifesat_group',     # 삶의만족도-소속집단
        'g181q018':'emg_joy_num',     # 감정빈도-즐거운
        'g181q019':'emg_happy_num',     # 감정빈도-행복한
        'g181q020':'emg_comfort_num',     # 감정빈도-편안한
        'g181q021':'emb_irr_num',     # 감정빈도-짜증나는
        'g181q022':'emb_negative_num',     # 감정빈도-부정적인
        'g181q023':'emb_spiritless',     # 감정빈도-무기력한
        'g181p001':'marriage_cat',     # 혼인여부
        'g181p008':'child_cat',     # 부양자녀 유무
        'g181p036':'parent_asset_cat',     # 부모님 자산규모
        'g181p046':'livetype_cat',     # 거주형태
        'g181p041':'support_cat',      # 경제적 지원여부
        # subset에서 추가한 변수들
        'worker_type':'worker_type',
        'regular_worker':'regular_worker',
        'turnover_exp':'turnover_exp',
        'work_exp':'work_exp'        
}

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


def engineer(df):
    
    # 이직경험 빈도 추가
    turnover_cond = [
    (df['g181d001'].isnull()==True ), # 첫 직장 전 취업한적 없음(알바포함)
    (df['g181d001']==1) & (df['g181d006']==1), # 알바 경험 있음
    (df['g181d001']==1) & (df['g181d006']==2) & (df['g181e001']==2), # 전직장 있음, 1번이직
    (df['g181e001']== 1)] # 전직장 2개 이상 있음
    choices = [0, 1, 2, 3]

    df['turnover_exp'] = np.select(turnover_cond, choices,default=3)
    
    df.drop(['g181d001','g181d006','g181e001'], axis=1, inplace=True)

    # 알바이외 근로경험 여부
    df['work_exp'] = np.where(df['turnover_exp'].isin([0,1]),0,1)
    
    # 구직활동기간 결측값 0으로 처리
    
    df['g181a189'] = df['g181a189'].fillna(0)
    
    # feature name 변경 (map)
    
    df.columns = df.columns.map(feature_dict)
    
    df = df.drop(df[df['month_wage_num'] == -1].index) # 급여 모르는 경우 제거
    
    df['work_year'] = 2019 - df['year_start_date']  # 근무기간
    
    df['work_time_num'] = df['tw_min'] + df['tw_hour'] # 출근소요시간
    
    # 보험 수
    insurances_col = [col for col in df if col.startswith('ins')]
    df['insurances_num'] = 0
    for col in insurances_col:
        df['temp'] = np.where(df[col] == 1, 1, 0)
        df['insurances_num'] += df['temp']
    
    # 회사 전반적 만족도
    biz_sat_col = [col for col in df if col.startswith('sat')]
    df['biz_sat'] = df[biz_sat_col].sum(axis=1)
    
    # 긍정적 감정
    pos_col = [col for col in df if col.startswith('emg') ] 
    df['pos'] = df[pos_col].sum(axis=1)
    
    # 부정적 감정
    neg_col = [col for col in df if col.startswith('neg') ]
    df['neg'] = df[neg_col].sum(axis=1)
    
    # 삶의 만족도
    lifesat_col = [col for col in df if col.startswith('lifesat')]
    df['lifesat'] = df[lifesat_col].sum(axis=1)
        
    # 혜택 수
    benefit_col  =['pension_cat',     # 퇴직금 제공 여부
        'payed_vc_cat',     # 제공여부 2- 유급휴가
        'maternity_cat',     #  6- 육아휴직
        'overtime_pay_cat',     # 8- 시간 외 수당
        'bonus_cat',     # 9- 상여금
        'weekly_hl_cat',     # 11- 유급주휴
        'baby_vc_cat'] # 출산휴가

    df['benefit_num'] = 0

    for cols in benefit_col:
        df['temp'] = np.where(df[cols]==1, 1, 0)
        df['benefit_num'] += df['temp']
    
    df.drop(columns='temp',axis=1,inplace=True)
    
    return df


# Preprocessing Class 정의




# Scalings



# Pipeline
        
        


"""        
data = data.drop(config["drop_columns"], axis=1)
# Define X (independent variables) and y (target variable)
X = np.array(data.drop(config["target_name"], 1))
y = np.array(data[config["target_name"]])
# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state= config["random_state"])

"""
