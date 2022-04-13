feature_dict =  {
        'g181pid':'id', # id 
        'g181majorcat':'majorcat', # 전공계열
        'g181sex':'sex_cat', # 성별   
        'g181birthy':'birth_year', # 출생년
        'g181age':'age_num', # 연령
        'g181graduy':'gradu_year', #졸업년
### 사업체 관련       
        'g181a001':'year_start_date', # 현일자리 시작년
        'g181a002':'month_start_date', # 현일자리 시작월
        'g181a004_10':'ind_cat', # 일자리 산업 대분류
        'g181a010':'corp_worker_cat', # 기업체 종사자 수 # 결측값문제로 categorical로 변경
        'g181a011':'biz_worker_num', # 사업체 종사자 수
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
        'g181p036':'parent_asset_num',     # 부모님 자산규모
        'g181p046':'livetype_cat',     # 거주형태
        'g181p041':'support_cat',      # 경제적 지원여부
        # subset에서 추가한 변수들
        'worker_type':'worker_type_cat',
        'regular_worker':'regular_worker_cat',
        'turnover_exp':'turnover_exp_num',
        'work_exp':'work_exp_cat'        
}

import numpy as np
# make pipeline
from sklearn.pipeline import make_pipeline

class Preprocess():
    """
    X_train, X_test
    """
    
    def __init__(self):
        """
        
        """
        print("Preprocessing Class")
        self.numeric_df = None
        self.cat_df = None
            
    def get_dtypes(self,data):
        # onehot 할 피처
        self.onehot_col = [col for col in data.columns.to_list() if col.endswith('cat') ] 
        self.numeric_col = [col for col in data.columns.to_list() if col not in self.onehot_col]
        
    def col_type_df(self,data):
        all_cols = [col for col in data.columns.to_list()]
        df = pd.DataFrame({"col":all_cols})
        
        col_cond = [
            (df['col'] in self.onehot_col) ,
            (df['col'] in self.numeric_col),
            (df['col'] in self.ordinal_col)
        ]
        col_type = ["category","numeric","ordinal"]
        df["type"] = np.select(col_cond,col_type,default="other")
                
        return df
    
    def get_ordinal_col(self,data):
        self.ordinal_col = [col for col in data.columns.to_list() if col.endswith('ordinal') ]
        
    def engineer(self,data):
        
        # id 칼럼 제거
        data = data.drop(['g181pid'], axis=1)
    
        # 이직경험 빈도 추가
        turnover_cond = [
        (data['g181d001'].isnull()==True ), # 첫 직장 전 취업한적 없음(알바포함)
        (data['g181d001']==1) & (data['g181d006']==1), # 알바 경험 있음
        (data['g181d001']==1) & (data['g181d006']==2) & (data['g181e001']==2), # 전직장 있음, 1번이직
        (data['g181e001']== 1)] # 전직장 2개 이상 있음
        choices = [0, 1, 2, 3]

        data['turnover_exp'] = np.select(turnover_cond, choices,default=3)
    
        data.drop(['g181d001','g181d006','g181e001'], axis=1, inplace=True)

        # 알바이외 근로경험 여부
        data['work_exp'] = np.where(data['turnover_exp'].isin([0,1]),0,1)
    
        # 구직활동기간 결측값 0으로 처리
    
        data['g181a189'] = data['g181a189'].fillna(0)
    
        # feature name 변경 (map)
    
        data.columns = data.columns.map(feature_dict)
    
    
        data = data.drop(data[data['month_wage_num'] == -1].index) # 급여 모르는 경우 제거
    
        data['work_year'] = 2019 - data['year_start_date']  # 근무기간
    
        data['work_time_num'] = data['tw_min'] + data['tw_hour'] # 출근소요시간
    
        # 보험 수
        insurances_col = [col for col in data if col.startswith('ins')]
        data['insurances_num'] = 0
        for col in insurances_col:
            data['temp'] = np.where(data[col] == 1, 1, 0)
            data['insurances_num'] += data['temp']
    
        # 회사 전반적 만족도
        biz_sat_col = [col for col in data if col.startswith('sat')]
        data['biz_sat'] = data[biz_sat_col].sum(axis=1)
    
        # 긍정적 감정
        pos_col = [col for col in data if col.startswith('emg') ] 
        data['pos'] = data[pos_col].sum(axis=1)
    
        # 부정적 감정
        neg_col = [col for col in data if col.startswith('neg') ]
        data['neg'] = data[neg_col].sum(axis=1)
    
        # 삶의 만족도
        lifesat_col = [col for col in data if col.startswith('lifesat')]
        data['lifesat'] = data[lifesat_col].sum(axis=1)
        
        # 혜택 수
        benefit_col  =['pension_cat',     # 퇴직금 제공 여부
            'payed_vc_cat',     # 제공여부 2- 유급휴가
            'maternity_cat',     #  6- 육아휴직
            'overtime_pay_cat',     # 8- 시간 외 수당
            'bonus_cat',     # 9- 상여금
            'weekly_hl_cat',     # 11- 유급주휴
            'baby_vc_cat'] # 출산휴가

        data['benefit_num'] = 0

        for cols in benefit_col:
            data['temp'] = np.where(data[cols]==1, 1, 0)
            data['benefit_num'] += data['temp']
        
        # 결측값 0으로 처리(구직활동기간이 0이기에)    
        data['seeking_time_num'] = data['seeking_time_num'].fillna(0)
    
        data.drop(columns='temp',axis=1,inplace=True) # temp 삭제
        data.drop(data[data.month_wage_num==-1].index,inplace=True) # 급여 모르는 경우 제거
        
        # remove date
        data.drop(columns=['birth_year','gradu_year'],axis=1,inplace=True) # 구직활동기간 결측값 제거
        
        # -1 이슈로 삭제
        data.drop(columns = ['biz_worker_num'],axis=1,inplace=True) 
        
        # data quality 이슈로 삭제
        data.drop(columns = ['month_wage_num'],axis=1,inplace=True)
        
    
        return data
    
    def drop_col(self,data,remove_col):
        
        all_cols = [self.numeric_col,self.onehot_col]
        
        for col in all_cols:
            if remove_col in col:
                col.remove(remove_col)
            else:
                continue
        
        data.drop(columns = remove_col,axis=1,inplace=True)
        
        return data
    

    def base_pipline(self,data):
                
        #ohe_pipeline = make_pipeline(OneHotEncoder(use_cat_names=True, handle_unknown='ignore'),SimpleImputer(strategy='constant', fill_value='missing'))
        #ord_pipeline = make_pipeline(OrdinalEncoder(handle_unknown='ignore'),SimpleImputer(strategy='constant', fill_value=-2))
        #num_pipeline = make_pipeline(StandardScaler(),SimpleImputer(strategy='median'))
        
        ohe_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        #ord_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
        #                            ('ord', OrdinalEncoder())])
                
        num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])
        
        preprocess_pipeline = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numeric_col),
                ('ohe', ohe_pipeline, self.onehot_col),
            ]
        )
        
        #preprocess_pipeline
    
        return preprocess_pipeline
    
        
class PreprocessSelector():
    """
    전처리 방식 선택
    """
    def __init__(self):
        self.data=None
        self._preprocessor=Preprocess()

    def strategy(self, data, strategy_type="strategy1"):
        self.data=data
        if strategy_type=='strategy1':
            self._strategy1()
        elif strategy_type=='strategy2':
            self._strategy2()

        return self.data

    def _base_strategy(self):
        self.data=self._preprocessor.engineer(self.data)
        self._preprocessor.get_dtypes(self.data)

    def _strategy1(self):

        self._base_strategy()
        self.data=self._preprocessor.drop_col(self.data,"corp_worker_cat")
        self.data=self._preprocessor.base_pipline(self.data)

    def _strategy2(self):
        """
        remove features
        """
        self._base_strategy()
        #self.data = self._preprocessor.drop_col(self.data,"corp_worker_cat")
        self.daa = self._preprocessor.drop_col(self.data,"") 
        
        
# preprocess_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler , OrdinalEncoder

SimpleImputer.get_feature_names_out = (lambda self, names=None: self.feature_names_in_)
#OrdinalEncoder.get_feature_names_out = (lambda self, names=None: self.feature_names_in_) 왜 안먹힘?

def base_pipline(data):
        #OrdinalEncoder.get_feature_names_out = (lambda self, names=None: self.feature_names_in_)
        #ohe_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(use_cat_names=True, handle_unknown='ignore'))
        #ord_pipeline = make_pipeline(SimpleImputer(strategy='median'),OrdinalEncoder(handle_unknown='ignore'))
        #num_pipeline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
        
        ohe_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        ord_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                    ('ord', OrdinalEncoder())])        
        num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])
        
        
        preprocess_pipeline = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numeric_col),
                ('ohe', ohe_pipeline, onehot_col),
                ('ord', ord_pipeline, ordinal_col)
            ]
        )
        
        
        return preprocess_pipeline
    
