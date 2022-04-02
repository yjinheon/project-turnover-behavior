# Preprocessing Class

class Preprocess():
    """
    X_train, X_test  데이터에 대한 처리
    """
    
    def __init__(self):
        """
        
        """
        print("Preprocessing Class")
        self.numeric_df = None
        self.cat_df = None
        self.ordinal = None
        
    
    def get_dtypes(self,data):
        self.numeric_df = data.select_dtypes(include=['number'])
        self.cat_df = data.select_dtypes(exclude=['number'])
        
                 
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
        data.drop(columns='corp_worker_num',axis=1,inplace=True) 
        data.drop(data[data.month_wage_num==-1].index,inplace=True) # 급여 모르는 경우 제거
    
        return data
    
    def drop_col(self,data,col):
        """
        추가적인 가설검증이나 모델링 결과에 따라 드롭해야할 컬럼이 생길 경우 
        """
        data.drop(columns=col,axis=1,inplace=True)
        
        return data
        
    
    def base_pipline(self,data):
        
        numeric_col = self.numeric_df.columns.tolist()
        cat_col = self.cat_df.columns.tolist()
        one_hot_col = [col for col in cat_col if col not in self.ordinal]
        """

        """
        
        cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(use_cat_names=True,cols=one))
            ('ordial', OrdinalEncoder(cols=self.ordinal_col))))')
        ])
        num_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', MinMaxScaler())
        ])
        preprocess_pipeline = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numeric_col),
                ('cat', cat_pipeline, cat_col),
            ]
        )
        data = preprocess_pipeline.fit_transform(data)
        return data
            
        



class SelectPreprocessor():
    """
    Preprocess strategies defined and exected in this class
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.pipeline import Pipeline
    

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
        """
        drop_strategy = {'PassengerId': 1,  # 1 indicate axis 1(column)
                         'Cabin': 1,
                         'Ticket': 1}
        self.data = self._preprocessor.drop(self.data, drop_strategy)

        fill_strategy = {'Age': 'Median',
                         'Fare': 'Median',
                         'Embarked': 'Mode'}
        self.data = self._preprocessor.fillna(self.data, fill_strategy)

        self.data = self._preprocessor.feature_engineering(self.data, 1)


        self.data = self._preprocessor._label_encoder(self.data)

        """
        self._preprocessor.base_pipline()
        
    def _strategy1(self):
        self._base_strategy()

        self._preprocessor.drop_col(self.data,'corp_worker_num') #   data.drop(columns='corp_worker_num',axis=1,inplace=True) 
        

    def _strategy2(self):
        self._base_strategy()

        self.data=self._preprocessor._get_dummies(self.data,
                                        prefered_columns=None)#None mean that all feature will be dummied

