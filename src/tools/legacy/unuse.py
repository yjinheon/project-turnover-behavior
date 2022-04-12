# base pipeline 
     """
        - categorical_feature imputation 고려
        - OneHotEncoder 
            - handle_unknown = 'ignore' 옵션 -> specifically useful if you don't know all possible categories
            - sparse = False 옵션 -> 기본값은 True. 리턴값을 sparse matrix에서 array로 변환
            
                
        ohe_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=-2)),
            ('one-hot', OneHotEncoder(use_cat_names=True, handle_unknown='ignore'))
        ])
        
        ord_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=-2)),
            ('ordinal', OrdinalEncoder(handle_unknown='ignore'))
        ])
        
        num_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ])
        
        preprocess_pipeline = ColumnTransformer(
            transformers=[
                ('ohe', ohe_pipeline, self.onehot_col),
                ('ord', ord_pipeline, self.ordinal_col),
                ('num', num_pipeline, self.numeric_col)
                ]
        )
        
        return preprocess_pipeline.fit_transform(data)

        """
        #make_pipeline(StandardScaler(), GaussianNB(priors=None))


def ce_pipeline(self,data):
        """
        category encode를 활용한 전처리 파이프라인
        
        encoder_list = [ce.backward_difference.BackwardDifferenceEncoder, 
               ce.basen.BaseNEncoder,
               ce.binary.BinaryEncoder,
                ce.cat_boost.CatBoostEncoder,
                ce.hashing.HashingEncoder,
                ce.helmert.HelmertEncoder,
                ce.james_stein.JamesSteinEncoder,
                ce.one_hot.OneHotEncoder,
                ce.leave_one_out.LeaveOneOutEncoder,
                ce.m_estimate.MEstimateEncoder,
                ce.ordinal.OrdinalEncoder,
                ce.polynomial.PolynomialEncoder,
                ce.sum_coding.SumEncoder,
                ce.target_encoder.TargetEncoder,
                ce.woe.WOEEncoder
                ]
for encoder in encoder_list:
    
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('woe', encoder())])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
    
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=500))])
    
    model = pipe.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(encoder)
    print(f1_score(y_test, y_pred, average='macro'))
        """
        pass