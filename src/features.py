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







engineer(df):
    """
    preprocessing dataframe
    
    - 임금종사자(근로자만 분석대상으로 추출)
    - 상용근로자만 분석대상으로 추출
    """
    worker_df = df[df['g181sq006'].notnull()]
    
    # 임금종사자 여부
    worker_cond = [
        (worker_df['g181sq006'] == 1),
        (worker_df['g181sq006'] == 2),
        (worker_df
    ]
    
    worker_choices = ['employed','self_employed','non_paid']
    
    worker_df['worker_type'] = np.select(worker_cond, worker_choices,default=None)
    
    # 임금근로자만 추출
    worker_df = worker_df['worker_type'] == 'employed'
    
    
    # 상용 근로자만 추출
    """
    1	상용근로자
    2	임시근로자
    3	일용근로자
    """
    
    # 이직 경험빈도
    
    turnover_cond = [
        (worker_df['g181d001'].isnull()==True),
        (worker_df['g181d001'] == 1),
        
# 다중공성성 확인




# Scalings



# Pipeline
        
        
        
        
        
data = data.drop(config["drop_columns"], axis=1)
# Define X (independent variables) and y (target variable)
X = np.array(data.drop(config["target_name"], 1))
y = np.array(data[config["target_name"]])
# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state= config["random_state"])

