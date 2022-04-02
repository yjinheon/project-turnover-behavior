#  데이터프레임 진단용 함수들
#  통계적 진단은 statsmodel 활용
#  정규성 검정 등은 pinguin package 활용
# Reference:
# - https://github.com/slapadasbas


from numpy.lib.function_base import insert
import pandas as pd
import numpy as np
from pandas.core.tools import numeric
import seaborn as sns
import matplotlib.pyplot as plt

from tools.visualization.visualize import Dataplots
from pandas.api.types import is_numeric_dtype

sns.set()

"""

categorical _barbplot

def initial_eda(df):
    # List of categorical columns
    cat_cols = df.select_dtypes('object').columns
    
    for col in cat_cols:
        # Formatting
        column_name = col.title().replace('_', ' ')
        title= 'Distribution of ' + column_name
        
        # Unique values <= 12 to avoid overcrowding
        if len(df[col].value_counts())<=12: 
            plt.figure(figsize = (8, 6))        
            sns.countplot(x=df[col], 
                          data=df, 
                          palette="Paired",
                          order = df[col].value_counts().index)
            plt.title(title, fontsize = 18, pad = 12)
            plt.xlabel(column_name, fontsize = 15)
            plt.xticks(rotation=20)
            plt.ylabel("Frequency",fontsize = 15)
            plt.show();
        else:
            print(f'{column_name} has {len(df[col].value_counts())} unique values. Alternative EDA should be considered.')
    return

"""


class LookData(Dataplots):
    """
    EDA tool class
    - ex) get_demo : demographic data
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    def __init__(self,df) :
        self.df = df
        self.numeric_df = self.df.select_dtypes(include=['number'])
        self.cat_df = self.df.select_dtypes(exclude=['number'])
        
    def dtypes_df(self):
        """
        데이터프레임의 컬럼 타입 및 개수 확인
        보다 심플한 타입별 확인을 위해 사용
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns.to_list()
        cat_cols = self.df.select_dtypes(exclude=['number']).columns.to_list()
        print("*"*30)
        print(f'There are {len(numeric_cols)} numerical features:', '\n')
        print(f"Threre are {len(cat_cols)} categorical features:")
        all_cols = numeric_cols + cat_cols
        al_colname = ["numeric" if col in numeric_cols else "categorical" for col in all_cols]
        
        res = pd.DataFrame({"column": all_cols, "type": al_colname})
        
        return res.sort_values(by=['type'])
    
    def _dtypes(self):
        return pd.DataFrame(self.df.dtypes).rename(columns={0: 'dtype'})
        
    
    def _count_zeros(self,s):
        return len([f for f in s.values if f == 0])
    
    def _count_negative(self,s):
        return len([f for f in s.values if f < 0])

    def _compute_outlier(self,s):
        """ A private function that returns the list of outliers in a pd.Series
        :param df: pd.Series
        The column of the dataframe to be analyzed
        :return: list
        The list of outliers
        :raises: TypeError
        If the datatype of the column is not integer or float
        """

        if not is_numeric_dtype(s):
            raise TypeError("Must pass a column with numeric data type")

        df_ = sorted(s)
        q1, q3 = np.percentile(df_, [25, 75])
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = [x for x in df_ if x < lower_bound or x > upper_bound]

        return outliers


    def _remove_outliers(self,s):
        """ 
        A private function that removes the outliers in a pd.Series
        """
        return s.loc[~s.isin(self._compute_outlier(s))]


    def diagnose(self):
        """
        결측값과 unique value 확인
        """
        
        df_ = self._dtypes().join(pd.DataFrame(self.df.isnull().T.sum(axis=1)).rename(columns={0: 'missing_value_cnt'}))
        df_['missing_value_ratio'] = df_['missing_value_cnt'].apply(lambda x: (x / len(self.df)) * 100)
        df_ = df_.join(pd.DataFrame(self.df.apply(pd.value_counts).T.count(axis=1)).rename(columns={0: 'unique_value_cnt'}))
        df_['unique_value_ratio'] = df_['unique_value_cnt'].apply(lambda x: (x / len(self.df)) * 100)
        
        return df_
    
    def diagnose_numeric(self):
        
        """
        cv(coefficient of variation) : std / mean
        cv는 측정단위가 다른 numeric data의 상대비교를 위해 사용
        """
        numeric_cols=self.numeric_df.columns.to_list()
        n = self._dtypes().join(self.df.describe().T, how='right')
        
        outliers = [len(self._compute_outlier(self.df[col])) for col in numeric_cols]
        n['median'] = np.array([np.median(self.df[col]) for col in numeric_cols])
        n['zeros_cnt'] = np.array([self._count_zeros(self.df[col]) for col in numeric_cols])
        n['negative_cnt'] = np.array([self._count_negative(self.df[col]) for col in numeric_cols])
        n['outliers_cnt'] = np.array(outliers)
        n['count'] = np.array([len(self.df[col]) for col in numeric_cols])
        n['coefficient of variance'] = np.array([(np.std(self.df[col]/np.mean(self.df[col]))) for col in numeric_cols])
        
        return n
    
    def diagnose_categorical(self):
        cat_cols = self.cat_df.columns.to_list()
        dfs = []
        for col in cat_cols:
            x_ = self.df[col].dropna()
            x_ = pd.Series(x_).to_frame() # convert series to dataframe
            x_['variable'] = str(col)  
            # method chaining 을 여러 줄에 걸쳐서
            x_ = x_.groupby(['variable',col]).size() \
                .to_frame() \
                .reset_index() \
                .rename(columns={col : 'levels', 0:'freq'}) 
            dfs.append(x_)
        
        try:
            dt = pd.concat([d for d in dfs])
            dt = dt.reset_index().drop('index',axis=1) 
            
            dt['count'] = self.df.shape[0]
            dt['ratio'] = dt['freq'] / dt['count'] * 100
            dt['rank'] = dt.sort_values(['freq'], ascending=False).groupby('variable').cumcount()+1
            
            return dt
            
            
        except ValueError as v:
            if len(cat_cols) == 0:
                raise Exception("There are no categorical variables in the dataframe")
            else:
                raise Exception(v)
        
    def glimpse(self, maxvals=10, maxlen=110):
        """
        데이터 프레임의 구조확인. dplyr 의 glimpes와 같은 기능
        """
        print('Shape: ', self.df.shape)
    
        def pad(y):
            max_len = max([len(x) for x in y])
            return [x.ljust(max_len) for x in y]
    
        # Column Name
        toprnt = pad(self.df.columns.tolist())
    
        # Column Type
        toprnt = pad([toprnt[i] + ' ' + str(self.df.iloc[:,i].dtype) for i in range(self.df.shape[1])])
    
        # Num NAs
        num_nas = [self.df.iloc[:,i].isnull().sum() for i in range(self.df.shape[1])]
        num_nas_ratio = [int(round(x*100/self.df.shape[0])) for x in num_nas]
        num_nas_str = [str(x) + ' (' + str(y) + '%)' for x,y in zip(num_nas, num_nas_ratio)]
        max_len = max([len(x) for x in num_nas_str])
        num_nas_str = [x.rjust(max_len) for x in num_nas_str]
        toprnt = [x + ' ' + y + ' NAs' for x,y in zip(toprnt, num_nas_str)]
    
        # Separator
        toprnt = [x + ' : ' for x in toprnt]
    
        # Values
        toprnt = [toprnt[i] + ', '.join([str(y) for y in df.iloc[:min([maxvals,self.df.shape[0]]), i]]) for i in range(self.df.shape[1])]
    
        # Trim to maxlen
        toprnt = [x[:min(maxlen, len(x))] for x in toprnt]
    
        for x in toprnt:
            print(x)


    def diagnose_outliers(self):
        
        numeric_cols = [f for f in self.df.columns if 'int' in str(self.df[f].dtype) or 'float' in str(self.df[f].dtype)]
        q = self.df[numeric_cols].count().to_frame().drop(0, axis=1)
        outliers = [self._compute_outlier(self.df[f]) for f in numeric_cols]
        
        q['outliers_cnt'] = [len(x) for x in outliers]
        q['outliers_ratio'] = q['outliers_cnt'].apply(lambda x: (int(x) / len(self.df)) * 100)
        q['outliers_mean'] = [np.mean(x) for x in outliers]
        q['with_outliers_mean'] = [np.mean(self.df[f]) for f in numeric_cols]
        q['without_outliers_mean'] = [np.mean([x for x in zip(self._remove_outliers(self.df[f]))]) for f in numeric_cols]
        q['rate'] = q['outliers_mean'] / q['with_outliers_mean']
        
        return q




if __name__ == "__main__":
    df = sns.load_dataset('iris')
    ins = Lookdata(df)
    diag=ins.diagnose()
    print(diag)
    print(ins.dtypes_df())
    dd=ins.diagnose_outliers()
    print(dd)
    print(ins.diagnose_numeric())
    print(ins.diagnose_categorical())
    ins.plot_outliers("sepal_length")
    ins.plot_outliers("sepal_width")
    ins.plot_outliers_all()